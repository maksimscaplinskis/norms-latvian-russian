import os
import logging

from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.responses import Response
from starlette.websockets import WebSocketDisconnect

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.soniox.stt import SonioxInputParams, SonioxSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.frames.frames import LLMTextFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.interruptions.min_words_interruption_strategy import MinWordsInterruptionStrategy
from pipecat.frames.frames import TTSSpeakFrame, TTSUpdateSettingsFrame

import re

INTRO_LV = "Labdien, AM Dental Studio, kā varu palīdzēt?"

CYR = re.compile(r"[А-Яа-яЁё]")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-pipecat")

app = FastAPI()

IN_SR = 8000
OUT_SR = 24000

PIPELINE_SAMPLE_RATE = IN_SR

silence_80ms = b"\x00" * int(OUT_SR * 0.08 * 2)  # 2 bytes/sample (pcm16)

SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
SONIOX_MODEL = os.getenv("SONIOX_MODEL", "stt-rt-v3")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")

GOOGLE_TTS_VOICE_LV = os.getenv("GOOGLE_TTS_VOICE_LV", "lv-LV-Chirp3-HD-Algenib")
GOOGLE_TTS_VOICE_RU = os.getenv("GOOGLE_TTS_VOICE_RU", "ru-RU-Chirp3-HD-Algenib")

SYSTEM_PROMPT = """
    You are the AI VOICE receptionist for AM Dental Studio, a dental clinic.

    GOAL
    Understand the caller’s need and, when appropriate, book a visit.
    You are not a doctor: do not diagnose or provide treatment plans; the dentist evaluates in-person.

    LANGUAGE (LV/RU) — IMPORTANT
    - Do NOT ask “which language do you prefer”.
    - Never mix Latvian and Russian in the same answer.

    VOICE STYLE (STRICT)
    - One sentence only, ideally 12–14 words.
    - Ask only ONE question per turn.
    - No repetition; if no answer, rephrase only once.
    - No greetings (assume the greeting already happened).
    - Never say “write”; say “please tell/say”.
    - Never ask for a phone number.
    - Dont use ":", return time with simple space "14 00"

    CLINIC INFO (ONLY if asked)
    - Hours: Mon–Fri 08:00–16:00; weekends — by appointment.
    - Services: treatment, hygiene, prosthetics, implants, surgery, aligners/caps, adults and children.
    - Prices: “Exact price depends on your case; the dentist will confirm at consultation.”
    - Address: Rēzekne, Latgales iela 93.

    SPECIAL STT RULE
    If caller says “Mani sauc Zobs / mani sauc zobs”, interpret as “Man sāp zobs” (toothache).

    BOOKING LOGIC (highest priority)
    - If the caller asks to book immediately (e.g., “pierakstiet mani / gribu pierakstīties” or “запишите меня / хочу записаться”),
    never ask “do you want to book?” — immediately start booking.

    FLOW (one question per turn)
    1) If reason is unknown: ask the reason. If already stated, do not ask again.
    2) If caller asks only for info (address/hours/services/prices): answer briefly AND in the same sentence gently offer a visit.
    3) If reason is known and booking not offered yet: offer a visit with one question.
    If caller asks to book at any time: go to booking steps (no re-offer).
    - IMPORTANT: You must collect the visit reason. If the caller did NOT state a reason, ask for it once BEFORE booking and offering time options.
    4) Booking steps:
    a) Ask first name + last name.
    b) Ask desired date.
    c) Offer 2 time options within 08 00–16 00 and ask which fits.
    d) If chosen time is taken: offer another time same date.
    e) If no times on that date: offer the nearest next available date/time.
    5) Closing: confirm date, time, reason, “AM Dental Studio”, and end politely.

    SAFETY
    If severe swelling, breathing difficulty, or uncontrolled bleeding: say to call emergency (112).
""".strip()

vad_params = VADParams(
    confidence=0.65,   # стартуйте с 0.6-0.75
    start_secs=0.2,   # сколько речи нужно, чтобы считать "начал говорить"
    stop_secs=0.4,    # сколько тишины, чтобы считать "закончил"
    min_volume=0.5,   # полезно на телефонии
)

vad_analyzer = SileroVADAnalyzer(sample_rate=PIPELINE_SAMPLE_RATE, params=vad_params)

def twilio_serializer_params() -> TwilioFrameSerializer.InputParams:
    """
    Build serializer params to keep Twilio audio at 8kHz and avoid forced hangup
    if Twilio creds are not present.
    """
    auto_hang_up = bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"))
    return TwilioFrameSerializer.InputParams(
        twilio_sample_rate=PIPELINE_SAMPLE_RATE,
        sample_rate=PIPELINE_SAMPLE_RATE,
        auto_hang_up=auto_hang_up,
    )


async def build_transport(websocket: WebSocket) -> FastAPIWebsocketTransport:
    transport_type, call_data = await parse_telephony_websocket(websocket)
    if transport_type != "twilio":
        raise RuntimeError(f"Unsupported transport type: {transport_type}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        params=twilio_serializer_params(),
    )

    params = FastAPIWebsocketParams(
        serializer=serializer,
        add_wav_header=False,
        session_timeout=None,
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
        audio_out_sample_rate=OUT_SR,
        audio_out_channels=1,
        audio_in_channels=1,
        vad_analyzer=vad_analyzer,
    )

    return FastAPIWebsocketTransport(websocket=websocket, params=params)


def build_services():
    if not SONIOX_API_KEY:
        raise RuntimeError("SONIOX_API_KEY is not set")
    # if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
    #     raise RuntimeError("ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID is not set")

    stt = SonioxSTTService(
        api_key=SONIOX_API_KEY,
        sample_rate=PIPELINE_SAMPLE_RATE,
        params=SonioxInputParams(
            model=SONIOX_MODEL,
            audio_format="pcm_s16le",
            num_channels=1,
            language_hints=[Language.RU, Language.LV],
            enable_language_identification=True,
            client_reference_id="twilio-pipecat",
        ),
    )

    llm = OpenAILLMService(
        model=OPENAI_MODEL,
        params=BaseOpenAILLMService.InputParams(
            max_completion_tokens=96,
            temperature=0.2,
            top_p=1.0,
            seed=42,                   # опционально: детерминизм
            extra={
                "reasoning_effort": "none",
                "verbosity": "low",
            },
        ),
    )
    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(context)

    # class LoggingElevenLabsTTSService(ElevenLabsTTSService):
    #     async def run_tts(self, text: str):
    #         logger.info("GPT reply -> TTS: %s", text)
    #         started = False
    #         try:
    #             async for frame in super().run_tts(text):
    #                 if isinstance(frame, TTSAudioRawFrame) and not started:
    #                     logger.info("ElevenLabs started streaming audio")
    #                     started = True
    #                 yield frame
    #         finally:
    #             logger.info("ElevenLabs finished for text")

    # tts = LoggingElevenLabsTTSService(
    #     api_key=ELEVENLABS_API_KEY,
    #     voice_id=ELEVENLABS_VOICE_ID,
    #     model=ELEVENLABS_MODEL_ID,
    #     sample_rate=PIPELINE_SAMPLE_RATE,
    # )

    tts = LoggingGoogleTTSService(
        credentials=os.environ["GCP_SA_JSON"],
        voice_lv=GOOGLE_TTS_VOICE_LV,
        voice_ru=GOOGLE_TTS_VOICE_RU,
        sample_rate=OUT_SR,
        params=GoogleTTSService.InputParams(language=Language.LV, speaking_rate=1),
    )

    return stt, llm, context_aggregator, tts

class LoggingGoogleTTSService(GoogleTTSService):
    @property
    def chunk_size(self) -> int:
        chunk_seconds = float(os.getenv("GOOGLE_TTS_CHUNK_SECONDS", "1.0"))
        # sample_rate становится >0 после start(); до этого можно опереться на _init_sample_rate
        sr = self.sample_rate or getattr(self, "_init_sample_rate", 0) or OUT_SR
        return int(sr * chunk_seconds * 2)  # 2 bytes/sample (pcm16)
    
    def __init__(self, *, voice_lv: str, voice_ru: str, **kwargs):
        super().__init__(voice_id=voice_lv, **kwargs)
        self.voice_lv = voice_lv
        self.voice_ru = voice_ru

    async def run_tts(self, text: str):
        if CYR.search(text):
            self.set_voice(self.voice_ru)
            await self._update_settings({"language": Language.RU})
        else:
            self.set_voice(self.voice_lv)
            await self._update_settings({"language": Language.LV})

        logger.info("GPT reply -> Google TTS: %s", text)

        started = False
        async for frame in super().run_tts(text):
            if isinstance(frame, TTSAudioRawFrame) and not started:
                logger.info("Google TTS started streaming audio")
                started = True
            yield frame
        logger.info("Google TTS finished for text")

class FrameLogObserver(BaseObserver):
    """Logs key frames for GPT reply and TTS lifecycle."""

    def __init__(self):
        super().__init__()
        self._tts_started = False

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, LLMTextFrame):
            logger.info("GPT reply frame: %s", getattr(frame, "text", ""))
        if isinstance(frame, TTSAudioRawFrame) and not self._tts_started:
            self._tts_started = True
            logger.info("ElevenLabs started streaming audio (first chunk %d bytes)", len(frame.audio))
        if isinstance(frame, TTSStoppedFrame):
            logger.info("ElevenLabs finished (TTSStoppedFrame)")
            self._tts_started = False

def build_pipeline(
    transport: FastAPIWebsocketTransport,
) -> tuple[PipelineRunner, PipelineTask]:
    stt, llm, context_aggregator, tts = build_services()

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    params = PipelineParams(
        audio_in_sample_rate=PIPELINE_SAMPLE_RATE,
        audio_out_sample_rate=OUT_SR,
        enable_heartbeats=False,
        allow_interruptions=True,
        # interruption_strategies=[MinWordsInterruptionStrategy(min_words=2)],
    )

    task = PipelineTask(pipeline, params=params, observers=[FrameLogObserver()])
    runner = PipelineRunner(handle_sigint=False, handle_sigterm=False)

    @transport.event_handler("on_client_connected")
    async def _on_client_connected(_transport, _client):
        await task.queue_frames([
            TTSAudioRawFrame(silence_80ms, OUT_SR, 1),
            TTSUpdateSettingsFrame({"language": Language.LV}),
            TTSSpeakFrame(INTRO_LV),
        ])

    return runner, task


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/twilio/voice")
async def twilio_voice(
    request: Request,
    CallSid: str = Form(None),
    From: str = Form(None),
    To: str = Form(None),
):
    logger.info("Incoming call: CallSid=%s, From=%s, To=%s", CallSid, From, To)

    # Важно: host должен быть публичным (ngrok/домен), с портом если он есть
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.hostname
        or "localhost"
    )

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/twilio-stream" track="inbound_track"/>
  </Connect>
</Response>"""

    return Response(content=twiml.strip(), media_type="text/xml")


@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    logger.info("Twilio WebSocket connected")

    try:
        transport = await build_transport(ws)
        runner, task = build_pipeline(transport)
    except Exception as e:
        logger.exception("Failed to initialize pipeline: %s", e)
        await ws.close()
        return

    try:
        await runner.run(task)
    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.exception("Error while running pipeline: %s", e)
        try:
            await task.cancel(reason=str(e))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("twilio_stream handler finished")
