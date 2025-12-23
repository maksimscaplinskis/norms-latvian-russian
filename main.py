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
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.soniox.stt import SonioxInputParams, SonioxSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.frames.frames import LLMTextFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.interruptions.min_words_interruption_strategy import MinWordsInterruptionStrategy

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-pipecat")

app = FastAPI()

PIPELINE_SAMPLE_RATE = 8000

SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
SONIOX_MODEL = os.getenv("SONIOX_MODEL", "stt-rt-v3")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")

SYSTEM_PROMPT = """
You are a bilingual (Russian/Latvian) phone agent for a dental clinic. Keep replies concise (1–2 sentences) and stay in the caller's language (Russian if caller speaks Russian, Latvian if caller speaks Latvian). Do not mix languages in one answer. Collect the visit reason and offer to book. If caller asks for address/hours/services, answer briefly and offer to book. If emergency (heavy bleeding, trouble breathing, strong swelling), say to call emergency number 112.
""".strip()

vad_params = VADParams(
    confidence=0.6,   # стартуйте с 0.6-0.75
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
        audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
        audio_out_channels=1,
        audio_in_channels=1,
        vad_enabled=True,
        vad_analyzer=vad_analyzer,
        vad_audio_passthrough=True,
    )

    return FastAPIWebsocketTransport(websocket=websocket, params=params)


def build_services():
    if not SONIOX_API_KEY:
        raise RuntimeError("SONIOX_API_KEY is not set")
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        raise RuntimeError("ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID is not set")

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

    llm = OpenAILLMService(model=OPENAI_MODEL)
    context = LLMContext([{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(context)

    class LoggingElevenLabsTTSService(ElevenLabsTTSService):
        async def run_tts(self, text: str):
            logger.info("GPT reply -> TTS: %s", text)
            started = False
            try:
                async for frame in super().run_tts(text):
                    if isinstance(frame, TTSAudioRawFrame) and not started:
                        logger.info("ElevenLabs started streaming audio")
                        started = True
                    yield frame
            finally:
                logger.info("ElevenLabs finished for text")

    tts = LoggingElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        model=ELEVENLABS_MODEL_ID,
        sample_rate=PIPELINE_SAMPLE_RATE,
    )

    return stt, llm, context_aggregator, tts


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
        audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
        enable_heartbeats=False,
        allow_interruptions=True,
        # interruption_strategies=[MinWordsInterruptionStrategy(min_words=2)],
    )

    task = PipelineTask(pipeline, params=params, observers=[FrameLogObserver()])
    runner = PipelineRunner(handle_sigint=False, handle_sigterm=False)
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
