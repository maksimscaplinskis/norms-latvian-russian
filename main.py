import os
import json
import base64
import logging
import asyncio
import threading

from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.responses import Response
from starlette.websockets import WebSocketDisconnect

import websockets
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-soniox-openai-eleven")

SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# Максимально простой промпт под автосервис
SYSTEM_PROMPT = (
    """Tu esi laipns un uzmanīgs AI administrators zobārstniecības klīnikai AM Dental Studio.

    Tavs mērķis – kā dzīvs administrators saprast pacienta vajadzību un, ja vien tas ir saprātīgi, maigi novest līdz pierakstam vizītei.

    KLĪNIKA
    – AM Dental Studio, Rēzekne, Latgales iela 93.
    – Darba laiks: P–Pk 08:00–16:00, brīvdienās – pēc pieraksta.
    – Strādā ar pieaugušajiem un bērniem, piedāvā ārstēšanu, higiēnu, protezēšanu, implantus, ķirurģiju un zobu taisnošanu ar kapēm.

    VALODA
    – Pēc pirmajiem pacienta vārdiem nosaki: latviešu vai krievu.
    – Atbildi stingri tajā pašā valodā, līdz pacients pats lūdz mainīt.
    – Nerunā sarežģīti, bez liekas medicīniskas terminoloģijas.

    STILS
    – Ļoti īsas atbildes: 1–2 teikumi.
    – Vienmēr tikai viens jautājums vienlaicīgi.
    – Neatkārto savus iepriekšējos teikumus vārds vārdā.
    – Nesāc ar „Labdien/Здравствуйте”, ja saruna jau notiek – ej uzreiz pie lietas.

    GALVENĀ SARUNAS KĶĒDE
    1) Noskaidro mērķi:
    – „Pastāstiet, lūdzu, ar ko varam jums palīdzēt?” / krieviski ekvivalents.
    – Saprot, vai tas ir: akūtas sāpes, plānota ārstēšana/plombe, higiēna/balināšana, bērns, implanti/protezēšana, zobu taisnošana, vai tikai jautājums (cena, adrese, darba laiks u.c.).

    2) Piedāvā vizīti (ja nav neatliekams gadījums):
    – Īsa informācija + maigs piedāvājums:
    – „Mēs varam piedāvāt konsultāciju pie ārsta, lai izvērtētu situāciju un sastādītu plānu. Vai vēlaties pierakstīties uz vizīti?”
    – Ja pacients jau pats saka, ka grib pierakstīties, šo jautājumu vairs neuzdod – ej uz datu savākšanu.

    3) Ja pacients piekrīt pierakstam, savāc datus pa vienam jautājumam:
    – Vārds un, ja iespējams, uzvārds.
    – Īss problēmas apraksts (piemēram, „sāpes apakšējā labajā zobā divas dienas”).
    – Vēlamais dienas laiks: rīts, diena vai pēcpusdiena, un aptuvenā diena.
    – „Vai vēlaties pierakstīties pie kāda konkrēta ārsta, vai tas nav būtiski?”

    4) Piedāvā konkrētāku laiku:
    – Ņem vērā: P–Pk 08–16, brīvdienās – pēc pieraksta.
    – Piedāvā vienu konkrētu variantu („trešdien ap 10:00”). Ja neder – piedāvā citu, arī pa vienam.

    5) Kopsavilkums un noslēgums:
    – „Apstiprinu: [datums], ap [laiks], [īss pakalpojums/problēma] AM Dental Studio, Latgales iela 93, Rēzekne.”
    – Pateicies par zvanu/ziņu un pieklājīgi atvadies.

    INFORMĀCIJA BEZ DIAGNOZES
    – Ja jautā tikai par adresi, darba laiku, pakalpojumiem vai cenām – atbildi īsi un skaidri.
    – Par cenām: „Precīza cena atkarīga no situācijas, to vislabāk pateiks ārsts konsultācijā. Mēs varam piedāvāt laiku vizītei.”
    – Tu neesi ārsts: nestādi diagnozes un nesniedz detalizētu ārstēšanas shēmu. Sarežģītos jautājumos saki, ka to izvērtēs ārsts vizītē, un piedāvā pierakstu.

    DROŠĪBA
    – Ja pacients apraksta ļoti smagus simptomus (izteikta sejas tūska, augsta temperatūra, elpošanas grūtības, smaga trauma),
    iesaki nekavējoties vērsties neatliekamajā palīdzībā (113) vai pie dežūrārsta, negaidot plānveida vizīti.
    """
)

GREETING_TEXT = "Labdien, AM Dental Studio. Kā varu palīdzēt?"

# ============================
#   STT: Soniox
# ============================

class SttSession:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws: websockets.WebSocketClientProtocol | None = None

    async def connect(self):
        if not self.api_key:
            raise RuntimeError("SONIOX_API_KEY is not set")

        # WebSocket Soniox
        self.ws = await websockets.connect("wss://stt-rt.soniox.com/transcribe-websocket")
        logger.info("Connected to Soniox WebSocket")

        # Конфиг для телефонного zvana (mulaw 8kHz, endpoint detection + LID) :contentReference[oaicite:2]{index=2}
        config_msg = {
            "api_key": self.api_key,
            "model": "stt-rt-preview",
            "audio_format": "mulaw",
            "sample_rate": 8000,
            "num_channels": 1,
            "enable_language_identification": True,
            "language_hints": ["ru", "lv"],
            "enable_endpoint_detection": True, #TODO:Luche postavitj False
            "client_reference_id": "twilio-call",
        }
        await self.ws.send(json.dumps(config_msg))
        logger.info("Sent Soniox config")

    async def send_audio(self, audio_bytes: bytes):
        if self.ws:
            await self.ws.send(audio_bytes)

    async def finalize(self):
        """Сообщаем Soniox, что аудио больше не будет."""
        if self.ws:
            try:
                await self.ws.send(b"")
            except Exception:
                pass

    async def receive_loop(self, handler):
        """Читаем все сообщения от Soniox и передаём в handler."""
        if not self.ws:
            return
        try:
            async for message in self.ws:
                try:
                    resp = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON Soniox message: %r", message)
                    continue
                await handler(resp)
        except Exception as e:
            logger.exception("Error in SttSession.receive_loop: %s", e)

    async def close(self):
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


# ============================
#   TTS: ElevenLabs -> Twilio
# ============================

class TtsSession:
    def __init__(self, eleven_client: ElevenLabs | None, ws: WebSocket, loop: asyncio.AbstractEventLoop):
        self.eleven_client = eleven_client
        self.ws = ws
        self.loop = loop
        self.stream_sid: str | None = None
        self._cancel_event = threading.Event()
        self._active = False

    def set_stream_sid(self, sid: str | None):
        self.stream_sid = sid

    def is_active(self) -> bool:
        return self._active

    def cancel(self):
        """Запросить остановку текущего TTS-стрима (для barge-in)."""
        if self._active:
            logger.info("TTS cancel requested")
        self._cancel_event.set()

    async def speak(self, text: str):
        """Стрим ElevenLabs TTS обратно в Twilio через media-сообщения."""
        if not self.eleven_client:
            logger.warning("ELEVENLABS_API_KEY is not set, skip TTS")
            return
        if not self.stream_sid:
            logger.warning("streamSid is not set, skip TTS")
            return
        if not text.strip():
            return

        # сбрасываем cancel и помечаем, что сейчас говорим
        self._cancel_event.clear()
        self._active = True
        logger.info("TTS start, text='%s'", text)

        def _run():
            logger.info("TTS thread started")
            try:
                response = self.eleven_client.text_to_speech.stream(
                    voice_id=ELEVENLABS_VOICE_ID,
                    model_id=ELEVENLABS_MODEL_ID,
                    text=text,
                    output_format="ulaw_8000",
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.0,
                        style=0.0,
                        use_speaker_boost=True,
                        speed=1.5,
                    ),
                )
                for chunk in response:
                    if self._cancel_event.is_set():
                        logger.info("TTS streaming cancelled mid-stream")
                        break
                    if not chunk:
                        continue
                    if not isinstance(chunk, (bytes, bytearray)):
                        continue

                    payload = base64.b64encode(chunk).decode("ascii")
                    msg = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": payload},
                    }
                    fut = asyncio.run_coroutine_threadsafe(
                        self.ws.send_text(json.dumps(msg)),
                        self.loop,
                    )
                    try:
                        fut.result()
                    except Exception as e:
                        logger.exception("Error sending audio to Twilio: %s", e)
                        break
            except Exception as e:
                logger.exception("Error in ElevenLabs TTS streaming: %s", e)
            finally:
                self._active = False
                logger.info("TTS thread finished")

        # блокирующий TTS — в отдельном thread
        await asyncio.to_thread(_run)


# ============================
#   CallSession: Twilio + логика
# ============================

class CallSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.loop = asyncio.get_running_loop()
        self.stream_sid: str | None = None
        self.stt = SttSession(SONIOX_API_KEY)
        self.tts = TtsSession(eleven_client, ws, self.loop)

        # контекст для GPT
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        self.user_utterance = ""
        self.llm_lock = asyncio.Lock()
        self._finished = False
        self._greeting_sent = False

    async def send_clear(self):
        """Отправляем в Twilio 'clear' для бардж-ина."""
        if not self.stream_sid:
            return
        msg = {
            "event": "clear",
            "streamSid": self.stream_sid,
        }
        try:
            await self.ws.send_text(json.dumps(msg))
            logger.info("Sent Twilio clear for streamSid=%s", self.stream_sid)
        except Exception as e:
            logger.exception("Error sending Twilio clear: %s", e)

    async def barge_in(self, reason: str = ""):
        """Barge-in: пользователь перебивает — рубим TTS и чистим буфер Twilio."""
        if not self.tts.is_active():
            return
        if reason:
            logger.info("BARGE-IN (%s): cancelling TTS and clearing Twilio audio", reason)
        else:
            logger.info("BARGE-IN: cancelling TTS and clearing Twilio audio")

        # остановить TTS-стрим ElevenLabs
        self.tts.cancel()
        # очистить буфер аудио в Twilio (media queue) 
        await self.send_clear()

    async def generate_gpt_reply(self, user_text: str) -> str:
        if not openai_client:
            logger.warning("OPENAI_API_KEY is not set, skip GPT call")
            return ""

        self.messages.append({"role": "user", "content": user_text})
        messages_for_call = list(self.messages)

        def _run_sync(msgs):
            assistant_text = ""
            try:
                stream = openai_client.chat.completions.create(
                    model="gpt-5.1",
                    messages=msgs,
                    stream=True,
                    max_completion_tokens=96,
                    temperature=0.4,
                    reasoning_effort="none",
                )
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    token = getattr(delta, "content", None)
                    if not token:
                        continue
                    piece = token if isinstance(token, str) else str(token)
                    assistant_text += piece
                    # logger.info("GPT partial: %s", assistant_text)
            except Exception as e:
                logger.exception("Error in GPT stream: %s", e)
            return assistant_text.strip()

        assistant_text = await asyncio.to_thread(_run_sync, messages_for_call)
        if assistant_text:
            self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    async def handle_user_utterance(self, text: str):
        """Soniox дал <end> — закончили фразу пользователя."""
        text = text.strip()
        if not text:
            return

        async with self.llm_lock:
            logger.info("User utterance (final): %s", text)
            reply = await self.generate_gpt_reply(text)
            if reply:
                logger.info("GPT final reply: %s", reply)
                await self.tts.speak(reply)

    async def handle_stt_response(self, resp: dict):
        # ошибки Soniox
        if resp.get("error_code"):
            logger.error(
                "Soniox error %s: %s",
                resp.get("error_code"),
                resp.get("error_message"),
            )
            return

        tokens = resp.get("tokens", [])
        if not tokens:
            if resp.get("finished"):
                logger.info(
                    "Soniox finished: final_audio_proc_ms=%s total_audio_proc_ms=%s",
                    resp.get("final_audio_proc_ms"),
                    resp.get("total_audio_proc_ms"),
                )
                self._finished = True
            return

        # лог партиалов (для понимания задержки)
        partial_text = "".join(t.get("text", "") for t in tokens)
        if partial_text.strip():
            logger.info("Soniox partial: %s", partial_text)

        # перебираем токены
        for t in tokens:
            txt = t.get("text", "") or ""
            if not txt:
                continue

            # --- BARGE-IN: как только видим живой текст во время TTS ---
            if txt.strip() and self.tts.is_active():
                # здесь можно указать reason, чтобы в логах было видно, по какому токену сработало
                await self.barge_in(reason=f"token='{txt}'")
                # после этого TTS перестанет слать аудио, а Twilio очистит буфер
                # продолжаем обрабатывать текст как обычно (накапливаем фразу)

            # --- ENDPOINT DETECTION ---
            if txt == "<end>":
                final = self.user_utterance.strip()
                logger.info("Soniox END token received, final user text: '%s'", final)
                if final:
                    asyncio.create_task(self.handle_user_utterance(final))
                self.user_utterance = ""
                # флаг отдельный больше не нужен, всё держим на user_utterance
                continue

            # --- накапливаем только финальные токены в текущую реплику пользователя ---
            if t.get("is_final"):
                self.user_utterance += txt

    async def twilio_loop(self):
        """Читаем события Twilio и шлём аудио в Soniox."""
        try:
            while True:
                msg = await self.ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")

                if event == "connected":
                    logger.info("Twilio event=connected")

                elif event == "start":
                    start = data.get("start", {})
                    self.stream_sid = start.get("streamSid")
                    self.tts.set_stream_sid(self.stream_sid)
                    logger.info("Twilio stream START: %s", self.stream_sid)

                    # Сразу после старта стрима — приветствие ElevenLabs
                    if not self._greeting_sent:
                        self._greeting_sent = True
                        asyncio.create_task(self.send_greeting())

                elif event == "media":
                    media = data.get("media", {})
                    payload_b64 = media.get("payload")
                    if not payload_b64:
                        continue
                    audio_bytes = base64.b64decode(payload_b64)
                    try:
                        await self.stt.send_audio(audio_bytes)
                    except Exception as e:
                        logger.exception("Error sending audio to Soniox: %s", e)
                        break

                elif event == "stop":
                    logger.info("Twilio stream STOP received")
                    await self.stt.finalize()
                    break

                else:
                    logger.debug("Unhandled Twilio event: %s", event)

        except WebSocketDisconnect:
            logger.info("Twilio WebSocket disconnected")
            await self.stt.finalize()
        except Exception as e:
            logger.exception("Error in twilio_loop: %s", e)
            await self.stt.finalize()

    async def stt_loop(self):
        """Цикл чтения результатов Soniox."""
        await self.stt.receive_loop(self.handle_stt_response)
        self._finished = True

    async def run(self):
        """Запускаем STT и Twilio петли параллельно."""
        try:
            await self.stt.connect()
        except Exception as e:
            logger.exception("Cannot start SttSession: %s", e)
            await self.ws.close()
            return

        try:
            await asyncio.gather(
                self.twilio_loop(),
                self.stt_loop(),
            )
        finally:
            await self.stt.close()
            try:
                await self.ws.close()
            except Exception:
                pass

    async def send_greeting(self):
        """
        Первое приветствие через ElevenLabs.
        Сразу кладём его в контекст LLM как ответ ассистента.
        Приветствие тоже можно перебить, потому что идёт через общий TTS.
        """
        text = GREETING_TEXT.strip()
        if not text:
            return

        logger.info("Sending initial greeting TTS: %s", text)

        # Записываем приветствие как первую реплику ассистента,
        # чтобы модель знала, что мы уже поздоровались.
        self.messages.append({"role": "assistant", "content": text})

        # Произносим через тот же TTS-пайплайн (barge-in уже работает)
        await self.tts.speak(text)


# ============================
#   HTTP endpoints
# ============================

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

    host = request.url.hostname

    # Простая приветственная фраза до подключения стрима
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="wss://{host}/twilio-stream" />
        </Connect>
    </Response>"""

    return Response(content=twiml.strip(), media_type="text/xml")


@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    logger.info("Twilio WebSocket connected")
    session = CallSession(ws)
    await session.run()
    logger.info("twilio_stream handler finished")
