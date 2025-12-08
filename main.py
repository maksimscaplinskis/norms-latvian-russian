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
    "Ты голосовой ассистент автосервиса. "
    "Определи, говорит ли клиент по-русски или по-латышски, "
    "и отвечай только на этом языке. "
    "Говори очень короткими фразами (1–2 предложения). "
    "Из наводящих вопросов ты можешь уточнить только марку автомобиля"
    "Не начинай ответ с приветствия. "
    "Твои задачи: "
    "1) Понять проблему с машиной. Если клиент уже описал проблему, не переспрашивай её ещё раз. "
    "2) Понять, хочет ли клиент записаться на осмотр. Если клиент сам говорит, что хочет записаться, сразу переходи к записи. "
    "3) Если клиент хочет записаться, предложи один конкретный вариант времени (дату и время) и спроси, подходит ли он. "
    "4) После подтверждения времени кратко поблагодари и заверши разговор."
)

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
            "enable_endpoint_detection": True,
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
        """Запрос на остановку текущего TTS-стрима (для barge-in)."""
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

        def _run():
            try:
                # ulaw_8000 специально под Twilio :contentReference[oaicite:3]{index=3}
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
                        # скорость в допустимом диапазоне ~1.1 
                        speed=1.1,
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
        self.awaiting_new_utterance = True
        self.llm_lock = asyncio.Lock()
        self._finished = False

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

    async def barge_in(self):
        """Barge-in: пользователь перебивает — рубим TTS и чистим буфер Twilio."""
        if self.tts.is_active():
            logger.info("BARGE-IN detected: cancelling TTS and clearing Twilio audio")
            self.tts.cancel()
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
                    model=OPENAI_MODEL,
                    messages=msgs,
                    stream=True,
                    max_completion_tokens=64,
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
                    logger.info("GPT partial: %s", assistant_text)
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

        # лог партиалов
        partial_text = "".join(t.get("text", "") for t in tokens)
        if partial_text.strip():
            logger.info("Soniox partial: %s", partial_text)

        # обработка токенов (+ barge-in)
        for t in tokens:
            txt = t.get("text", "") or ""

            # <end> — конец высказывания пользователя :contentReference[oaicite:5]{index=5}
            if txt == "<end>":
                final = self.user_utterance.strip()
                if final:
                    # не блокируем цикл Soniox
                    asyncio.create_task(self.handle_user_utterance(final))
                self.user_utterance = ""
                self.awaiting_new_utterance = True
                continue

            # начало НОВОЙ реплики пользователя после <end>
            if self.awaiting_new_utterance and txt.strip():
                # если ассистент сейчас говорит — бардж-ин
                if self.tts.is_active():
                    await self.barge_in()
                self.awaiting_new_utterance = False

            # накапливаем только финальные токены
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
    <Say language="ru-RU" voice="woman">
        Соединение установлено. Говорите, я вас слушаю.
    </Say>
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
