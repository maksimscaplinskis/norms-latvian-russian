import os
import json
import base64
import logging
import time
import asyncio
import httpx

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings, RealtimeEvents
import elevenlabs  # для Scribe Realtime
from starlette.websockets import WebSocketState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio")

app = FastAPI()

# ====  OpenAI общая конфигурация  ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set!")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ====  ElevenLabs общая конфигурация  ====

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID")

if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY is not set!")

try:
    eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
except Exception as e:
    logger.exception(f"Failed to init ElevenLabs client: {e}")
    eleven_client = None


# Хранилище STT-сессий по streamSid (теперь Scribe v2 Realtime)
stt_sessions: dict[str, "ScribeRealtimeSession"] = {}

# Хранилище LLM-сессий по streamSid
llm_sessions: dict[str, "LLMConversation"] = {}

# Хранилище Twilio WebSocket + event loop по streamSid (для отправки TTS)
twilio_connections: dict[str, tuple[WebSocket, asyncio.AbstractEventLoop]] = {}

# Мета-инфа по стримам: времена и т.п.
stream_meta: dict[str, dict] = {}


# ====  OpenAI LLM  ====
class LLMConversation:
    """
    Одна LLM-сессия на один Twilio streamSid.
    Хранит контекст и ходит в OpenAI (streaming).
    """

    def __init__(self, stream_sid: str):
        self.stream_sid = stream_sid
        self.messages: list[dict] = []

        system_prompt = (
            "Tu esi auto servisa balss asistents. "
            "Sākumā uzmanīgi noklausies klienta problēmu. "
            "Pēc pirmajiem vārdiem nosaki, vai klients runā latviski vai krieviski, "
            "un turpmāk runā tikai šajā valodā. "
            "Runā īsiem, vienkāršiem teikumiem. "
            "Kad saproti problēmu, piedāvā pierakstu uz auto pārbaudi "
            "un palīdz izvēlēties dienu un laiku. "
            "Kad pieraksts apstiprināts, pateicies un pieklājīgi nobeidz sarunu. "
            "Nejautā klientam pārāk detalizētu tehnisko informāciju par viņa auto."
        )
        self.messages.append({"role": "system", "content": system_prompt})

    def handle_user_utterance(self, text: str, lang_code: str | None = None) -> str:
        """
        Добавляем фразу пользователя в контекст, вызываем OpenAI (stream=True),
        стримим токены в лог, возвращаем финальный текст ответа.
        """

        # Можно при желании добавить инфу о языке в контекст
        # if lang_code:
        #     self.messages.append({
        #         "role": "system",
        #         "content": f"Lietotājs runā valodā: {lang_code}. "
        #                    f"Atbildi šajā pašā valodā."
        #     })

        self.messages.append({"role": "user", "content": text})
        logger.info(f"[{self.stream_sid}] LLM(OpenAI): sending user text: {text!r}")

        reply_text = ""

        try:
            stream = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self.messages,
                stream=True,
                max_completion_tokens=64,
                temperature=0.4,
                reasoning_effort="none",
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                reply_text += delta
                logger.info(f"[{self.stream_sid}] LLM partial: {delta!r}")

            logger.info(f"[{self.stream_sid}] LLM final: {reply_text!r}")
            self.messages.append({"role": "assistant", "content": reply_text})

        except Exception as e:
            logger.exception(f"[{self.stream_sid}] LLM error: {e}")
            reply_text = ""

        return reply_text


# ====  ElevenLabs Scribe v2 Realtime STT  ====
class ScribeRealtimeSession:
    """
    Одна STT-сессия ElevenLabs Scribe v2 Realtime на один Twilio streamSid.
    Принимает μ-law 8kHz байты от Twilio и отправляет их в Scribe по WebSocket.
    """

    def __init__(self, stream_sid: str, on_final_callback=None):
        self.stream_sid = stream_sid
        self.on_final_callback = on_final_callback
        self.connection = None
        self._closed = False

    async def start(self):
        """
        Открываем WebSocket-подключение к Scribe Realtime.
        Используем ulaw_8000 + VAD, чтобы не думать про commit().
        """
        if not ELEVENLABS_API_KEY:
            logger.error(
                f"[{self.stream_sid}] ELEVENLABS_API_KEY is not set, cannot start Scribe session"
            )
            return

        config = {
            "model_id": "scribe_v2_realtime",
            "audio_format": "ulaw_8000",   # под Twilio μ-law 8kHz :contentReference[oaicite:2]{index=2}
            "sample_rate": 8000,
            "commit_strategy": "vad",
            "vad_silence_threshold_secs": 0.5,
            "vad_threshold": 0.4,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 150,
            "include_timestamps": False,
        }

        logger.info(f"[{self.stream_sid}] Connecting Scribe Realtime with config={config}")
        # Используем high-level helper из SDK, как в доках
        # connection = await elevenlabs.speech_to_text.realtime.connect(config)
        self.connection = await elevenlabs.speech_to_text.realtime.connect(config)

        # Регистрируем обработчики событий
        self.connection.on(RealtimeEvents.SESSION_STARTED, self._on_session_started)
        self.connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self._on_partial_transcript)
        self.connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self._on_committed_transcript)
        self.connection.on(RealtimeEvents.ERROR, self._on_error)
        self.connection.on(RealtimeEvents.AUTH_ERROR, self._on_auth_error)
        self.connection.on(RealtimeEvents.QUOTA_EXCEEDED, self._on_quota_exceeded)
        self.connection.on(RealtimeEvents.CLOSE, self._on_close)

    async def push_audio(self, mulaw_bytes: bytes):
        """
        Получает μ-law 8kHz байты, кодирует в base64 и отправляет в Scribe.
        """
        if not mulaw_bytes or not self.connection or self._closed:
            return

        audio_b64 = base64.b64encode(mulaw_bytes).decode("ascii")

        try:
            await self.connection.send(
                {
                    "audio_base_64": audio_b64,
                    "sample_rate": 8000,
                }
            )
        except Exception as e:
            logger.exception(f"[{self.stream_sid}] Error sending audio to Scribe: {e}")

    async def stop(self):
        """
        Завершаем сессию: коммитим остаток и закрываем WebSocket.
        """
        if not self.connection or self._closed:
            return

        try:
            # финальный коммит на всякий случай
            await self.connection.commit()
        except Exception as e:
            logger.warning(
                f"[{self.stream_sid}] Error on Scribe commit during stop: {e}"
            )

        try:
            await self.connection.close()
        except Exception as e:
            logger.warning(
                f"[{self.stream_sid}] Error while closing Scribe connection: {e}"
            )

        self._closed = True

    # ----- callbacks -----

    def _on_session_started(self, data):
        logger.info(f"[{self.stream_sid}] Scribe session started: {data}")

    def _on_partial_transcript(self, data):
        text = getattr(data, "text", None)
        if not text and isinstance(data, dict):
            text = data.get("text")
        if text:
            logger.info(f"[{self.stream_sid}] Scribe partial: {text!r}")

    def _on_committed_transcript(self, data):
        # data может быть объектом или dict — подстрахуемся
        text = getattr(data, "text", None)
        if not text and isinstance(data, dict):
            text = data.get("text")
        if not text:
            return

        logger.info(f"[{self.stream_sid}] Scribe committed: {text!r}")

        # Scribe может возвращать language_code, но можно просто передавать None,
        # т.к. GPT сам по тексту поймёт RU/LV
        lang_code = getattr(data, "language_code", None)
        lang_raw = lang_code

        if self.on_final_callback:
            try:
                self.on_final_callback(self.stream_sid, text, lang_raw)
            except Exception as e:
                logger.exception(
                    f"[{self.stream_sid}] Error in on_final_callback: {e}"
                )

    def _on_error(self, error):
        logger.error(f"[{self.stream_sid}] Scribe ERROR: {error}")

    def _on_auth_error(self, error):
        logger.error(f"[{self.stream_sid}] Scribe AUTH_ERROR: {error}")

    def _on_quota_exceeded(self, error):
        logger.error(f"[{self.stream_sid}] Scribe QUOTA_EXCEEDED: {error}")

    def _on_close(self, *args, **kwargs):
        logger.info(f"[{self.stream_sid}] Scribe connection closed")
        self._closed = True


# ====  Twilio Media Stream: WebSocket с аудио  ====
@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    logger.info("Twilio WS connected")

    stream_sid: str | None = None
    loop = asyncio.get_running_loop()
    first_media_ts = None

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            event = data.get("event")

            if event == "connected":
                logger.info("Twilio event=connected")

            elif event == "start":
                start_info = data["start"]
                stream_sid = start_info["streamSid"]
                logger.info(f"Twilio stream START streamSid={stream_sid}")

                # сохраняем момент получения START (для замера до приветствия)
                stream_meta[stream_sid] = {
                    "start_ts": time.perf_counter(),
                }

                twilio_connections[stream_sid] = (ws, loop)

                # создаём Scribe STT-сессию под этот streamSid
                if stream_sid in stt_sessions:
                    logger.warning(
                        f"STT session for {stream_sid} already exists, overwriting"
                    )

                session = ScribeRealtimeSession(
                    stream_sid,
                    on_final_callback=handle_final_transcript,
                )
                await session.start()
                stt_sessions[stream_sid] = session

                # приветствие (TTS → Twilio)
                asyncio.create_task(send_initial_greeting(ws, stream_sid))

            elif event == "media":
                if first_media_ts is None:
                    first_media_ts = time.perf_counter()
                    logger.info("[%s] First media frame from Twilio", stream_sid)

                if not stream_sid:
                    # на всякий случай
                    logger.warning("Got media before start; skipping")
                    continue

                payload_b64 = data["media"]["payload"]
                mulaw_bytes = base64.b64decode(payload_b64)

                session = stt_sessions.get(stream_sid)
                if session:
                    # отправляем аудио в Scribe
                    await session.push_audio(mulaw_bytes)

            elif event == "stop":
                logger.info(f"Twilio stream STOP streamSid={stream_sid}")
                if stream_sid and stream_sid in stt_sessions:
                    try:
                        await stt_sessions[stream_sid].stop()
                    finally:
                        del stt_sessions[stream_sid]

                # чистим Twilio WebSocket для этого streamSid
                if stream_sid and stream_sid in twilio_connections:
                    del twilio_connections[stream_sid]

                # чистим мету
                if stream_sid and stream_sid in stream_meta:
                    del stream_meta[stream_sid]

                break

            else:
                logger.info(f"Unknown Twilio event: {event}")

    except WebSocketDisconnect:
        logger.info("Twilio WS disconnected")
    except Exception as e:
        logger.exception(f"Error in twilio_stream: {e}")
    finally:
        # на всякий случай чистим сессию, если осталась
        if stream_sid and stream_sid in stt_sessions:
            try:
                await stt_sessions[stream_sid].stop()
            except Exception:
                logger.exception(
                    f"[{stream_sid}] Error while stopping STT session in finally"
                )
            del stt_sessions[stream_sid]

        if stream_sid and stream_sid in twilio_connections:
            del twilio_connections[stream_sid]

        logger.info("Twilio handler finished")


# ====  Twilio webhook: TwiML, подключающий Media Stream  ====
@app.post("/voice")
async def voice_webhook(request: Request):
    """
    Twilio webhook: отдаём TwiML, который сразу подключает Media Stream.
    Клиент говорит первым.
    """
    host = request.url.hostname
    # Если у тебя всегда HTTPS, смело фиксируем wss
    ws_url = f"wss://{host}/twilio-stream"

    logger.info(f"/voice: building TwiML with ws_url={ws_url}")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="{ws_url}">
                <Parameter name="botSession" value="car-assistant" />
            </Stream>
        </Connect>
    </Response>"""

    # Важно: отдать как XML
    return PlainTextResponse(content=twiml, media_type="application/xml")


def handle_final_transcript(stream_sid: str, text: str, lang_raw: str | None):
    """
    Вызывается, когда Scribe STT отдала финальный текст (committed transcript).
    Создаёт/берёт LLMConversation и отправляет туда фразу.
    Потом запускает TTS → Twilio.
    """
    # Попробуем вытащить код языка, если lang_raw — JSON (в Azure так было),
    # для Scribe чаще всего это уже строка кода языка или None.
    lang_code = None
    if lang_raw:
        try:
            obj = json.loads(lang_raw)
            lang_code = obj.get("language") or obj.get("Language")
        except Exception:
            # если это уже строка (например "lv", "ru") или вообще что-то другое
            lang_code = str(lang_raw)

    logger.info(
        f"[{stream_sid}] Final transcript for LLM "
        f"[lang_raw={lang_raw!r}, lang_code={lang_code!r}]: {text!r}"
    )

    # Берём/создаём LLM-сессию
    if stream_sid not in llm_sessions:
        llm_sessions[stream_sid] = LLMConversation(stream_sid)

    conv = llm_sessions[stream_sid]

    # синхронный вызов OpenAI (как и раньше)
    reply_text = conv.handle_user_utterance(text, lang_code=lang_code)

    if reply_text:
        logger.info(f"[{stream_sid}] LLM reply ready (for TTS): {reply_text!r}")

        # сразу запускаем стрим TTS -> Twilio
        try:
            stream_tts_to_twilio(stream_sid, reply_text)
        except Exception as e:
            logger.exception(f"[{stream_sid}] Error while streaming TTS: {e}")


def stream_tts_to_twilio(stream_sid: str, text: str):
    """
    Сгенерировать речь через ElevenLabs и постримить её в Twilio как
    media-сообщения (mulaw/8000 base64) в bidirectional Media Stream.
    Работает из обычного (не-async) потока.
    """
    if not eleven_client:
        logger.warning(f"[{stream_sid}] ElevenLabs client not initialized, skip TTS")
        return
    if not ELEVENLABS_VOICE_ID:
        logger.warning(f"[{stream_sid}] ELEVENLABS_VOICE_ID is not set, skip TTS")
        return

    conn = twilio_connections.get(stream_sid)
    if not conn:
        logger.warning(f"[{stream_sid}] No Twilio WebSocket for this streamSid, cannot send TTS")
        return

    ws, loop = conn

    logger.info(f"[{stream_sid}] TTS: sending text to ElevenLabs ({len(text)} chars)")

    try:
        # Важно: формат ulaw_8000 – Twilio требует audio/x-mulaw 8kHz, base64.
        audio_stream = eleven_client.text_to_speech.stream(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL_ID,
            text=text,
            output_format="ulaw_8000",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.0,  # тут была опечатка similarity_byle
                use_speaker_boost=True,
                speed=1.3,
            ),
        )

        for chunk in audio_stream:
            # если чанк пустой или это не байты — пропускаем
            if not chunk or not isinstance(chunk, (bytes, bytearray)):
                continue

            # если Twilio уже отписался (stop/close) — выходим из цикла
            if stream_sid not in twilio_connections:
                logger.info(f"[{stream_sid}] Twilio connection gone, stop TTS streaming")
                break

            payload_b64 = base64.b64encode(chunk).decode("ascii")

            def _schedule_send(payload=payload_b64):
                """
                Этот код выполняется уже внутри event loop.
                Здесь ещё раз проверяем, что соединение живое,
                и отлавливаем ошибки отправки.
                """
                conn_inner = twilio_connections.get(stream_sid)
                if not conn_inner:
                    # стрим уже закрыт / удалён
                    return

                ws_inner, _ = conn_inner

                async def _send_chunk():
                    if ws_inner.application_state.name != "CONNECTED":
                        # WebSocket уже закрыт — ничего не делаем
                        return

                    msg = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload},
                    }

                    try:
                        await ws_inner.send_text(json.dumps(msg))
                    except RuntimeError as e:
                        # Классический случай: "Unexpected ASGI message 'websocket.send'..."
                        logger.warning(f"[{stream_sid}] TTS chunk send failed (WS closed): {e}")
                    except Exception:
                        logger.exception(f"[{stream_sid}] Unexpected error while sending TTS chunk")

                asyncio.create_task(_send_chunk())

            # планируем отправку чанка в event loop из текущего потока
            loop.call_soon_threadsafe(_schedule_send)

    except Exception as e:
        logger.exception(f"[{stream_sid}] ElevenLabs TTS error: {e}")


async def send_initial_greeting(ws: WebSocket, stream_sid: str):
    greeting_text = (
        "Labdien! Esmu virtuālais autoservisa palīgs. "
    )
    logger.info(f"[{stream_sid}] GREETING TTS: starting greeting")
    await eleven_stream_tts_to_twilio(ws, stream_sid, greeting_text, prefix="GREETING TTS")


async def eleven_stream_tts_to_twilio(
    ws: WebSocket,
    stream_sid: str,
    text: str,
    prefix: str = "TTS",
):
    """
    Стримим ответ из ElevenLabs в Twilio Media Stream.
    Используем U-Law 8kHz, чтобы не перекодировать.
    """
    if not ELEVENLABS_API_KEY:
        logger.error("ELEVENLABS_API_KEY is not set")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    params = {"output_format": "ulaw_8000"}
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,  # ⚠ ОБЯЗАТЕЛЬНО
        # "voice_settings": {
        #     "stability": 0.5,
        #     "similarity_boost": 0.0,
        #     "use_speaker_boost": True,
        # },
    }

    logger.info(
        f"[{stream_sid}] {prefix}: sending text to ElevenLabs (len={len(text)} chars)"
    )

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            params=params,
            json=payload,  # ⚠ именно json, не data
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                logger.error(
                    f"[{stream_sid}] {prefix} error: ElevenLabs HTTP {resp.status_code}, body={body!r}"
                )
                return

            # первая порция аудио – удобно для измерения латентности
            first_chunk_sent = False

            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue

                if ws.client_state != WebSocketState.CONNECTED:
                    logger.info(
                        f"[{stream_sid}] {prefix}: WebSocket already closed, stop sending audio"
                    )
                    break

                if not first_chunk_sent:
                    logger.info(
                        f"[{stream_sid}] {prefix}: first audio chunk ready to send to Twilio"
                    )
                    first_chunk_sent = True

                msg = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": base64.b64encode(chunk).decode("ascii"),
                    },
                }
                await ws.send_text(json.dumps(msg))
