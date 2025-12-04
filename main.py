import os
import json
import base64
import logging
import time
import asyncio
import httpx
import audioop
import websockets
from urllib.parse import urlencode
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError


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
    STT-сессия ElevenLabs Scribe v2 Realtime для одного Twilio streamSid.
    Берём μ-law 8kHz от Twilio, конвертим в PCM 16kHz и шлём в WebSocket.
    """

    def __init__(self, stream_sid: str, on_final_callback=None):
        self.stream_sid = stream_sid
        self.on_final_callback = on_final_callback

        self._rate_state = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._closed = False

    async def start(self):
        """
        Открываем WebSocket к Scribe v2 Realtime и запускаем send/recv циклы.
        """
        if not ELEVENLABS_API_KEY:
            logger.error("[%s] ELEVENLABS_API_KEY is not set, cannot start Scribe", self.stream_sid)
            return

        params = {
            "model_id": "scribe_v2_realtime",
            "audio_format": "pcm_16000",
            "commit_strategy": "vad",
            "vad_silence_threshold_secs": 0.5,
            "vad_threshold": 0.4,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 150,
            "include_timestamps": "false",
        }

        ws_url = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?" + urlencode(params)
        logger.info("[%s] Connecting Scribe Realtime with params=%s", self.stream_sid, params)

        try:
            self._ws = await websockets.connect(
                ws_url,
                extra_headers={"xi-api-key": ELEVENLABS_API_KEY},
                ping_interval=30,
                ping_timeout=10,
            )
            logger.info("[%s] Scribe Realtime WebSocket connected", self.stream_sid)
        except Exception as e:
            logger.exception("[%s] Failed to connect to Scribe Realtime: %s", self.stream_sid, e)
            return

        # запускаем фоновые задачи
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

    def push_audio(self, mulaw_bytes: bytes):
        """
        Получает μ-law 8kHz от Twilio, конвертирует в 16kHz PCM и кладёт в очередь.
        """
        if not mulaw_bytes or self._closed:
            return

        # μ-law 8k -> PCM16 8k
        pcm16_8k = audioop.ulaw2lin(mulaw_bytes, 2)

        # 8k -> 16k
        pcm16_16k, self._rate_state = audioop.ratecv(
            pcm16_8k,
            2,      # sample width
            1,      # channels
            8000,   # in_rate
            16000,  # out_rate
            self._rate_state,
        )

        try:
            self._audio_queue.put_nowait(pcm16_16k)
        except asyncio.QueueFull:
            logger.warning("[%s] Scribe audio queue full, dropping chunk", self.stream_sid)

    def stop(self):
        """
        Синхронный стоп: ставим флаг, гасим задачи и закрываем WebSocket.
        Вызывается из Twilio-хэндлера без await.
        """
        if self._closed:
            return

        self._closed = True

        # разблокировать send_loop
        try:
            self._audio_queue.put_nowait(b"")
        except Exception:
            pass

        async def _cleanup():
            if self._send_task:
                self._send_task.cancel()
            if self._recv_task:
                self._recv_task.cancel()
            if self._ws and not self._ws.closed:
                try:
                    await self._ws.close()
                except Exception:
                    logger.exception("[%s] Error while closing Scribe WS", self.stream_sid)

            logger.info("[%s] Scribe session stopped", self.stream_sid)

        # Запускаем асинхронную очистку в текущем event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_cleanup())
        except RuntimeError:
            # нет активного loop (на всякий случай) – игнорируем
            pass

    async def _send_loop(self):
        assert self._ws is not None
        try:
            while not self._closed:
                chunk = await self._audio_queue.get()
                if not chunk or self._closed:
                    break

                if self._ws.closed:
                    break

                audio_b64 = base64.b64encode(chunk).decode("ascii")
                msg = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": audio_b64,
                    "sample_rate": 16000,
                }

                try:
                    await self._ws.send(json.dumps(msg))
                except Exception as e:
                    logger.exception("[%s] Error sending audio to Scribe: %s", self.stream_sid, e)
                    break
        finally:
            logger.info("[%s] Scribe send loop finished", self.stream_sid)

    async def _recv_loop(self):
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except Exception:
                    logger.warning("[%s] Non-JSON message from Scribe: %r", self.stream_sid, raw)
                    continue

                mtype = data.get("message_type")
                text = data.get("text") or ""
                lang_code = data.get("language_code")  # может и не быть

                if mtype == "session_started":
                    logger.info("[%s] Scribe session_started: %s", self.stream_sid, data)
                elif mtype == "partial_transcript":
                    if text:
                        logger.info("[%s] Scribe partial: %r", self.stream_sid, text)
                elif mtype in ("committed_transcript", "committed_transcript_with_timestamps"):
                    logger.info("[%s] Scribe committed: %r (lang=%s)", self.stream_sid, text, lang_code)
                    if self.on_final_callback and text:
                        try:
                            # handle_final_transcript(stream_sid, text, lang_raw)
                            self.on_final_callback(self.stream_sid, text, lang_code)
                        except Exception:
                            logger.exception("[%s] Error in on_final_callback", self.stream_sid)
                elif mtype and mtype.endswith("_error"):
                    logger.error("[%s] Scribe error event: %s", self.stream_sid, data)
                else:
                    logger.debug("[%s] Scribe event %s: %s", self.stream_sid, mtype, data)

        except (ConnectionClosedOK, ConnectionClosedError):
            logger.info("[%s] Scribe websocket closed", self.stream_sid)
        except Exception as e:
            logger.exception("[%s] Scribe recv loop error: %s", self.stream_sid, e)
        finally:
            self._closed = True
            logger.info("[%s] Scribe recv loop finished", self.stream_sid)


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
                asyncio.create_task(session.start())
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
                    stt_sessions[stream_sid].stop()
                    del stt_sessions[stream_sid]

                if stream_sid and stream_sid in twilio_connections:
                    del twilio_connections[stream_sid]

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
        if stream_sid and stream_sid in stt_sessions:
            stt_sessions[stream_sid].stop()
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
