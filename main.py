import os
import json
import logging
import time
import asyncio

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from elevenlabs import (
    ElevenLabs,
    AudioFormat,
    CommitStrategy,
    RealtimeAudioOptions,
    RealtimeEvents,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio")

# FastAPI INIT
app = FastAPI()

# ElevenLabs INIT Config
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY is not set – Scribe STT will fail")

eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ElevenLabs SST Config
SCRIBE_CONFIG = RealtimeAudioOptions(
    model_id="scribe_v2_realtime",
    # language_code не указываем → автоопределение (RU/LV/EN и т.д.)
    audio_format=AudioFormat.ULAW_8000,
    commit_strategy=CommitStrategy.VAD,   # авто-коммиты по VAD
    vad_silence_threshold_secs=1.2,      # можно будет подкрутить
    vad_threshold=0.4,
    min_speech_duration_ms=150,
    min_silence_duration_ms=120,
    include_timestamps=False,
)

# WebSocket
@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    t0 = time.perf_counter()
    await ws.accept()
    logger.info("WS accepted, dt=%.3f", time.perf_counter() - t0)

    stream_sid = None
    media_frames = 0

    # === NEW: соединение с Eleven Scribe для этого звонка ===
    scribe_connection = None

    # Эти переменные потом пригодятся, когда будем слать текст в Azure
    last_partial = ""
    last_committed = ""

    # --- колбэки Scribe (вызваются из SDK, не async) ---

    def on_partial(msg):
        """
        partial_transcript – живой текст по мере речи.
        """
        nonlocal last_partial
        text = None
        # msg может быть dict или объект, подстрахуемся
        if isinstance(msg, dict):
            text = msg.get("text")
        else:
            text = getattr(msg, "text", None)

        if text:
            last_partial = text
            logger.info("SCRIBE partial: %s", text)

    def on_committed(msg):
        """
        committed_transcript – законченный фрагмент (реплика пользователя).
        """
        nonlocal last_committed
        text = None
        if isinstance(msg, dict):
            text = msg.get("text")
        else:
            text = getattr(msg, "text", None)

        if text:
            last_committed = text
            logger.info("SCRIBE COMMITTED: %s", text)
            # Тут позже будем дергать Azure/LLM

    def on_scribe_error(err):
        logger.error("SCRIBE ERROR: %s", err)

    def on_scribe_close():
        logger.info("SCRIBE connection closed")

    try:
        while True:
            # Twilio присылает текстовые JSON-сообщения
            msg = await ws.receive_text()
            now = time.perf_counter()

            data = json.loads(msg)
            event = data.get("event")

            logger.info("Received event=%s dt=%.3f", event, now - t0)

            if event == "connected":
                logger.info("Twilio event=connected")

            elif event == "start":
                stream_sid = data["start"]["streamSid"]
                logger.info("Twilio stream START streamSid=%s", stream_sid)

                # === NEW: открываем соединение с Eleven Scribe ===
                try:
                    scribe_connection = await eleven.speech_to_text.realtime.connect(
                        SCRIBE_CONFIG
                    )
                    logger.info("SCRIBE connected for streamSid=%s", stream_sid)

                    # Регистрируем обработчики событий
                    scribe_connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial)
                    scribe_connection.on(
                        RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed
                    )
                    scribe_connection.on(RealtimeEvents.ERROR, on_scribe_error)
                    scribe_connection.on(RealtimeEvents.CLOSE, on_scribe_close)

                except Exception as e:
                    logger.exception("Failed to connect to Scribe: %s", e)
                    # Можно решить: рвём звонок или просто живем без STT

            elif event == "media":
                media_frames += 1

                # === NEW: отправляем аудио в Scribe ===
                if scribe_connection is not None:
                    try:
                        payload_b64 = data["media"]["payload"]  # μ-law 8kHz, base64
                        # Важно: передать sample_rate=8000
                        await scribe_connection.send(
                            {
                                "audio_base_64": payload_b64,
                                "sample_rate": 8000,
                            }
                        )
                    except Exception as e:
                        logger.exception("Error sending audio to Scribe: %s", e)

            elif event == "mark":
                mark_name = data["mark"]["name"]
                logger.info("Twilio mark name=%s", mark_name)

            elif event == "stop":
                logger.info("Twilio stream STOP streamSid=%s", stream_sid)

                # === NEW: закрываем Scribe соединение ===
                if scribe_connection is not None:
                    try:
                        await scribe_connection.close()
                    except Exception as e:
                        logger.exception("Error closing Scribe connection: %s", e)

                break

            else:
                logger.info("Unknown Twilio event: %s", event)

    except WebSocketDisconnect:
        logger.info("Twilio WS disconnected")
    except Exception as e:
        logger.exception("Error in twilio_stream: %s", e)
    finally:
        logger.info("Twilio handler finished")


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


