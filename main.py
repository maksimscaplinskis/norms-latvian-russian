import os
import json
import logging
import time
import asyncio

import websockets

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio")

app = FastAPI()

ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")

SCRIBE_WS_URL = (
    "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    "?model_id=scribe_v2_realtime"
    "&audio_format=ulaw_8000"      # формат, который даёт Twilio
    "&commit_strategy=vad"         # авто-коммит по паузам
    "&include_timestamps=false"
)

async def scribe_receiver(ws):
    """
    Читаем события от ElevenLabs Scribe и логируем транскрипт.
    """
    try:
        async for msg in ws:
            data = json.loads(msg)
            mtype = data.get("message_type")

            if mtype == "session_started":
                logger.info(
                    f"Scribe session_started session_id={data.get('session_id')}"
                )

            elif mtype == "partial_transcript":
                text = data.get("text", "")
                if text:
                    logger.info(f"Scribe partial: {text!r}")

            elif mtype == "committed_transcript":
                text = data.get("text", "")
                if text:
                    logger.info(f"Scribe committed: {text!r}")

            else:
                # На первых порах просто смотрим, что там ещё приходит
                logger.info(f"Scribe event: {data}")

    except Exception as e:
        logger.exception(f"Scribe receiver error: {e}")

@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    logger.info("Twilio WS connected")

    stream_sid = None
    media_frames = 0

    scribe_ws = None
    scribe_task = None

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            event = data.get("event")

            if event == "connected":
                logger.info("Twilio event=connected")

            elif event == "start":
                stream_sid = data["start"]["streamSid"]
                logger.info(f"Twilio stream START streamSid={stream_sid}")

                # --- Подключаемся к ElevenLabs Scribe ---
                if not ELEVEN_API_KEY:
                    logger.error("ELEVENLABS_API_KEY is not set!")
                else:
                    logger.info("Connecting to ElevenLabs Scribe websocket...")
                    scribe_ws = await websockets.connect(
                        SCRIBE_WS_URL,
                        extra_headers={
                            "xi-api-key": ELEVEN_API_KEY,
                        },
                    )
                    # Запускаем приём транскриптов в фоне
                    scribe_task = asyncio.create_task(scribe_receiver(scribe_ws))

            elif event == "media":
                media_frames += 1
                # диагноcтика по-прежнему
                if media_frames % 50 == 0:
                    logger.info(
                        f"Received {media_frames} media frames for streamSid={stream_sid}"
                    )

                # Отправляем аудио в ElevenLabs, если подключены
                if scribe_ws is not None:
                    payload_b64 = data["media"]["payload"]  # база64 от Twilio (ulaw 8k)

                    msg_to_scribe = {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": payload_b64,
                        "sample_rate": 8000,   # Twilio Media Streams = 8kHz
                        # "commit": False      # с VAD можно commit не ставить
                    }

                    try:
                        await scribe_ws.send(json.dumps(msg_to_scribe))
                    except Exception as e:
                        logger.exception(f"Error sending audio to Scribe: {e}")

            elif event == "stop":
                logger.info(f"Twilio stream STOP streamSid={stream_sid}")
                break

            else:
                logger.info(f"Unknown Twilio event: {event}")

    except WebSocketDisconnect:
        logger.info("Twilio WS disconnected")

    except Exception as e:
        logger.exception(f"Error in twilio_stream: {e}")

    finally:
        # Чисто закрываем Scribe
        if scribe_task is not None:
            scribe_task.cancel()
        if scribe_ws is not None:
            try:
                await scribe_ws.close()
            except Exception:
                pass

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


