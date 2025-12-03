import os
import json
import logging
import time
import asyncio
import httpx

import websockets

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio")

app = FastAPI()

ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")

SCRIBE_BASE_URL = (
    "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    "?model_id=scribe_v2_realtime"
    "&audio_format=ulaw_8000"
    "&commit_strategy=vad"
    "&include_timestamps=false"
)

TOKEN_URL = "https://api.elevenlabs.io/v1/speech-to-text/get-realtime-token"

async def get_scribe_token() -> str | None:
    if not ELEVEN_API_KEY:
        logger.error("ELEVENLABS_API_KEY is not set!")
        return None

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                TOKEN_URL,
                headers={
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVEN_API_KEY,
                },
                json={
                    "model_id": "scribe_v2_realtime",
                    "ttl_secs": 300,
                },
            )
        if resp.status_code != 200:
            logger.error(
                "Failed to get scribe token: %s %s",
                resp.status_code,
                resp.text[:200],
            )
            return None

        data = resp.json()
        token = data.get("token")
        if not token:
            logger.error("No 'token' field in scribe token response: %s", data)
            return None

        logger.info("Got scribe token")
        return token

    except Exception as e:
        logger.exception(f"Error getting scribe token: {e}")
        return None

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

                # --- Подключаемся к ElevenLabs Scribe через token ---
                token = await get_scribe_token()
                if not token:
                    logger.error("Cannot start Scribe: no token")
                else:
                    scribe_url = f"{SCRIBE_BASE_URL}&token={token}"
                    logger.info(f"Connecting to ElevenLabs Scribe websocket: {scribe_url}")
                    try:
                        scribe_ws = await websockets.connect(scribe_url)
                        scribe_task = asyncio.create_task(scribe_receiver(scribe_ws))
                    except Exception as e:
                        logger.exception(f"Error connecting to Scribe websocket: {e}")
                        scribe_ws = None
                        scribe_task = None

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


