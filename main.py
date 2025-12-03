import json
import logging
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio")

app = FastAPI()

@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    t0 = time.perf_counter()
    await ws.accept()
    logger.info("WS accepted, dt=%.3f", time.perf_counter() - t0)

    stream_sid = None
    media_frames = 0

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
                logger.info(f"Twilio stream START streamSid={stream_sid}")

            elif event == "media":
                media_frames += 1
                # payload = data["media"]["payload"]  # base64 аудио 20ms μ-law 8kHz
                if media_frames % 50 == 0:
                    logger.info(
                        f"Received {media_frames} media frames for streamSid={stream_sid}"
                    )

            elif event == "mark":
                mark_name = data["mark"]["name"]
                logger.info(f"Twilio mark name={mark_name}")

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
        logger.info("Twilio handler finished")


