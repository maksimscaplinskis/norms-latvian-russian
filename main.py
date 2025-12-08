import os
import json
import base64
import logging
import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from fastapi import Form, Request
from starlette.websockets import WebSocketDisconnect

import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-soniox")

app = FastAPI()

SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")  # не забудь выставить в Azure env


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
    logger.info(f"Incoming call: CallSid={CallSid}, From={From}, To={To}")

    # Лучше пропиши свой хост явно, чтобы не зависеть от Host header
    # host = "testnormswebapp.azurewebsites.net"
    host = request.url.hostname

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say language="ru-RU" voice="woman">
            Это тестовый ответ. Сейчас мы подключили поток аудио в реальном времени.
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

    if not SONIOX_API_KEY:
        logger.error("SONIOX_API_KEY is not set!")
        await ws.close()
        return

    # Подключаемся к Soniox WebSocket API
    try:
        soniox_ws = await websockets.connect(
            "wss://stt-rt.soniox.com/transcribe-websocket"
        )
        logger.info("Connected to Soniox WebSocket")
    except Exception as e:
        logger.exception(f"Cannot connect to Soniox: {e}")
        await ws.close()
        return

    # Отправляем конфигурацию с реальными параметрами аудио
    config_msg = {
        "api_key": SONIOX_API_KEY,
        "model": "stt-rt-v3",        # можно поменять на нужную модель
        "audio_format": "mulaw",          # Twilio PCMU
        "sample_rate": 8000,
        "num_channels": 1,
        "enable_language_identification": True,
        "language_hints": ["ru", "lv"],   # русский + латышский
        "client_reference_id": "twilio-call",
    }

    try:
        await soniox_ws.send(json.dumps(config_msg))
        logger.info("Sent Soniox config")
    except Exception as e:
        logger.exception(f"Failed to send Soniox config: {e}")
        await soniox_ws.close()
        await ws.close()
        return

    async def twilio_to_soniox():
        """
        Читаем события от Twilio и прокидываем только аудио в Soniox.
        """
        try:
            while True:
                msg = await ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")

                if event == "start":
                    stream_sid = data.get("start", {}).get("streamSid")
                    logger.info(f"Twilio stream START: {stream_sid}")

                elif event == "media":
                    media = data.get("media", {})
                    payload_b64 = media.get("payload")
                    if not payload_b64:
                        continue

                    # base64 → raw μ-law 8kHz bytes
                    audio_bytes = base64.b64decode(payload_b64)

                    # отправляем как бинарный фрейм в Soniox
                    try:
                        await soniox_ws.send(audio_bytes)
                    except Exception as e:
                        logger.exception(f"Error sending audio to Soniox: {e}")
                        break

                elif event == "stop":
                    logger.info("Twilio stream STOP received")
                    # Пустой фрейм — сигнал Soniox завершить стрим
                    try:
                        await soniox_ws.send(b"")
                    except Exception:
                        pass
                    break

                else:
                    # можно залогировать другие события, если интересно
                    logger.debug(f"Unhandled Twilio event: {event}")

        except WebSocketDisconnect:
            logger.info("Twilio WebSocket disconnected")
            try:
                await soniox_ws.send(b"")
            except Exception:
                pass
        except Exception as e:
            logger.exception(f"Error in twilio_to_soniox: {e}")
            try:
                await soniox_ws.send(b"")
            except Exception:
                pass

    async def soniox_to_logs():
        """
        Читаем ответы Soniox и логируем финальные токены (текст).
        """
        try:
            async for message in soniox_ws:
                try:
                    resp = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON Soniox message: {message!r}")
                    continue

                if resp.get("error_code"):
                    logger.error(
                        f"Soniox error {resp.get('error_code')}: "
                        f"{resp.get('error_message')}"
                    )
                    break

                tokens = resp.get("tokens", [])
                # Собираем только финальные токены
                final_text = "".join(
                    t.get("text", "")
                    for t in tokens
                    if t.get("is_final")
                )

                if final_text:
                    language = resp.get("language")
                    logger.info(
                        "Soniox final: %s %s",
                        final_text,
                        f"(lang={language})" if language else "",
                    )

                if resp.get("finished"):
                    logger.info(
                        "Soniox finished: final_audio_proc_ms=%s total_audio_proc_ms=%s",
                        resp.get("final_audio_proc_ms"),
                        resp.get("total_audio_proc_ms"),
                    )
                    break

        except Exception as e:
            logger.exception(f"Error in soniox_to_logs: {e}")

    # Запускаем обе задачи параллельно
    try:
        await asyncio.gather(
            twilio_to_soniox(),
            soniox_to_logs(),
        )
    finally:
        logger.info("Closing Soniox WS")
        try:
            await soniox_ws.close()
        except Exception:
            pass

        try:
            await ws.close()
        except Exception:
            pass

        logger.info("twilio_stream handler finished")