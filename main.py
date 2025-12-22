import os
import json
import base64
import logging
import asyncio

from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.responses import Response
from starlette.websockets import WebSocketDisconnect

import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-soniox")

SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
SONIOX_MODEL = os.getenv("SONIOX_MODEL", "stt-rt-v3")

app = FastAPI()


class SttSession:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws: websockets.WebSocketClientProtocol | None = None

    async def connect(self):
        if not self.api_key:
            raise RuntimeError("SONIOX_API_KEY is not set")

        self.ws = await websockets.connect(
            "wss://stt-rt.soniox.com/transcribe-websocket"
        )
        logger.info("Connected to Soniox WebSocket")

        config_msg = {
            "api_key": self.api_key,
            "model": SONIOX_MODEL,
            "audio_format": "mulaw",
            "sample_rate": 8000,
            "num_channels": 1,
            "enable_language_identification": True,
            "language_hints": ["en","ru","lv"],
            "enable_endpoint_detection": True,
            "client_reference_id": "twilio-call",
        }
        await self.ws.send(json.dumps(config_msg))
        logger.info("Sent Soniox config")

    async def send_audio(self, audio_bytes: bytes):
        if self.ws:
            await self.ws.send(audio_bytes)

    async def receive_loop(self, handler):
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

    async def finalize(self):
        if self.ws:
            try:
                await self.ws.send(b"")
            except Exception:
                pass

    async def close(self):
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


class CallSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.stt = SttSession(SONIOX_API_KEY)
        self._finished = False

    async def handle_stt_response(self, resp: dict):
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

        partial = "".join(t.get("text", "") for t in tokens if not t.get("is_final"))
        final = "".join(t.get("text", "") for t in tokens if t.get("is_final"))

        if partial.strip():
            logger.info("Soniox partial: %s", partial.strip())
        if final.strip():
            logger.info("Soniox final: %s", final.strip())

    async def twilio_loop(self):
        try:
            while True:
                msg = await self.ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")

                if event == "connected":
                    logger.info("Twilio event=connected")

                elif event == "start":
                    start = data.get("start", {})
                    stream_sid = start.get("streamSid")
                    logger.info("Twilio stream START: %s", stream_sid)

                elif event == "media":
                    payload_b64 = (data.get("media") or {}).get("payload")
                    if not payload_b64:
                        continue
                    audio_bytes = base64.b64decode(payload_b64)
                    await self.stt.send_audio(audio_bytes)

                elif event == "stop":
                    logger.info("Twilio stream STOP received")
                    await self.stt.finalize()
                    break

                else:
                    logger.debug("Unhandled Twilio event: %s", event)

        except WebSocketDisconnect:
            logger.info("Twilio WebSocket disconnected")
        except Exception as e:
            logger.exception("Error in twilio_loop: %s", e)

    async def run(self):
        try:
            await self.stt.connect()
        except Exception as e:
            logger.exception("Cannot start SttSession: %s", e)
            await self.ws.close()
            return

        try:
            await asyncio.gather(
                self.twilio_loop(),
                self.stt.receive_loop(self.handle_stt_response),
            )
        finally:
            await self.stt.close()
            try:
                await self.ws.close()
            except Exception:
                pass


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

    host = request.url.hostname or "localhost"

    twiml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <Response>
        <Connect>
            <Stream url=\"wss://{host}/twilio-stream\" />
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
