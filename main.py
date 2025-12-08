import os
import json
import base64
import logging
import asyncio

from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.responses import Response
from starlette.websockets import WebSocketDisconnect

import websockets
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-soniox-openai-eleven")

app = FastAPI()

# === ENV / clients ===

SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None


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

    host = request.url.hostname

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

    if not SONIOX_API_KEY:
        logger.error("SONIOX_API_KEY is not set")
        await ws.close()
        return

    loop = asyncio.get_running_loop()
    talk_lock = asyncio.Lock()          # чтобы не было двух голосовых ответов одновременно
    stream_sid_holder = {"sid": None}   # сюда положим streamSid от Twilio

    # Простейший промпт ассистента (можем потом заменить на твой кастомный)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты голосовой ассистент автосервиса. "
                "Отвечай коротко, по делу, без приветствий в начале ответа. "
                "Определи язык пользователя (русский или латышский) по его словам "
                "и отвечай на том же языке."
            ),
        }
    ]

    # === подключение к Soniox ===
    try:
        soniox_ws = await websockets.connect("wss://stt-rt.soniox.com/transcribe-websocket")
        logger.info("Connected to Soniox WebSocket")
    except Exception as e:
        logger.exception(f"Cannot connect to Soniox: {e}")
        await ws.close()
        return

    config_msg = {
        "api_key": SONIOX_API_KEY,
        "model": "stt-rt-preview",
        "audio_format": "mulaw",
        "sample_rate": 8000,
        "num_channels": 1,
        "enable_language_identification": True,
        "language_hints": ["ru", "lv"],
        "enable_endpoint_detection": True,
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

    # === GPT ===

    async def generate_gpt_reply(user_text: str) -> str:
        """
        Вызываем OpenAI Chat Completions c stream=True в отдельном потоке.
        Возвращаем финальный текст ответа.
        """
        if not openai_client:
            logger.warning("OPENAI_API_KEY is not set, skip GPT call")
            return ""

        # Добавляем реплику пользователя в историю диалога
        messages.append({"role": "user", "content": user_text})
        messages_for_call = list(messages)  # копия для отдельного потока

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
                    # В большинстве случаев content — это строка
                    piece = token if isinstance(token, str) else str(token)
                    assistant_text += piece
                    logger.info("GPT partial: %s", assistant_text)
            except Exception as e:
                logger.exception(f"Error in GPT stream: {e}")
            return assistant_text.strip()

        assistant_text = await asyncio.to_thread(_run_sync, messages_for_call)

        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

        return assistant_text

    # === ElevenLabs TTS ===

    async def speak_with_elevenlabs(text: str):
        """
        Вызываем ElevenLabs TTS stream(output_format=ulaw_8000) и в реальном времени
        отсылаем чанки аудио обратно в Twilio как media-сообщения.
        """
        if not eleven_client:
            logger.warning("ELEVENLABS_API_KEY is not set, skip TTS")
            return
        if not ELEVENLABS_VOICE_ID:
            logger.warning("ELEVENLABS_VOICE_ID is not set, skip TTS")
            return
        if not text.strip():
            return

        sid = stream_sid_holder["sid"]
        if not sid:
            logger.warning("streamSid is not yet known, cannot send audio")
            return

        def _tts_and_stream():
            try:
                response = eleven_client.text_to_speech.stream(
                    voice_id=ELEVENLABS_VOICE_ID,
                    model_id=ELEVENLABS_MODEL_ID,
                    text=text,
                    output_format="ulaw_8000",  # μ-law 8kHz — формат Twilio
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.0,
                        style=0.0,
                        use_speaker_boost=True,
                        speed=1.3,
                    ),
                )
                for chunk in response:
                    if not chunk:
                        continue
                    if not isinstance(chunk, (bytes, bytearray)):
                        continue

                    payload = base64.b64encode(chunk).decode("ascii")
                    msg = {
                        "event": "media",
                        "streamSid": sid,
                        "media": {"payload": payload},
                    }
                    fut = asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps(msg)),
                        loop,
                    )
                    try:
                        fut.result()
                    except Exception as e:
                        logger.exception(f"Error sending audio to Twilio: {e}")
                        break
            except Exception as e:
                logger.exception(f"Error in ElevenLabs TTS streaming: {e}")

        await asyncio.to_thread(_tts_and_stream)

    # === обработка финальных фраз от пользователя ===

    async def handle_user_utterance(text: str):
        """
        Получили финальную фразу от Soniox (по <end>) — вызываем GPT и затем TTS.
        Используем lock, чтобы не говорить два ответа одновременно.
        """
        if not text.strip():
            return

        if talk_lock.locked():
            logger.info("Assistant is already speaking, skip utterance: %s", text)
            return

        async with talk_lock:
            logger.info("User utterance (final): %s", text)
            reply = await generate_gpt_reply(text)
            if reply:
                logger.info("GPT final reply: %s", reply)
                await speak_with_elevenlabs(reply)

    # === корутины для Twilio <-> Soniox ===

    async def twilio_to_soniox():
        """
        Читаем события от Twilio и отправляем аудио в Soniox.
        """
        try:
            while True:
                msg = await ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")

                if event == "connected":
                    logger.info("Twilio event=connected")

                elif event == "start":
                    start = data.get("start", {})
                    stream_sid = start.get("streamSid")
                    stream_sid_holder["sid"] = stream_sid
                    logger.info(f"Twilio stream START: {stream_sid}")

                elif event == "media":
                    media = data.get("media", {})
                    payload_b64 = media.get("payload")
                    if not payload_b64:
                        continue
                    audio_bytes = base64.b64decode(payload_b64)
                    try:
                        await soniox_ws.send(audio_bytes)
                    except Exception as e:
                        logger.exception(f"Error sending audio to Soniox: {e}")
                        break

                elif event == "stop":
                    logger.info("Twilio stream STOP received")
                    try:
                        await soniox_ws.send(b"")
                    except Exception:
                        pass
                    break

                else:
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

    async def soniox_to_logic():
        """
        Читаем ответы от Soniox:
        - логируем partial,
        - собираем финальную реплику по is_final,
        - по токену <end> запускаем GPT+TTS.
        """
        user_utterance = ""
        try:
            async for message in soniox_ws:
                try:
                    resp = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON Soniox message: {message!r}")
                    continue

                if resp.get("error_code"):
                    logger.error(
                        "Soniox error %s: %s",
                        resp.get("error_code"),
                        resp.get("error_message"),
                    )
                    break

                tokens = resp.get("tokens", [])
                if not tokens:
                    if resp.get("finished"):
                        logger.info(
                            "Soniox finished: final_audio_proc_ms=%s total_audio_proc_ms=%s",
                            resp.get("final_audio_proc_ms"),
                            resp.get("total_audio_proc_ms"),
                        )
                        break
                    continue

                # partial лог
                partial_text = "".join(t.get("text", "") for t in tokens)
                if partial_text.strip():
                    logger.info("Soniox partial: %s", partial_text)

                # разбираем токены
                for t in tokens:
                    txt = t.get("text", "") or ""
                    if txt == "<end>":
                        final = user_utterance.strip()
                        if final:
                            # запускаем обработку фразы (без await — отдельная таска)
                            asyncio.create_task(handle_user_utterance(final))
                        user_utterance = ""
                    elif t.get("is_final"):
                        user_utterance += txt

        except Exception as e:
            logger.exception(f"Error in soniox_to_logic: {e}")

    try:
        await asyncio.gather(
            twilio_to_soniox(),
            soniox_to_logic(),
        )
    finally:
        logger.info("Closing Soniox WS and Twilio WS")
        try:
            await soniox_ws.close()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("twilio_stream handler finished")
