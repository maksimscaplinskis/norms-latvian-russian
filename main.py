import os
import base64
import uuid
import json
from typing import Dict, List
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

import azure.cognitiveservices.speech as speechsdk  # Speech SDK
from openai import OpenAI  # v1 OpenAI client (Foundry / Azure OpenAI v1)

# ---------- Конфигурация из переменных окружения ----------

# Speech
SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")
DEFAULT_VOICE = os.environ.get("SPEECH_VOICE", "en-US-AmandaMultilingualNeural")

# Foundry / Azure OpenAI v1
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # https://...services.ai.azure.com/openai/v1/
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")        # deployment name, напр. car-assistant-gpt4o

if not SPEECH_KEY or not SPEECH_REGION:
    print("WARNING: SPEECH_KEY or SPEECH_REGION is not set")

if not (OPENAI_API_KEY and OPENAI_BASE_URL and OPENAI_MODEL):
    print("WARNING: OpenAI/Foundry config is incomplete "
          "(OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL)")

# ---------- Инициализация клиентов ----------

app = FastAPI()

logger = logging.getLogger("uvicorn.error")

# Speech
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_synthesis_voice_name = DEFAULT_VOICE

# Speech для Twilio: 8kHz μ-law
twilio_speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
twilio_speech_config.speech_synthesis_voice_name = DEFAULT_VOICE
twilio_speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
)

# OpenAI (Foundry / Azure OpenAI v1 endpoint)
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

SYSTEM_PROMPT = (
    "Tu esi draudzīgs un profesionāls virtuālais asistents autoservisā. "
    "Atbildi īsi, skaidri un vienkāršā valodā. "
    "Ja lietotājs raksta krieviski, atbildi krieviski; ja latviski – atbildi latviski."
)

# In-memory хранилище контекста по session_id (прототип)
SESSIONS: Dict[str, List[dict]] = {}

def synthesize_mulaw_base64_for_twilio(text: str) -> str:
    """
    Синтезирует речь в формате raw-8khz-8bit-mono-mulaw
    и возвращает base64-строку, готовую для Twilio media.payload.
    """
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=twilio_speech_config,
        audio_config=None,
    )
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_bytes = result.audio_data  # raw μ-law 8kHz, без WAV заголовка
        return base64.b64encode(audio_bytes).decode("ascii")
    else:
        details = getattr(result, "cancellation_details", None)
        msg = str(details.reason) if details else "Unknown TTS error for Twilio"
        raise RuntimeError(msg)

# ---------- Вспомогательная функция для TTS ----------

def synthesize_to_bytes(text: str) -> bytes:
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=None
    )
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    else:
        details = getattr(result, "cancellation_details", None)
        msg = str(details.reason) if details else "Unknown synthesis error"
        raise RuntimeError(msg)


# ---------- Маршруты ----------

@app.get("/")
async def root():
    return {"status": "ok", "message": "Azure voice gateway is running"}


# Простой чат с GPT (без голоса)
@app.post("/test-chat")
async def test_chat(payload: dict):
    user_text = payload.get("text") or ""
    if not user_text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.3,
        )
        answer = resp.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        print("OpenAI/Foundry error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# Текст -> голос (TTS). Возвращаем аудио в base64
@app.post("/test-tts")
async def test_tts(payload: dict):
    text = payload.get("text") or ""
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    try:
        audio_bytes = synthesize_to_bytes(text)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        return {
            "audio_base64": audio_b64,
            "voice": speech_config.speech_synthesis_voice_name,
        }
    except Exception as e:
        print("TTS error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# Текст -> голос, отдаём сразу audio/wav (удобно слушать в браузере)
@app.get("/tts-audio")
async def tts_audio(text: str):
    if not text:
        return JSONResponse({"error": "text query param is required"}, status_code=400)

    try:
        audio_bytes = synthesize_to_bytes(text)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        print("TTS error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# Новый эндпоинт диалога: GPT + TTS + session_id
@app.post("/dialog")
async def dialog(payload: dict):
    """
    Ожидает:
      {
        "text": "что говорит клиент",
        "session_id": "опционально, строка"
      }

    Возвращает:
      {
        "answer": "ответ ассистента",
        "audio_base64": "...",
        "session_id": "идентификатор сессии"
      }
    """
    user_text = payload.get("text") or ""
    if not user_text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    # берём session_id из запроса или создаём новый
    session_id = payload.get("session_id") or str(uuid.uuid4())

    # строим историю диалога
    history = SESSIONS.get(session_id)
    if not history:
        # новая сессия – добавляем system prompt
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

    history.append({"role": "user", "content": user_text})

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            temperature=0.3,
        )
        answer = resp.choices[0].message.content
        history.append({"role": "assistant", "content": answer})
        SESSIONS[session_id] = history  # сохраняем контекст

        # синтезируем голос
        audio_bytes = synthesize_to_bytes(answer)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        return {
            "answer": answer,
            "audio_base64": audio_b64,
            "session_id": session_id,
        }

    except Exception as e:
        print("Dialog error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# Пока заглушка – потом сюда прикрутим Twilio
@app.post("/voice", response_class=PlainTextResponse)
async def voice_webhook(request: Request):
    # Берём текущий хост, чтобы не хардкодить домен
    host = request.url.hostname
    ws_url = f"wss://{host}/twilio-stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="botSession" value="car-assistant" />
    </Stream>
  </Connect>
</Response>"""

    return twiml


@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    logger.info("Twilio WS connected")

    call_sid = None
    stream_sid = None
    responded = False  # чтобы ответить один раз

    try:
        while True:
            msg = await ws.receive_text()
            logger.info(f"Twilio WS raw message: {msg[:200]}")

            data = json.loads(msg)
            event = data.get("event")

            if event == "connected":
                logger.info(f"Twilio event=connected: {data}")

            elif event == "start":
                start_info = data.get("start", {})
                call_sid = start_info.get("callSid")
                stream_sid = start_info.get("streamSid")
                logger.info(f"Twilio stream START callSid={call_sid}, streamSid={stream_sid}")

            elif event == "media":
                media = data.get("media", {})
                chunk = media.get("chunk")
                ts = media.get("timestamp")
                payload_b64 = media.get("payload")
                logger.info(
                    f"Twilio media chunk={chunk}, ts={ts}, payload_len={len(payload_b64) if payload_b64 else 0}"
                )

                # --- ПРОСТОЙ ТЕСТОВЫЙ ОТВЕТ В ЗВОНКЕ ---
                # Один раз за звонок синтезируем фразу и шлём её в Twilio
                if not responded and stream_sid:
                    responded = True

                    # Здесь ТЕСТОВЫЙ текст – просто чтобы услышать голос
                    tts_text = (
                        "Sveiki! Šis ir tests. "
                        "Es runāju caur Azure balss servisu, integrētu ar Twilio."
                    )

                    try:
                        twilio_payload = synthesize_mulaw_base64_for_twilio(tts_text)

                        reply = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": twilio_payload
                            },
                        }

                        await ws.send_text(json.dumps(reply))
                        logger.info("Sent TTS audio back to Twilio")
                    except Exception as e:
                        logger.error(f"Error sending TTS to Twilio: {e}", exc_info=True)

            elif event == "stop":
                logger.info(f"Twilio stream STOP callSid={call_sid}, streamSid={stream_sid}")
                break

            else:
                logger.info(f"Twilio event other={event}: {data}")

    except WebSocketDisconnect:
        logger.info("Twilio WS disconnected (WebSocketDisconnect)")
    except Exception as e:
        logger.error(f"Twilio WS error: {e}", exc_info=True)
    finally:
        pass
