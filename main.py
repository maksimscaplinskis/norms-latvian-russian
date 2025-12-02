import os
import base64

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse

import azure.cognitiveservices.speech as speechsdk  # Speech SDK
from openai import OpenAI  # v1 OpenAI client (умеет работать с Azure/Foundry)


# ---------- Конфигурация из переменных окружения ----------

# Speech
SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")
DEFAULT_VOICE = os.environ.get("SPEECH_VOICE", "en-US-AmandaMultilingualNeural")

# Foundry / Azure OpenAI v1
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # типа https://xxx.services.ai.azure.com/openai/v1/
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")        # deployment name, например car-assistant-gpt4o

if not SPEECH_KEY or not SPEECH_REGION:
    print("WARNING: SPEECH_KEY or SPEECH_REGION is not set")

if not (OPENAI_API_KEY and OPENAI_BASE_URL and OPENAI_MODEL):
    print("WARNING: OpenAI/Foundry config is incomplete "
          "(OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL)")

# ---------- Инициализация клиентов ----------

app = FastAPI()

# Speech
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_synthesis_voice_name = DEFAULT_VOICE

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


# ---------- Маршруты ----------

@app.get("/")
async def root():
    return {"status": "ok", "message": "Azure voice gateway is running"}


@app.post("/test-chat")
async def test_chat(payload: dict):
    """
    Ожидает JSON: { "text": "..." }
    Возвращает: { "answer": "..." }
    """
    user_text = payload.get("text") or ""
    if not user_text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,  # это deployment name из Foundry
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


@app.post("/test-tts")
async def test_tts(payload: dict):
    """
    Ожидает JSON: { "text": "..." }
    Возвращает: { "audio_base64": "...", "voice": "..."}
    """
    text = payload.get("text") or ""
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    try:
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_bytes = result.audio_data
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
            return {
                "audio_base64": audio_b64,
                "voice": speech_config.speech_synthesis_voice_name,
            }
        else:
            details = getattr(result, "cancellation_details", None)
            msg = str(details.reason) if details else "Unknown synthesis error"
            return JSONResponse({"error": msg}, status_code=500)

    except Exception as e:
        print("TTS error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


# Пока заглушка – потом сюда прикрутим Twilio
@app.post("/voice", response_class=PlainTextResponse)
async def voice_webhook():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Voice gateway is not connected to Twilio yet.</Say>
</Response>"""
    return twiml


@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    print("WS connected (stub, Twilio is not wired yet)")
    try:
        while True:
            data = await ws.receive_text()
            print("WS message:", data[:200], "...")
    except Exception as e:
        print("WS closed:", e)