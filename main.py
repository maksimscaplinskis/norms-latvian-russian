import os
import base64
import uuid
import json
import logging
import audioop
from typing import Dict, List, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.languageconfig import AutoDetectSourceLanguageConfig
from openai import OpenAI


# ---------- Логгер ----------

logger = logging.getLogger("uvicorn.error")


# ---------- Конфиг из окружения ----------

SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")
DEFAULT_VOICE = os.environ.get("SPEECH_VOICE", "en-US-AmandaMultilingualNeural")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # https://...services.ai.azure.com/openai/v1/
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")        # deployment name, напр. car-assistant-gpt4o

if not SPEECH_KEY or not SPEECH_REGION:
    logger.warning("SPEECH_KEY or SPEECH_REGION is not set")

if not (OPENAI_API_KEY and OPENAI_BASE_URL and OPENAI_MODEL):
    logger.warning("OpenAI/Foundry config is incomplete "
                   "(OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL)")


# ---------- Инициализация клиентов ----------

app = FastAPI()

# TTS (HTTP) – обычный wav
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_synthesis_voice_name = DEFAULT_VOICE

# TTS для Twilio – 8kHz μ-law
twilio_speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
twilio_speech_config.speech_synthesis_voice_name = DEFAULT_VOICE
twilio_speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
)

# STT – автоопределение языка RU/LV
stt_speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
AUTO_DETECT_CONFIG = AutoDetectSourceLanguageConfig(
    languages=["lv-LV", "ru-RU"]
)

# OpenAI (Foundry / Azure OpenAI v1)
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

SYSTEM_PROMPT = (
    "Tu esi draudzīgs un profesionāls virtuālais asistents autoservisā. "
    "Atbildi īsi, skaidri un vienkāršā valodā. "
    "Ja lietotājs runā krieviski, atbildi krieviski; ja latviski – atbildi latviski."
)

SESSIONS: Dict[str, List[dict]] = {}


# ---------- Вспомогательные функции ----------

def synthesize_to_bytes(text: str) -> bytes:
    """TTS в обычный wav (для HTTP-эндпоинтов)."""
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,
    )
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    else:
        details = getattr(result, "cancellation_details", None)
        msg = str(details.reason) if details else "Unknown synthesis error"
        raise RuntimeError(msg)


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
        audio_bytes = result.audio_data  # raw μ-law
        return base64.b64encode(audio_bytes).decode("ascii")
    else:
        details = getattr(result, "cancellation_details", None)
        msg = str(details.reason) if details else "Unknown TTS error for Twilio"
        raise RuntimeError(msg)


def recognize_text_from_mulaw_bytes(mulaw_bytes: bytes) -> Tuple[str, str]:
    """
    Принимает байты в формате 8kHz μ-law (как шлёт Twilio),
    конвертит в 16-bit PCM и отдаёт в Azure STT.
    Возвращает (text, detected_language), где language ∈ { 'lv-LV', 'ru-RU', '' }.
    """
    if not mulaw_bytes:
        return "", ""

    # μ-law -> 16-bit PCM (2 байта на сэмпл)
    pcm16 = audioop.ulaw2lin(mulaw_bytes, 2)

    # 8kHz, 16-bit, mono PCM
    audio_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=8000,
        bits_per_sample=16,
        channels=1,
    )

    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
    push_stream.write(pcm16)
    push_stream.close()

    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=stt_speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=AUTO_DETECT_CONFIG,
    )

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = result.text or ""
        auto_result = speechsdk.AutoDetectSourceLanguageResult(result)
        detected_language = auto_result.language or ""
        return text, detected_language
    else:
        return "", ""


def run_dialog_turn(session_id: str, user_text: str, lang: str | None = None) -> Tuple[str, str]:
    """
    Одна реплика диалога: добавляем user_text в историю, вызываем GPT,
    сохраняем ответ. Возвращает (answer, session_id).
    """
    history = SESSIONS.get(session_id)
    if not history:
        # можем при желании менять system prompt в зависимости от lang
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

    history.append({"role": "user", "content": user_text})

    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=history,
        temperature=0.3,
    )
    answer = resp.choices[0].message.content

    history.append({"role": "assistant", "content": answer})
    SESSIONS[session_id] = history

    return answer, session_id


# ---------- HTTP-эндпоинты ----------

@app.get("/")
async def root():
    return {"status": "ok", "message": "Azure voice gateway is running"}


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
        logger.error("OpenAI/Foundry error: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


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
        logger.error("TTS error: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/tts-audio")
async def tts_audio(text: str):
    if not text:
        return JSONResponse({"error": "text query param is required"}, status_code=400)

    try:
        audio_bytes = synthesize_to_bytes(text)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error("TTS error: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/dialog")
async def dialog(payload: dict):
    """
    HTTP версия диалога – для тестов.
    Ожидает: { "text": "...", "session_id": "..."? }
    """
    user_text = payload.get("text") or ""
    if not user_text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    session_id = payload.get("session_id") or str(uuid.uuid4())

    try:
        answer, session_id = run_dialog_turn(session_id, user_text, None)
        audio_bytes = synthesize_to_bytes(answer)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        return {
            "answer": answer,
            "audio_base64": audio_b64,
            "session_id": session_id,
        }
    except Exception as e:
        logger.error("Dialog HTTP error: %s", e, exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------- Twilio: вебхук + WebSocket ----------

# Вебхук Twilio → возвращаем TwiML без приветствия, чтобы первым говорил клиент
@app.post("/voice", response_class=PlainTextResponse)
async def voice_webhook(request: Request):
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
    session_id = None

    audio_buffer = bytearray()
    responded = False  # пока делаем один полный цикл STT→GPT→TTS за звонок

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
                session_id = call_sid  # используем callSid как session_id для GPT
                logger.info(f"Twilio stream START callSid={call_sid}, streamSid={stream_sid}")

            elif event == "media":
                media = data.get("media", {})
                chunk = int(media.get("chunk", "0"))
                ts = int(media.get("timestamp", "0"))
                payload_b64 = media.get("payload")

                if payload_b64:
                    mulaw_bytes = base64.b64decode(payload_b64)
                    audio_buffer.extend(mulaw_bytes)

                logger.info(
                    f"Twilio media chunk={chunk}, ts={ts}, "
                    f"payload_len={len(payload_b64) if payload_b64 else 0}, "
                    f"buffer_size={len(audio_buffer)}"
                )

                # Как только клиент что-то сказал (первые пару секунд) – делаем один цикл
                if (not responded) and ts >= 2000 and len(audio_buffer) > 0 and stream_sid:
                    responded = True

                    try:
                        # STT с автоопределением RU/LV
                        text, lang = recognize_text_from_mulaw_bytes(bytes(audio_buffer))
                        logger.info(f"*** STT recognized text: {text!r}, language={lang}")

                        if text:
                            # GPT-диалог через общую логику
                            if not session_id:
                                session_id = str(uuid.uuid4())

                            answer, session_id = run_dialog_turn(session_id, text, lang)
                            logger.info(f"*** GPT answer: {answer!r}")

                            # TTS в μ-law для Twilio
                            twilio_payload = synthesize_mulaw_base64_for_twilio(answer)

                            reply = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": twilio_payload
                                },
                            }

                            await ws.send_text(json.dumps(reply))
                            logger.info("Sent GPT+TTS audio back to Twilio")

                        else:
                            logger.info("STT returned empty text – not sending response")

                    except Exception as e:
                        logger.error(f"Error in STT→GPT→TTS pipeline: {e}", exc_info=True)

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
