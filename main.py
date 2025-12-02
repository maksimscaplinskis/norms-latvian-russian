import os
import base64
import uuid
import json
import logging
import audioop
import asyncio
from typing import Dict, List, Tuple, Optional, Callable

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
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")        # deployment name, напр. car-assistant-mini

if not SPEECH_KEY or not SPEECH_REGION:
    logger.warning("SPEECH_KEY or SPEECH_REGION is not set")

if not (OPENAI_API_KEY and OPENAI_BASE_URL and OPENAI_MODEL):
    logger.warning("OpenAI/Foundry config is incomplete "
                   "(OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL)")


# ---------- Параметры "живости" диалога ----------

BASE_ENERGY_THRESHOLD = 800   # порог энергии (для грубого детектора "клиент говорит")
USER_TALKING_HOLD_MS = 600  # сколько мс после голоса считать, что клиент всё ещё говорит
NOISE_SMOOTHING = 0.98             # сглаживание среднего шума
NOISE_VOICE_FACTOR = 3.0           # во сколько раз голос громче шума

# ---------- Инициализация клиентов ----------

app = FastAPI()

# TTS (HTTP) – обычный wav (для тестовых эндпоинтов)
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

# Уменьшаем время "тишины в конце фразы", после которой STT отдаёт финальный текст
stt_speech_config.set_property(
    speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
    "300"  # 300 мс вместо дефолтных ~700–1500
)

# На всякий случай ограничим длинную начальную тишину
stt_speech_config.set_property(
    speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
    "3000"  # если абонент молчит > 3 сек, Azure всё равно завершит utterance
)

# OpenAI (Foundry / Azure OpenAI v1)
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

SYSTEM_PROMPT = (
    "Tu esi draudzīgs un profesionāls virtuālais asistents autoservisā. "
    "Atbildi īsi, skaidri un vienkāršā valodā. "
    "Ja lietotājs runā krieviski, atbildi krieviski; ja latviski – atbildi latviski. "
    "Atbildes garums – ne vairāk par 1–2 īsiem teikumiem."
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


def synthesize_mulaw_bytes_for_twilio(text: str) -> bytes:
    """
    Синтезирует речь в формате raw-8khz-8bit-mono-mulaw
    и возвращает байты (без WAV-заголовка).
    """
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=twilio_speech_config,
        audio_config=None,
    )
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data  # raw μ-law 8kHz
    else:
        details = getattr(result, "cancellation_details", None)
        msg = str(details.reason) if details else "Unknown TTS error for Twilio"
        raise RuntimeError(msg)


def run_dialog_turn(session_id: str, user_text: str, lang: Optional[str] = None) -> Tuple[str, str]:
    """
    Одна реплика диалога: добавляем user_text в историю, вызываем GPT,
    сохраняем ответ. Возвращает (answer, session_id).
    """
    history = SESSIONS.get(session_id)
    if not history:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

    history.append({"role": "user", "content": user_text})

    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=history,
        temperature=0.3,
        max_tokens=64,  # короткие ответы
    )
    answer = resp.choices[0].message.content

    history.append({"role": "assistant", "content": answer})
    SESSIONS[session_id] = history

    return answer, session_id


# ---------- HTTP-эндпоинты для теста ----------

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
            max_tokens=64,
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

@app.post("/voice", response_class=PlainTextResponse)
async def voice_webhook(request: Request):
    """
    Twilio webhook: без приветствия, клиент говорит первым.
    """
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


async def conversation_loop(
    ws: WebSocket,
    recognized_queue: "asyncio.Queue[Tuple[str, str]]",
    get_stream_sid: Callable[[], Optional[str]],
    initial_session_id: Optional[str],
    is_user_talking: Callable[[], bool],
):
    """
    Асинхронный цикл:
      - ждёт признанные фразы из Azure STT,
      - для каждой делает GPT → TTS,
      - отправляет аудио в Twilio кусками,
      - обрезает ответ, если клиент снова заговорил (barge-in).
    """
    session_id = initial_session_id

    try:
        while True:
            text, lang = await recognized_queue.get()
            if not text:
                continue

            text = text.strip()
            if not text:
                continue

            # убираем совсем короткий мусор, типа "ну", "ээ", "ģd"
            bare = text.replace(" ", "")
            if len(bare) < 3:
                logger.info(f"STT too short, ignoring: {text!r}")
                continue

            logger.info(f"STT final: text={text!r}, language={lang}")

            # GPT-ответ
            try:
                if not session_id:
                    session_id = str(uuid.uuid4())
                answer, session_id = run_dialog_turn(session_id, text, lang)
                logger.info(f"GPT answer: {answer!r}")
            except Exception as e:
                logger.error(f"GPT error: {e}", exc_info=True)
                if lang == "ru-RU":
                    answer = (
                        "Сейчас у меня технические проблемы. "
                        "Пожалуйста, коротко опишите, что происходит с вашей машиной."
                    )
                else:
                    answer = (
                        "Atvainojiet, man šobrīd ir tehniskas problēmas. "
                        "Lūdzu, īsumā pastāstiet, kas notiek ar jūsu auto."
                    )

            # TTS → μ-law
            try:
                audio_bytes = synthesize_mulaw_bytes_for_twilio(answer)
            except Exception as e:
                logger.error(f"TTS for Twilio error: {e}", exc_info=True)
                continue

            stream_sid = get_stream_sid()
            if not stream_sid:
                logger.warning("No streamSid, cannot send audio")
                continue

            # Отправляем аудио порциями ~20 ms (160 байт μ-law на 8 kHz)
            frame_size = 160  # 20 ms
            tts_start = asyncio.get_running_loop().time()
            min_tts_before_barge = 0.5  # 300 ms

            for i in range(0, len(audio_bytes), frame_size):
                # сколько уже проиграли
                elapsed = asyncio.get_running_loop().time() - tts_start

                # даём минимум 300ms, потом разрешаем бардж-ин
                if elapsed >= min_tts_before_barge and is_user_talking():
                    logger.info("User started talking, cutting off TTS (barge-in)")
                    break

                frame = audio_bytes[i:i + frame_size]
                if not frame:
                    continue

                payload = base64.b64encode(frame).decode("ascii")
                reply = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }
                await ws.send_text(json.dumps(reply))
                await asyncio.sleep(0.02)

            logger.info("Sent TTS audio back to Twilio")

    except asyncio.CancelledError:
        logger.info("Conversation loop cancelled")
    except Exception as e:
        logger.error(f"Error in conversation loop: {e}", exc_info=True)


@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    logger.info("Twilio WS connected")

    call_sid: Optional[str] = None
    stream_sid: Optional[str] = None

    # Очередь фраз, признанных Azure STT (text, lang)
    recognized_queue: "asyncio.Queue[Tuple[str, str]]" = asyncio.Queue()

    # Стейт для детектора "клиент говорит"
    shared_state = {
        "last_voice_ts": 0,
        "current_ts": 0,
        "noise_floor": 0.0,   # динамический уровень шума
    }

    def is_user_talking() -> bool:
        # динамический порог
        noise = shared_state["noise_floor"]
        threshold = max(BASE_ENERGY_THRESHOLD, noise * NOISE_VOICE_FACTOR)

        return (
            shared_state["last_voice_ts"] > 0
            and (shared_state["current_ts"] - shared_state["last_voice_ts"]) <= USER_TALKING_HOLD_MS
        )

    # STT объекты
    stt_stream: Optional[speechsdk.audio.PushAudioInputStream] = None
    recognizer: Optional[speechsdk.SpeechRecognizer] = None

    # таск диалога (STT → GPT → TTS)
    conversation_task: Optional[asyncio.Task] = None

    loop = asyncio.get_running_loop()

    first_utterance_sent = {"value": False}

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            event = data.get("event")

            if event == "connected":
                logger.info("Twilio event=connected")

            elif event == "start":
                start_info = data.get("start", {})
                call_sid = start_info.get("callSid")
                stream_sid = start_info.get("streamSid")
                logger.info(f"Twilio stream START callSid={call_sid}, streamSid={stream_sid}")

                initial_phrase = "Слушаю вас."  # или пустой бип, если сделаем заранее файл
                try:
                    audio_bytes = synthesize_mulaw_bytes_for_twilio(initial_phrase)
                    frame_size = 160
                    for i in range(0, len(audio_bytes), frame_size):
                        frame = audio_bytes[i:i + frame_size]
                        if not frame:
                            continue
                        payload = base64.b64encode(frame).decode("ascii")
                        reply = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload},
                        }
                        await ws.send_text(json.dumps(reply))
                        await asyncio.sleep(0.02)
                except Exception as e:
                    logger.error(f"Initial TTS error: {e}", exc_info=True)

                # Инициализируем STT continuous
                audio_format = speechsdk.audio.AudioStreamFormat(
                    samples_per_second=8000,
                    bits_per_sample=16,
                    channels=1,
                )
                stt_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
                audio_config = speechsdk.audio.AudioConfig(stream=stt_stream)

                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=stt_speech_config,
                    audio_config=audio_config,
                    auto_detect_source_language_config=AUTO_DETECT_CONFIG,
                )

                def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
                    result = evt.result
                    if result.reason == speechsdk.ResultReason.RecognizedSpeech and result.text:
                        auto_result = speechsdk.AutoDetectSourceLanguageResult(result)
                        lang = auto_result.language or ""
                        text = result.text
                        # передаём в очередь в event loop
                        loop.call_soon_threadsafe(
                            recognized_queue.put_nowait, (text, lang)
                        )

                def recognizing_cb(evt: speechsdk.SpeechRecognitionEventArgs):
                    # промежуточные результаты
                    if first_utterance_sent["value"]:
                        return  # для следующих фраз живём только на финалах

                    result = evt.result
                    if result.reason == speechsdk.ResultReason.RecognizingSpeech and result.text:
                        text = result.text.strip()
                        if len(text.replace(" ", "")) >= 3:
                            auto_result = speechsdk.AutoDetectSourceLanguageResult(result)
                            lang = auto_result.language or ""
                            first_utterance_sent["value"] = True
                            loop.call_soon_threadsafe(recognized_queue.put_nowait, (text, lang))

                recognizer.recognized.connect(recognized_cb)
                recognizer.recognizing.connect(recognizing_cb)

                # стартуем непрерывное распознавание (внутри своего потока)
                def start_recognition():
                    recognizer.start_continuous_recognition_async().get()

                loop.run_in_executor(None, start_recognition)

                # стартуем цикл диалога
                conversation_task = asyncio.create_task(
                    conversation_loop(
                        ws,
                        recognized_queue,
                        get_stream_sid=lambda: stream_sid,
                        initial_session_id=call_sid,
                        is_user_talking=is_user_talking,
                    )
                )

            elif event == "media":
                media = data.get("media", {})
                ts = int(media.get("timestamp", "0"))
                payload_b64 = media.get("payload")

                shared_state["current_ts"] = ts

                if not payload_b64 or not stt_stream:
                    continue

                mulaw_bytes = base64.b64decode(payload_b64)
                pcm16 = audioop.ulaw2lin(mulaw_bytes, 2)
                stt_stream.write(pcm16)

                # Оцениваем энергию текущего чанка
                energy = audioop.rms(pcm16, 2)  # RMS чуть стабильнее, чем max

                # Обновляем оценку шума (noise_floor) только для "не очень громких" сигналов
                noise = shared_state["noise_floor"]
                if noise == 0.0:
                    noise = float(energy)
                else:
                    # если энергия не слишком сильно выше текущего шума — считаем это фоном
                    if energy < noise * 1.5:
                        noise = NOISE_SMOOTHING * noise + (1.0 - NOISE_SMOOTHING) * energy

                shared_state["noise_floor"] = noise

                # Динамический порог: либо базовый, либо шум * фактор
                threshold = max(BASE_ENERGY_THRESHOLD, noise * NOISE_VOICE_FACTOR)

                # Если энергия сильно выше шума — считаем, что клиент говорит
                if energy > threshold:
                    shared_state["last_voice_ts"] = ts

            elif event == "stop":
                logger.info(f"Twilio stream STOP callSid={call_sid}, streamSid={stream_sid}")
                break

            else:
                logger.info(f"Twilio event other={event}")

    except WebSocketDisconnect:
        logger.info("Twilio WS disconnected (WebSocketDisconnect)")
    except Exception as e:
        logger.error(f"Twilio WS error: {e}", exc_info=True)
    finally:
        # Останавливаем STT
        if recognizer is not None:
            try:
                recognizer.stop_continuous_recognition_async().get()
            except Exception as e:
                logger.error(f"Error stopping recognizer: {e}", exc_info=True)
        if stt_stream is not None:
            try:
                stt_stream.close()
            except Exception:
                pass

        # Останавливаем цикл диалога
        if conversation_task is not None:
            conversation_task.cancel()
            try:
                await conversation_task
            except asyncio.CancelledError:
                pass

        logger.info("Twilio WS handler finished")
