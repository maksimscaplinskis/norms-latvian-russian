import os
import json
import base64
import logging
import audioop
import asyncio

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

import azure.cognitiveservices.speech as speechsdk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio")

app = FastAPI()

# ====  Azure Speech общая конфигурация  ====
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

if not SPEECH_KEY or not SPEECH_REGION:
    logger.warning("SPEECH_KEY or SPEECH_REGION is not set!")

speech_config = speechsdk.SpeechConfig(
    subscription=SPEECH_KEY,
    region=SPEECH_REGION,
)

# Авто-определение RU / LV
auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
    languages=["lv-LV", "ru-RU"]
)

# ====  OpenAI общая конфигурация  ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set!")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ====  ElevenLabs общая конфигурация  ====

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID")

if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY is not set!")

try:
    eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
except Exception as e:
    logger.exception(f"Failed to init ElevenLabs client: {e}")
    eleven_client = None


# Хранилище STT-сессий по streamSid
stt_sessions: dict[str, "AzureSTTSession"] = {}

# Хранилище LLM-сессий по streamSid
llm_sessions: dict[str, "LLMConversation"] = {}

# Хранилище Twilio WebSocket + event loop по streamSid (для отправки TTS)
twilio_connections: dict[str, tuple[WebSocket, asyncio.AbstractEventLoop]] = {}


# ====  OpenAI LLM  ====
class LLMConversation:
    """
    Одна LLM-сессия на один Twilio streamSid.
    Хранит контекст и ходит в OpenAI (streaming).
    """

    def __init__(self, stream_sid: str):
        self.stream_sid = stream_sid
        self.messages: list[dict] = []

        system_prompt = (
            "Ты голосовой ассистент автосервиса. "
            "Определи язык пользователя (русский или латышский) по его первым словам "
            "и отвечай строго на этом языке. "
            "Говори короткими, простыми фразами, не больше двух предложений за раз. "
            "Уточняй проблему с машиной и предлагай записаться на осмотр."
        )
        self.messages.append({"role": "system", "content": system_prompt})

    def handle_user_utterance(self, text: str, lang_code: str | None = None) -> str:
        """
        Добавляем фразу пользователя в контекст, вызываем OpenAI (stream=True),
        стримим токены в лог, возвращаем финальный текст ответа.
        """

        # if lang_code:
        #     self.messages.append({
        #         "role": "system",
        #         "content": f"Пользователь говорит на языке: {lang_code}. "
        #                    f"Отвечай на этом же языке."
        #     })

        self.messages.append({"role": "user", "content": text})
        logger.info(f"[{self.stream_sid}] LLM(OpenAI): sending user text: {text!r}")

        reply_text = ""

        try:
            stream = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self.messages,
                stream=True,
                max_completion_tokens=96,
                temperature=0.4,
                reasoning_effort="none"
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                reply_text += delta
                logger.info(f"[{self.stream_sid}] LLM partial: {delta!r}")

            logger.info(f"[{self.stream_sid}] LLM final: {reply_text!r}")
            self.messages.append({"role": "assistant", "content": reply_text})

        except Exception as e:
            logger.exception(f"[{self.stream_sid}] LLM error: {e}")
            reply_text = ""

        return reply_text


# ====  Azure STT  ====
class AzureSTTSession:
    """
    Одна STT-сессия Azure на один Twilio streamSid.
    Принимает μ-law 8kHz байты, конвертирует их в 16kHz 16-bit PCM
    и пишет в PushAudioInputStream.
    """

    def __init__(self, stream_sid: str, on_final_callback=None):
        self.stream_sid = stream_sid
        self._rate_state = None
        self.on_final_callback = on_final_callback

        # БЕЗ формата -> дефолт: 16kHz, 16bit, mono PCM
        self.push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

        self.recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect_config,
        )

        # События
        self.recognizer.recognizing.connect(self._on_recognizing)
        self.recognizer.recognized.connect(self._on_recognized)
        self.recognizer.canceled.connect(self._on_canceled)
        self.recognizer.session_started.connect(
            lambda evt: logger.info(f"[{self.stream_sid}] Azure STT session started")
        )
        self.recognizer.session_stopped.connect(
            lambda evt: logger.info(f"[{self.stream_sid}] Azure STT session stopped")
        )

        logger.info(f"[{self.stream_sid}] Creating Azure STT session")

        self.recognizer.start_continuous_recognition()

    def push_audio(self, mulaw_bytes: bytes):
        """
        Получает μ-law 8kHz байты от Twilio,
        конвертирует в 16-bit PCM 16kHz и отправляет в Azure.
        """
        if not mulaw_bytes:
            return

        # 1) μ-law (8bit) -> PCM 16bit, 8kHz
        pcm16_8k = audioop.ulaw2lin(mulaw_bytes, 2)  # 2 байта на сэмпл

        # 2) 8kHz -> 16kHz (ratecv поддерживает состояние между вызовами)
        pcm16_16k, self._rate_state = audioop.ratecv(
            pcm16_8k,
            2,      # ширина сэмпла (байты)
            1,      # каналов
            8000,   # in_rate
            16000,  # out_rate
            self._rate_state,
        )

        # 3) Пишем в поток Azure
        self.push_stream.write(pcm16_16k)

    def stop(self):
        try:
            logger.info(f"[{self.stream_sid}] Stopping Azure STT session")
            self.push_stream.close()
            self.recognizer.stop_continuous_recognition()
        except Exception as e:
            logger.exception(f"[{self.stream_sid}] Error while stopping STT: {e}")

    # ----- callbacks -----

    def _on_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs):
        result = evt.result
        text = result.text
        if not text:
            return

        lang = None
        # пробуем достать определённый язык
        props = result.properties
        lang = props.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
        )

        logger.info(
            f"[{self.stream_sid}] Recognizing (partial) "
            f"[lang={lang}]: {text}"
        )

    def _on_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs):
        result = evt.result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = result.text
            props = result.properties
            lang_raw = props.get(
                speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
            )

            logger.info(
                f"[{self.stream_sid}] Recognized (final) [lang={lang_raw}]: {text!r}"
            )

            # Вызовем колбэк (LLM)
            if self.on_final_callback and text:
                try:
                    self.on_final_callback(self.stream_sid, text, lang_raw)
                except Exception as e:
                    logger.exception(
                        f"[{self.stream_sid}] Error in on_final_callback: {e}"
                    )

        elif result.reason == speechsdk.ResultReason.NoMatch:
            logger.info(f"[{self.stream_sid}] NoMatch: {result.no_match_details}")

    def _on_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs):
        logger.warning(
            f"[{self.stream_sid}] CANCELED: reason={evt.reason}, "
            f"error_details={evt.error_details}"
        )


# ====  Twilio Media Stream: WebSocket с аудио  ====
@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    logger.info("Twilio WS connected")

    stream_sid = None
    loop = asyncio.get_running_loop()

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            event = data.get("event")

            if event == "connected":
                logger.info("Twilio event=connected")

            elif event == "start":
                start_info = data["start"]
                stream_sid = start_info["streamSid"]
                logger.info(f"Twilio stream START streamSid={stream_sid}")

                twilio_connections[stream_sid] = (ws, loop)

                # создаём STT сессию под этот streamSid
                if stream_sid in stt_sessions:
                    logger.warning(
                        f"STT session for {stream_sid} already exists, overwriting"
                    )
                stt_sessions[stream_sid] = AzureSTTSession(
                    stream_sid,
                    on_final_callback=handle_final_transcript,
                )

            elif event == "media":
                if not stream_sid:
                    # на всякий случай
                    logger.warning("Got media before start; skipping")
                    continue

                payload_b64 = data["media"]["payload"]
                mulaw_bytes = base64.b64decode(payload_b64)

                session = stt_sessions.get(stream_sid)
                if session:
                    session.push_audio(mulaw_bytes)

            elif event == "stop":
                logger.info(f"Twilio stream STOP streamSid={stream_sid}")
                if stream_sid and stream_sid in stt_sessions:
                    stt_sessions[stream_sid].stop()
                    del stt_sessions[stream_sid]

                # чистим Twilio WebSocket для этого streamSid
                if stream_sid and stream_sid in twilio_connections:
                    del twilio_connections[stream_sid]

                break

            else:
                logger.info(f"Unknown Twilio event: {event}")

    except WebSocketDisconnect:
        logger.info("Twilio WS disconnected")
    except Exception as e:
        logger.exception(f"Error in twilio_stream: {e}")
    finally:
        # на всякий случай чистим сессию, если осталась
        if stream_sid and stream_sid in stt_sessions:
            stt_sessions[stream_sid].stop()
            del stt_sessions[stream_sid]

        if stream_sid and stream_sid in twilio_connections:
            del twilio_connections[stream_sid]

        logger.info("Twilio handler finished")


# ====  Twilio webhook: TwiML, подключающий Media Stream  ====
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


def handle_final_transcript(stream_sid: str, text: str, lang_raw: str | None):
    """
    Вызывается, когда Azure STT распознал финальный текст.
    Создаёт/берёт LLMConversation и отправляет туда фразу.
    Пока что только логируем ответ LLM.
    """
    # Попробуем вытащить код языка из JSON строки Azure
    lang_code = None
    if lang_raw:
        try:
            obj = json.loads(lang_raw)
            # Azure иногда использует "language" или "Language"
            lang_code = obj.get("language") or obj.get("Language")
        except Exception:
            lang_code = lang_raw  # ну хоть что-то

    logger.info(
        f"[{stream_sid}] Final transcript for LLM "
        f"[lang_raw={lang_raw!r}, lang_code={lang_code!r}]: {text!r}"
    )

    # Берём/создаём LLM-сессию
    if stream_sid not in llm_sessions:
        llm_sessions[stream_sid] = LLMConversation(stream_sid)

    conv = llm_sessions[stream_sid]

    # ⚠️ Внимание: это синхронный вызов, но он идёт из потока Azure STT,
    # не блокируя обработку Twilio WebSocket.
    reply_text = conv.handle_user_utterance(text, lang_code=lang_code)

    if reply_text:
        logger.info(f"[{stream_sid}] LLM reply ready (for TTS): {reply_text!r}")

        # сразу запускаем стрим TTS -> Twilio
        try:
            stream_tts_to_twilio(stream_sid, reply_text)
        except Exception as e:
            logger.exception(f"[{stream_sid}] Error while streaming TTS: {e}")


def stream_tts_to_twilio(stream_sid: str, text: str):
    """
    Сгенерировать речь через ElevenLabs и постримить её в Twilio как
    media-сообщения (mulaw/8000 base64) в bidirectional Media Stream.
    Работает из обычного (не-async) потока.
    """
    if not eleven_client:
        logger.warning(f"[{stream_sid}] ElevenLabs client not initialized, skip TTS")
        return
    if not ELEVENLABS_VOICE_ID:
        logger.warning(f"[{stream_sid}] ELEVENLABS_VOICE_ID is not set, skip TTS")
        return

    conn = twilio_connections.get(stream_sid)
    if not conn:
        logger.warning(f"[{stream_sid}] No Twilio WebSocket for this streamSid, cannot send TTS")
        return

    ws, loop = conn

    logger.info(f"[{stream_sid}] TTS: sending text to ElevenLabs ({len(text)} chars)")

    try:
        # Важно: формат ulaw_8000 – Twilio требует audio/x-mulaw 8kHz, base64.
        audio_stream = eleven_client.text_to_speech.stream(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL_ID,
            text=text,
            output_format="ulaw_8000",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_byle=0.0,
                use_speaker_boost=True,
                speed=1.3,
            ),
        )

        for chunk in audio_stream:
            if not chunk or not isinstance(chunk, (bytes, bytearray)):
                continue

            payload_b64 = base64.b64encode(chunk).decode("ascii")

            def _send():
                # Эта функция уже выполняется в event loop’е
                if ws.application_state.name != "CONNECTED":
                    logger.warning(f"[{stream_sid}] WebSocket not connected, stop TTS send")
                    return
                msg = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload_b64},
                }
                # создаём таск на отправку
                asyncio.create_task(ws.send_text(json.dumps(msg)))

            # Планируем отправку чанка в event loop из нашего потока
            loop.call_soon_threadsafe(_send)

    except Exception as e:
        logger.exception(f"[{stream_sid}] ElevenLabs TTS error: {e}")        