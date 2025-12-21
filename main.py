import os
import json
import base64
import logging
import asyncio
import threading
import audioop
import time
import uuid
from collections import deque
from difflib import SequenceMatcher

from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.responses import Response
from starlette.websockets import WebSocketDisconnect

import websockets
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-soniox-openai-eleven")

SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# Максимально простой промпт под автосервис
SYSTEM_PROMPT = (
    """
    You are the AI VOICE receptionist for AM Dental Studio, a dental clinic.

    GOAL
    Understand the caller’s need and, when appropriate, book a visit.
    You are not a doctor: do not diagnose or provide treatment plans; the dentist evaluates in-person.

    LANGUAGE (LV/RU) — IMPORTANT
    - Do NOT ask “which language do you prefer”.
    - Never mix Latvian and Russian in the same answer.

    VOICE STYLE (STRICT)
    - One sentence only, ideally 12–14 words.
    - Ask only ONE question per turn.
    - No repetition; if no answer, rephrase only once.
    - No greetings (assume the greeting already happened).
    - Never say “write”; say “please tell/say”.
    - Never ask for a phone number.

    CLINIC INFO (ONLY if asked)
    - Hours: Mon–Fri 08:00–16:00; weekends — by appointment.
    - Services: treatment, hygiene, prosthetics, implants, surgery, aligners/caps, adults and children.
    - Prices: “Exact price depends on your case; the dentist will confirm at consultation.”
    - Address: Rēzekne, Latgales iela 93.

    SPECIAL STT RULE
    If caller says “Mani sauc Zobs / mani sauc zobs”, interpret as “Man sāp zobs” (toothache).

    BOOKING LOGIC (highest priority)
    - If the caller asks to book immediately (e.g., “pierakstiet mani / gribu pierakstīties” or “запишите меня / хочу записаться”),
    never ask “do you want to book?” — immediately start booking.

    FLOW (one question per turn)
    1) If reason is unknown: ask the reason. If already stated, do not ask again.
    2) If caller asks only for info (address/hours/services/prices): answer briefly AND in the same sentence gently offer a visit.
    3) If reason is known and booking not offered yet: offer a visit with one question.
    If caller asks to book at any time: go to booking steps immediately (no re-offer).
    4) Booking steps:
    a) Ask first name + last name.
    b) Ask desired date.
    c) Offer 2 time options within 08:00–16:00 and ask which fits.
    d) If chosen time is taken: offer another time same date.
    e) If no times on that date: offer the nearest next available date/time.
    5) Closing: confirm date, time, reason, “AM Dental Studio”, and end politely.

    SAFETY
    If severe swelling, breathing difficulty, or uncontrolled bleeding: say to call emergency (112).
    """
)

GREETING_TEXT = "Labdien, AM Dental Studio. Kā varu palīdzēt?"

# ============================
#   STT: Soniox
# ============================

class VadGate:
    def __init__(self):
        self.noise = 300.0
        self.echo = 0.0
        self.speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        self._last_mode: bool | None = None  # False/True, чтобы сбрасывать counters при переключении

    def reset_echo(self):
        self.echo = 0.0

    def _reset_state(self):
        self.speaking = False
        self.speech_frames = 0
        self.silence_frames = 0

    def update(self, ulaw_bytes: bytes, *, tts_playback_active: bool) -> tuple[int, bool]:
        # сброс накоплений при переключении режима
        if self._last_mode is None:
            self._last_mode = tts_playback_active
        elif self._last_mode != tts_playback_active:
            self._last_mode = tts_playback_active
            self._reset_state()

        pcm = audioop.ulaw2lin(ulaw_bytes, 2)
        rms = audioop.rms(pcm, 2)

        if tts_playback_active:
            # оценка "эха" (уровень того, что возвращается, пока мы говорим)
            if self.echo == 0.0:
                self.echo = float(rms)
            else:
                self.echo = 0.98 * self.echo + 0.02 * float(rms)

            thr = max(2200.0, self.echo * 1.7 + 600.0)
            on_frames = 6    # >=120ms
            off_frames = 12  # >=240ms
        else:
            # обычный noise floor
            self.noise = 0.98 * self.noise + 0.02 * min(rms, 2000)
            thr = max(900.0, self.noise * 3.0 + 200.0)
            on_frames = 3
            off_frames = 10

        is_speech_now = rms > thr

        if is_speech_now:
            self.speech_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1
            self.speech_frames = 0

        if not self.speaking and self.speech_frames >= on_frames:
            self.speaking = True
        if self.speaking and self.silence_frames >= off_frames:
            self.speaking = False

        return rms, self.speaking

class SttSession:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws: websockets.WebSocketClientProtocol | None = None

    async def connect(self):
        if not self.api_key:
            raise RuntimeError("SONIOX_API_KEY is not set")

        # WebSocket Soniox
        self.ws = await websockets.connect("wss://stt-rt.soniox.com/transcribe-websocket")
        logger.info("Connected to Soniox WebSocket")

        # Конфиг для телефонного zvana (mulaw 8kHz, endpoint detection + LID) :contentReference[oaicite:2]{index=2}
        config_msg = {
            "api_key": self.api_key,
            "model": "stt-rt-v3",
            "audio_format": "mulaw",
            "sample_rate": 8000,
            "num_channels": 1,
            "enable_language_identification": True,
            "language_hints": ["ru", "lv"],
            "enable_endpoint_detection": True, #TODO:Luche postavitj False
            "client_reference_id": "twilio-call",
        }
        await self.ws.send(json.dumps(config_msg))
        logger.info("Sent Soniox config")

    async def send_audio(self, audio_bytes: bytes):
        if self.ws:
            await self.ws.send(audio_bytes)

    async def finalize(self):
        """Сообщаем Soniox, что аудио больше не будет."""
        if self.ws:
            try:
                await self.ws.send(b"")
            except Exception:
                pass

    async def finalize_segment(self, trailing_silence_ms: int = 300):
        """Принудительно финализировать текущий сегмент (manual finalization)."""
        if not self.ws:
            return
        msg = {"type": "finalize", "trailing_silence_ms": trailing_silence_ms}
        await self.ws.send(json.dumps(msg))

    async def receive_loop(self, handler):
        """Читаем все сообщения от Soniox и передаём в handler."""
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

    async def close(self):
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

    async def keepalive(self):
        if self.ws:
            await self.ws.send(json.dumps({"type": "keepalive"}))



# ============================
#   TTS: ElevenLabs -> Twilio
# ============================

class TtsSession:
    def __init__(self, eleven_client: ElevenLabs | None, ws: WebSocket, loop: asyncio.AbstractEventLoop):
        self.eleven_client = eleven_client
        self.ws = ws
        self.loop = loop
        self.stream_sid: str | None = None
        self._cancel_event = threading.Event()
        self._active = False
        self._start_ts = 0.0
        self._sent_any_audio = False

    def set_stream_sid(self, sid: str | None):
        self.stream_sid = sid

    def is_active(self) -> bool:
        return self._active
    
    def sent_any_audio(self) -> bool:
        return self._sent_any_audio

    def started_ago(self) -> float:
        return time.monotonic() - self._start_ts if self._start_ts else 0.0

    def cancel(self):
        """Запросить остановку текущего TTS-стрима (для barge-in)."""
        if self._active:
            logger.info("TTS cancel requested")
        self._cancel_event.set()

    async def speak(self, text: str) -> str | None:
        """Стрим ElevenLabs TTS обратно в Twilio через media-сообщения + mark в конце."""

        self._start_ts = time.monotonic()
        self._sent_any_audio = False

        if not self.eleven_client:
            logger.warning("ELEVENLABS_API_KEY is not set, skip TTS")
            return None
        if not self.stream_sid:
            logger.warning("streamSid is not set, skip TTS")
            return None
        if not text.strip():
            return None

        mark_name = f"tts_end_{uuid.uuid4().hex[:10]}"

        def _send(obj: dict):
            fut = asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps(obj)), self.loop
            )
            fut.result()

        # сбрасываем cancel и помечаем, что сейчас говорим
        self._cancel_event.clear()
        self._active = True
        logger.info("TTS start, text='%s'", text)

        def _run():
            logger.info("TTS thread started")
            try:
                response = self.eleven_client.text_to_speech.stream(
                    voice_id=ELEVENLABS_VOICE_ID,
                    model_id=ELEVENLABS_MODEL_ID,
                    text=text,
                    output_format="ulaw_8000",
                    voice_settings=VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.0,
                        style=0.0,
                        use_speaker_boost=True,
                        speed=1.2,
                    ),
                )

                frame_size = 160  # 20ms @ 8kHz mu-law
                buf = b""

                t0 = time.monotonic()
                frames_sent = 0

                for chunk in response:
                    if self._cancel_event.is_set():
                        logger.info("TTS streaming cancelled mid-stream")
                        break
                    if not chunk or not isinstance(chunk, (bytes, bytearray)):
                        continue

                    buf += bytes(chunk)

                    # отправляем ровными 20ms фреймами
                    while len(buf) >= frame_size:
                        if self._cancel_event.is_set():
                            break

                        part = buf[:frame_size]
                        buf = buf[frame_size:]

                        payload = base64.b64encode(part).decode("ascii")
                        _send({
                            "event": "media",
                            "streamSid": self.stream_sid,
                            "media": {"payload": payload},
                        })
                        self._sent_any_audio = True

                        # pacing: не быстрее реального времени
                        frames_sent += 1
                        target = t0 + frames_sent * 0.02
                        sleep_s = target - time.monotonic()
                        if sleep_s > 0:
                            time.sleep(sleep_s)

                # добиваем остаток (если есть)
                if not self._cancel_event.is_set() and buf:
                    payload = base64.b64encode(buf).decode("ascii")
                    _send({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": payload},
                    })
                    self._sent_any_audio = True

                # mark в самом конце (важно для tracking окончания TTS)
                _send({
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": mark_name},
                })

            except Exception as e:
                logger.exception("Error in ElevenLabs TTS streaming: %s", e)
            finally:
                self._active = False
                logger.info("TTS thread finished")

        await asyncio.to_thread(_run)
        return mark_name


# ============================
#   CallSession: Twilio + логика
# ============================

class CallSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.loop = asyncio.get_running_loop()
        self.stream_sid: str | None = None
        self.stt = SttSession(SONIOX_API_KEY)
        self.tts = TtsSession(eleven_client, ws, self.loop)
        self._last_stt_activity_ts = self.loop.time()
        self._last_finalize_sent_ts = 0.0
        self._finalize_inflight = False
        self.vad = VadGate()
        self.pre_roll = deque(maxlen=10)     # ~200ms если 20ms фреймы
        self.tts_playback_active = False     # шире чем tts.is_active()
        self.pending_marks: set[str] = set()
        self.barge_in_armed = False
        self.last_tts_text = ""
        self._last_soniox_send_ts = self.loop.time()

        self.last_tts_end_ts = 0.0
        self._turn_buf = ""
        self._turn_flush_task: asyncio.Task | None = None
        self._last_user_speech_ts = 0.0
        self.lang_locked: str | None = None  # "ru" | "lv"

        # контекст для GPT
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        self.user_utterance = ""
        self.llm_lock = asyncio.Lock()
        self._finished = False
        self._greeting_sent = False

    TURN_MERGE_WINDOW = 0.5  # 0.5–0.9 обычно оптимально

    def _arm_turn_flush(self):
        if self._turn_flush_task and not self._turn_flush_task.done():
            self._turn_flush_task.cancel()
        self._turn_flush_task = asyncio.create_task(self._turn_flush_worker())

    async def _turn_flush_worker(self):
        try:
            await asyncio.sleep(self.TURN_MERGE_WINDOW)
        except asyncio.CancelledError:
            return

        # если пользователь всё ещё говорил недавно — подождём ещё
        if self.loop.time() - self._last_user_speech_ts < self.TURN_MERGE_WINDOW:
            self._arm_turn_flush()
            return

        text = self._turn_buf.strip()
        self._turn_buf = ""
        if text:
            asyncio.create_task(self.handle_user_utterance(text))

    def _norm(self, s: str) -> str:
        s = s.lower().strip()
        s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
        return " ".join(s.split())

    def _detect_lang(self, text: str) -> str | None:
        t = text.lower()
        has_cyr = any("а" <= ch <= "я" or ch == "ё" for ch in t)
        if has_cyr:
            return "ru"
        latv_chars = set("āčēģīķļņšūž")
        if any(ch in latv_chars for ch in t):
            return "lv"
        has_ascii = any(ch.isalpha() and ch.isascii() for ch in t)
        if has_ascii:
            return "lv"
        return None

    async def _detect_lang_switch_via_llm(self, text: str) -> str | None:
        """
        Tool calling: переключаем язык только по явной просьбе.
        Игнорируем перечисления языков без команды переключения.
        """
        if not openai_client:
            return None

        tool = {
            "type": "function",
            "function": {
                "name": "detect_language_switch",
                "description": "Call only if the user explicitly asked to switch the conversation language. Ignore mentions of language proficiency or lists of languages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "language": {
                            "type": "string",
                            "enum": ["ru", "lv"],
                            "description": "Target language requested by the user ('ru' for Russian, 'lv' for Latvian).",
                        }
                    },
                    "required": ["language"],
                },
            },
        }

        messages = [
            {
                "role": "system",
                "content": "If the user explicitly asks to switch the conversation language to Russian or Latvian, call the detect_language_switch tool. Do not call it for general mentions of languages or proficiency lists.",
            },
            {"role": "user", "content": text},
        ]

        try:
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=[tool],
                tool_choice="auto",
                max_completion_tokens=32,
                temperature=0.0,
                reasoning_effort="none",
                verbosity="low",
            )
        except Exception as e:
            logger.exception("Error detecting language switch via tool call: %s", e)
            return None

        if not resp.choices:
            return None

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if not fn or fn.name != "detect_language_switch":
                continue
            try:
                args = json.loads(fn.arguments or "{}")
            except json.JSONDecodeError:
                continue
            lang = args.get("language")
            if lang in ("ru", "lv"):
                return lang
        return None

    def is_probable_echo(self, user_text: str) -> bool:
        a = self._norm(user_text)
        b = self._norm(self.last_tts_text)
        if len(a) < 12 or len(b) < 12:
            return False
        return SequenceMatcher(None, a, b).ratio() >= 0.82

    async def stt_segmenter_loop(self):
        """
        Если давно не приходили токены и есть накопленный текст — шлём Soniox {"type":"finalize"}.
        Без endpoint detection это заменяет ожидание <end>.
        """
        IDLE_SECS = 0.9          # можно тюнить
        MIN_INTERVAL_SECS = 2.0  # чтобы не флудить finalize (Soniox не рекомендует слишком часто)

        while not self._finished:
            await asyncio.sleep(0.1)

            if not self.user_utterance.strip():
                continue

            idle = self.loop.time() - self._last_stt_activity_ts
            if idle < IDLE_SECS:
                continue

            if self._finalize_inflight:
                continue

            if (self.loop.time() - self._last_finalize_sent_ts) < MIN_INTERVAL_SECS:
                continue

            self._finalize_inflight = True
            self._last_finalize_sent_ts = self.loop.time()

            try:
                await self.stt.finalize_segment(trailing_silence_ms=300)
            except Exception as e:
                logger.exception("Error sending Soniox finalize: %s", e)
                self._finalize_inflight = False

    async def send_clear(self):
        """Отправляем в Twilio 'clear' для бардж-ина."""
        if not self.stream_sid:
            return
        msg = {
            "event": "clear",
            "streamSid": self.stream_sid,
        }
        try:
            await self.ws.send_text(json.dumps(msg))
            logger.info("Sent Twilio clear for streamSid=%s", self.stream_sid)
        except Exception as e:
            logger.exception("Error sending Twilio clear: %s", e)

    async def barge_in(self, reason: str = ""):
        # уже сработало — повторно не делаем clear
        if self.barge_in_armed:
            return

        self.barge_in_armed = True
        logger.info("BARGE-IN (%s): cancelling TTS and clearing Twilio audio", reason)

        self.tts.cancel()
        await self.send_clear()

        # раз уж прервали TTS — считаем, что playback закончен
        self.tts_playback_active = False
        self.pending_marks.clear()

        # ВАЖНО: pre_roll НЕ чистим здесь!
        # Он должен быть отправлен в STT после barge_in в twilio_loop

        if hasattr(self.vad, "reset_echo"):
            self.vad.reset_echo()

    async def generate_gpt_reply(self, user_text: str) -> str:
        if not openai_client:
            logger.warning("OPENAI_API_KEY is not set, skip GPT call")
            return ""

        self.messages.append({"role": "user", "content": user_text})
        messages_for_call = list(self.messages)
        if self.lang_locked:
            lang_name = "Russian" if self.lang_locked == "ru" else "Latvian"
            messages_for_call.append({
                "role": "system",
                "content": f"Stick to {lang_name} only. Do not mix languages unless the user explicitly asks to switch.",
            })

        def _run_sync(msgs):
            assistant_text = ""
            try:
                stream = openai_client.chat.completions.create(
                    model="gpt-5.1",
                    messages=msgs,
                    stream=True,
                    max_completion_tokens=64,
                    temperature=0.25,
                    reasoning_effort="none",
                    verbosity="low",
                )
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    token = getattr(delta, "content", None)
                    if not token:
                        continue
                    piece = token if isinstance(token, str) else str(token)
                    assistant_text += piece
                    # logger.info("GPT partial: %s", assistant_text)
            except Exception as e:
                logger.exception("Error in GPT stream: %s", e)
            return assistant_text.strip()

        assistant_text = await asyncio.to_thread(_run_sync, messages_for_call)
        if assistant_text:
            self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    async def handle_user_utterance(self, text: str):
        text = text.strip()
        if not text:
            return

        # короткое эхо: "Labdien", "Jā" часто приходит как подстрока последнего TTS
        a = self._norm(text)
        b = self._norm(self.last_tts_text)
        if a and b and len(a) <= 12 and a in b:
            logger.info("Dropped short echo-substring: %r (in last_tts=%r)", text, self.last_tts_text)
            return

        # анти-эхо по similarity
        if self.is_probable_echo(text):
            logger.info("Dropped probable echo: %r", text)
            return

        lang_from_llm = await self._detect_lang_switch_via_llm(text)
        if lang_from_llm and lang_from_llm != self.lang_locked:
            self.lang_locked = lang_from_llm

        if not self.lang_locked:
            detected = self._detect_lang(text)
            if detected:
                self.lang_locked = detected

        async with self.llm_lock:
            logger.info("User utterance (final): %s", text)
            reply = await self.generate_gpt_reply(text)
            if not reply:
                return

            logger.info("GPT final reply: %s", reply)

            self.last_tts_text = reply
            self.vad.reset_echo()
            self.tts_playback_active = True
            self.barge_in_armed = False
            self.pre_roll.clear()

            mark = await self.tts.speak(reply)
            if mark:
                self.pending_marks.add(mark)
            else:
                # если TTS не стартанул/упал — не залипаем в playback_active
                self.tts_playback_active = False

    async def handle_stt_response(self, resp: dict):
        # ошибки Soniox
        if resp.get("error_code"):
            logger.error(
                "Soniox error %s: %s",
                resp.get("error_code"),
                resp.get("error_message"),
            )
            return

        tokens = resp.get("tokens", [])
        # if tokens:
        #     self._last_stt_activity_ts = self.loop.time()
        if not tokens:
            if resp.get("finished"):
                logger.info(
                    "Soniox finished: final_audio_proc_ms=%s total_audio_proc_ms=%s",
                    resp.get("final_audio_proc_ms"),
                    resp.get("total_audio_proc_ms"),
                )
                self._finished = True
            return

        # лог партиалов (для понимания задержки)
        partial_text = "".join(t.get("text", "") for t in tokens)
        if partial_text.strip():
            logger.info("Soniox partial: %s", partial_text)

        # перебираем токены
        for t in tokens:
            txt = t.get("text", "") or ""
            if not txt:
                continue

            # --- BARGE-IN: как только видим живой текст во время TTS ---
            # if txt.strip() and self.tts.is_active():
                # здесь можно указать reason, чтобы в логах было видно, по какому токену сработало
                # await self.barge_in(reason=f"token='{txt}'")
                # после этого TTS перестанет слать аудио, а Twilio очистит буфер
                # продолжаем обрабатывать текст как обычно (накапливаем фразу)

            # --- MANUAL FINALIZATION MARKER ---
            # if txt == "<fin>":
            #     final = self.user_utterance.strip()
            #     logger.info("Soniox FIN token received, final user text: '%s'", final)
            #     if final:
            #         asyncio.create_task(self.handle_user_utterance(final))
            #     self.user_utterance = ""
            #     self._finalize_inflight = False
            #     continue

            # --- ENDPOINT DETECTION (fallback, если когда-то включите обратно) ---
            if txt == "<end>":
                final = self.user_utterance.strip()
                logger.info("Soniox END token received, final user text: '%s'", final)
                self.user_utterance = ""

                if final:
                    self._turn_buf = (self._turn_buf + " " + final).strip() if self._turn_buf else final
                    self._arm_turn_flush()

                continue

            # --- накапливаем только финальные токены в текущую реплику пользователя ---
            if t.get("is_final"):
                self.user_utterance += txt

    async def twilio_loop(self):
        """Читаем события Twilio и шлём аудио в Soniox."""
        try:
            while True:
                msg = await self.ws.receive_text()
                data = json.loads(msg)
                event = data.get("event")

                if event == "connected":
                    logger.info("Twilio event=connected")

                elif event == "start":
                    start = data.get("start", {})
                    self.stream_sid = start.get("streamSid")
                    self.tts.set_stream_sid(self.stream_sid)
                    logger.info("Twilio stream START: %s", self.stream_sid)

                    # Сразу после старта стрима — приветствие ElevenLabs
                    if not self._greeting_sent:
                        self._greeting_sent = True
                        asyncio.create_task(self.send_greeting())

                elif event == "mark":
                    name = (data.get("mark") or {}).get("name")
                    if name:
                        logger.info("Twilio mark received: %s", name)
                        self.pending_marks.discard(name)

                    if not self.pending_marks:
                        self.tts_playback_active = False
                        self.barge_in_armed = False
                        self.pre_roll.clear()
                        if hasattr(self.vad, "reset_echo"):
                            self.vad.reset_echo()

                elif event == "media":
                    media = data.get("media", {})
                    payload_b64 = media.get("payload")
                    if not payload_b64:
                        continue
                    audio_bytes = base64.b64decode(payload_b64)

                    # TTS мог уже закончиться, но mark ещё не пришёл — снимаем gating и отдаём накопленный pre-roll в STT
                    if self.tts_playback_active and (not self.tts.is_active()):
                        if self.pre_roll:
                            try:
                                for b in self.pre_roll:
                                    await self.stt.send_audio(b)
                                    self._last_soniox_send_ts = self.loop.time()
                            except Exception:
                                pass
                        self.pre_roll.clear()
                        self.tts_playback_active = False
                        self.barge_in_armed = False
                        self.pending_marks.clear()
                        if hasattr(self.vad, "reset_echo"):
                            self.vad.reset_echo()

                    # VAD
                    rms, speaking = self.vad.update(audio_bytes, tts_playback_active=self.tts_playback_active)
                    if (not self.tts_playback_active) and speaking:
                        self._last_user_speech_ts = self.loop.time()

                    # WATCHDOG: если TTS активен, но ещё не отправил ни одного аудио слишком долго — снимаем gating
                    if self.tts_playback_active and self.tts.is_active() and (not self.tts.sent_any_audio()):
                        if self.tts.started_ago() > 2.0:
                            logger.warning("TTS stalled before first audio; disabling playback gating")
                            self.tts_playback_active = False
                            self.pending_marks.clear()
                            self.barge_in_armed = False
                            self.pre_roll.clear()
                            if hasattr(self.vad, "reset_echo"):
                                self.vad.reset_echo()

                    if self.tts_playback_active:
                        # копим pre-roll всегда, пока идёт playback
                        self.pre_roll.append(audio_bytes)

                        # barge-in только если TTS реально начал отдавать аудио (иначе вы отменяете "пустоту")
                        if (self.tts.is_active() and self.tts.sent_any_audio() and speaking and not self.barge_in_armed):
                            # берём snapshot pre-roll ДО barge_in (barge_in больше не чистит pre_roll, но так надежнее)
                            pre = list(self.pre_roll)
                            self.pre_roll.clear()

                            await self.barge_in(reason=f"VAD rms={rms}")

                            # отправляем pre-roll в STT, чтобы не потерять начало фразы
                            try:
                                for b in pre:
                                    await self.stt.send_audio(b)
                                    self._last_soniox_send_ts = self.loop.time()
                            except Exception:
                                pass

                        # пока не “перебили” — НЕ шлём в STT (анти-эхо)
                        if not self.barge_in_armed:
                            continue

                    # обычный режим — шлём в STT
                    await self.stt.send_audio(audio_bytes)
                    self._last_soniox_send_ts = self.loop.time()

                elif event == "stop":
                    logger.info("Twilio stream STOP received")
                    await self.stt.finalize()
                    break

                else:
                    logger.debug("Unhandled Twilio event: %s", event)

        except WebSocketDisconnect:
            logger.info("Twilio WebSocket disconnected")
            await self.stt.finalize()
        except Exception as e:
            logger.exception("Error in twilio_loop: %s", e)
            await self.stt.finalize()

    async def stt_keepalive_loop(self):
        while not self._finished:
            await asyncio.sleep(5)

            # если давно не слали аудио (например, во время TTS gating) — шлём keepalive
            if self.loop.time() - self._last_soniox_send_ts > 5:
                try:
                    await self.stt.keepalive()
                    logger.debug("Sent Soniox keepalive")
                except Exception:
                    pass

    async def stt_loop(self):
        """Цикл чтения результатов Soniox."""
        await self.stt.receive_loop(self.handle_stt_response)
        self._finished = True

    async def run(self):
        """Запускаем STT и Twilio петли параллельно."""
        try:
            await self.stt.connect()
        except Exception as e:
            logger.exception("Cannot start SttSession: %s", e)
            await self.ws.close()
            return

        try:
            await asyncio.gather(
                self.twilio_loop(),
                self.stt_loop(),
                # self.stt_segmenter_loop(),
                self.stt_keepalive_loop(),
            )
        finally:
            await self.stt.close()
            try:
                await self.ws.close()
            except Exception:
                pass

    async def send_greeting(self):
        """
        Первое приветствие через ElevenLabs.
        Сразу кладём его в контекст LLM как ответ ассистента.
        Приветствие тоже можно перебить, потому что идёт через общий TTS.
        """
        text = GREETING_TEXT.strip()
        if not text:
            return

        logger.info("Sending initial greeting TTS: %s", text)

        # Записываем приветствие как первую реплику ассистента,
        # чтобы модель знала, что мы уже поздоровались.
        self.messages.append({"role": "assistant", "content": text})

        # Произносим через тот же TTS-пайплайн (barge-in уже работает)
        self.vad.reset_echo()
        self.last_tts_text = text
        self.tts_playback_active = True
        self.barge_in_armed = False
        self.pre_roll.clear()

        mark = await self.tts.speak(text)
        if mark:
            self.pending_marks.add(mark)
        else:
            self.tts_playback_active = False


# ============================
#   HTTP endpoints
# ============================

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

    host = request.url.hostname

    # Простая приветственная фраза до подключения стрима
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="wss://{host}/twilio-stream" />
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
