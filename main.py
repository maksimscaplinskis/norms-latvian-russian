import logging

from fastapi import FastAPI, Form, Request
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio-voice")

app = FastAPI()


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
    """
    Webhook для входящих звонков Twilio.
    Twilio шлет сюда POST с form-data.
    Мы логируем вызов и отвечаем простым TwiML.
    """

    logger.info(f"Incoming call: CallSid={CallSid}, From={From}, To={To}")

    # Простой ответ на русском (можешь поменять текст и язык)
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say language="ru-RU" voice="woman">
            Это тестовый ответ с сервера в Азуре. Соединение с Twilio работает.
        </Say>
    </Response>"""

    # ВАЖНО: Twilio ожидает Content-Type: text/xml
    return Response(content=twiml.strip(), media_type="text/xml")
