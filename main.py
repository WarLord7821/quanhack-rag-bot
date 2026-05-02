from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse

# Import the RAG logic from your existing module
from ai_engine import generate_response

# Initialize the FastAPI application
app = FastAPI(
    title="QuanHack Educational Enquiry Webhook",
    description="Webhook for routing Twilio messages to the RAG AI Engine."
)

@app.post("/webhook")
async def twilio_webhook(request: Request):
    """
    Handles incoming POST requests from Twilio.
    Extracts the message body, processes it through the AI engine,
    and returns a TwiML XML response.
    """
    # 1. Extract form data from the incoming request
    # Twilio sends data as application/x-www-form-urlencoded
    form_data = await request.form()
    
    # 2. Extract the user's message from the 'Body' field
    # We use .get() with a fallback empty string to prevent KeyError if 'Body' is missing
    user_message = form_data.get("Body", "")
    
    # Optional: Log the incoming message for debugging
    print(f"Received message: {user_message}")

    # 3. Pass the text to the AI engine to generate a response
    if user_message.strip():
        ai_answer = generate_response(user_message)
    else:
        # Fallback if the user sends an empty message or media without text
        ai_answer = "I didn't quite catch that. How can I help you with QuanHack Academy today?"

    # 4. Format the AI's answer using Twilio's MessagingResponse
    twiml_response = MessagingResponse()
    twiml_response.message(ai_answer)

    # 5. Return the valid TwiML XML to Twilio
    return Response(content=str(twiml_response), media_type="application/xml")

# ==========================================
# Execution Instructions
# ==========================================
# To run this server locally, use the Uvicorn ASGI server:
# $ uvicorn main:app --reload