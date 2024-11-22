from fastapi import FastAPI, Body
from fastapi.responses import PlainTextResponse
import uvicorn
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from docker_kb.data_utils import KB
from docker_kb.utils import generate_prompt
import os


app = FastAPI()

# Encoder initialization
encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# LLM API connection
try:
    client = Mistral(api_key=os.environ.get("API_KEY"))
except:
    raise RuntimeError("Please provide a valid API key to access the chatbot")

if os.environ.get("DATA_PATH") is not None:
    # Knowledge base initialization
    kb_obj = KB(
        encoder_model=encoder_model, data_path=os.environ.get("DATA_PATH")
    )
else:
    raise RuntimeError


@app.post("/conversation/")
async def process_query(
    request: str = Body(..., media_type="text/plain")
) -> PlainTextResponse:

    response = {"code": 200, "result": ""}

    # Fetch context
    retrieved_chunks = kb_obj.retrieve_context(
        request, top_k=10, method="cosine"
    )

    # Build LLM input
    messages, prompt = generate_prompt(
        query=request, retrieved_chunks=retrieved_chunks, one_shot=True
    )

    # Send a request to a LLM model through an API
    chat_response = client.chat.complete(
        model="mistral-large-latest", messages=messages
    )

    if chat_response is not None and chat_response.choices is not None:
        response["result"] = chat_response.choices[0].message.content

    print(f"Chatbot prompt:\n{prompt}\n\n")
    print(f"Chatbot answer:\n{response['result']}\n")

    return PlainTextResponse(response["result"])


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0000", port=8000)
