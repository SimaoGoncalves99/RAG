import os
import time
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import PlainTextResponse
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

from docker_kb.data_utils import KB
from docker_kb.utils import generate_prompt

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
    t0 = time.time()

    response = {"code": 200, "result": ""}

    print("Retrieving text chunks...\n")
    # Fetch context
    retrieved_chunks = kb_obj.retrieve_context(
        request, top_k=10, method="cosine"
    )
    print(
        "Elapsed time for retrieving the text chunks: {:.1f} seconds\n".format(
            time.time() - t0
        )
    )

    t_prompt = time.time()
    print("Building prompt...\n")
    # Build LLM input
    messages, prompt = generate_prompt(
        query=request, retrieved_chunks=retrieved_chunks, one_shot=True
    )
    print(
        "Elapsed time for building the prompt: {:.1f} seconds\n".format(
            time.time() - t_prompt
        )
    )

    print("Awaiting for chatbot answer...")
    t_answer = time.time()
    # Send a request to a LLM model through an API
    chat_response = client.chat.complete(
        model="mistral-large-latest", messages=messages
    )
    print(
        "Elapsed time for the chatbot answer: {:.1f} seconds\n".format(
            time.time() - t_answer
        )
    )

    if chat_response is not None and chat_response.choices is not None:
        response["result"] = chat_response.choices[0].message.content

    print(f"Chatbot prompt:\n{prompt}\n\n")
    print(f"Chatbot answer:\n{response['result']}\n\n")

    print(
        "Elapsed time for processing the user request: {:.1f} seconds\n".format(
            time.time() - t0
        )
    )

    return PlainTextResponse(response["result"])


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0000", port=8000)
