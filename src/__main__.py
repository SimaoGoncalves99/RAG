import os
import time
from fastapi import Body, FastAPI
from fastapi.responses import PlainTextResponse
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

from docker_kb.data_utils import KB
from docker_kb.utils import generate_prompt

import debugpy

debugpy.listen(("0.0.0.0", 5678))

description = """
Welcome to Big Company's IT Chatbot API!

This Chatbot is specialized in addressing common issues 
across various topics, such as containerization (Docker) and related technologies.

Please send a request through the /conversation/ POST request for the Chatbot to address your issue

## Request body description ##:

Write your query on the field `string` in the `Request body`. Just press `Try it out` to get started!

"""

app = FastAPI(
    title="Big Company's IT Chatbot", description=description, version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# Define the API endpoints
@app.get("/")
def health():
    return {"message": "OK 🚀"}


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
