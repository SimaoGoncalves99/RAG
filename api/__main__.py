from fastapi import FastAPI,APIRouter,Depends,HTTPException,status, Body
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import session
from pydantic import BaseModel
import uvicorn
from typing import Optional,Dict,Any
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from docker_kb.data_utils import KB
from docker_kb.utils import generate_prompt
import os


app = FastAPI()

#Encoder initialization
encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#LLM API connection
try:
    client = Mistral(api_key=os.environ.get("API_KEY"))
except:
    raise RuntimeError("Please provide a valid API key to access the chatbot")

if os.environ.get("DATA_PATH") is not None:
    #Knowledge base initialization
    kb_obj = KB(encoder_model=encoder_model,data_path=os.environ.get("DATA_PATH"))
else:
    raise RuntimeError


@app.post("/conversation/")
async def process_query(request:str = Body(..., media_type="text/plain"))->PlainTextResponse:

    #Fetch context
    retrieved_chunks = kb_obj.retrieve_context(request,top_k=3,method='cosine')

    #Build LLM input
    messages = generate_prompt(query=request,retrieved_chunks=retrieved_chunks)

    #Send a request to a LLM model through an API
    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages
    )

    response = chat_response.choices[0].message.content

    return PlainTextResponse(response)

if __name__ == "__main__":
    uvicorn.run("__main__:app",host="0000",port=8000)