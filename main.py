import argparse
from sentence_transformers import SentenceTransformer
import requests
from tqdm import tqdm 
import json
from docker_kb.data_utils import KB
from docker_kb.utils import generate_prompt
from mistralai import Mistral


def main(args):

    #Encoder initialization
    encoder_model = SentenceTransformer(args.embedding_model)

    #LLM API connection
    client = Mistral(api_key=args.api_key)

    #Knowledge base initialization
    kb_obj = KB(encoder_model=encoder_model,data_path=args.documents_path,load_kb=args.load_kb)
        
    #Save the knowledge base content
    if args.save_kb: #TODO Convert the embeddings to list before saving!
        with open(args.json_path,"w") as file:
            json.dump(kb_obj.database,file)

    #Fetch context
    retrieved_chunks = kb_obj.retrieve_context(args.query,top_k=3,method='cosine')

    #Build LLM input
    messages = generate_prompt(query=args.query,retrieved_chunks=retrieved_chunks)

    #Send a request to a LLM model through an API
    chat_response = client.chat.complete(
        model=args.llm_model,
        messages=messages
    )

    #Print the chatbot answer
    print(chat_response.choices[0].message.content)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--documents_path", type=str,default="/mnt/data/docker_docs", help="Increase output verbosity")

    parser.add_argument("--json_path", type=str,default="./kb_json", help="Increase output verbosity")

    parser.add_argument("--query", type=str, default="What is docker?", help="Encoder model")

    parser.add_argument("--top_k", type=int,default=3, help="Number of chunks to return")

    parser.add_argument("--load_kb", action="store_true", help="Load the knowledge base saved in memory")

    parser.add_argument("--api_key", type=str,default=" 2nn1vqvgifrwP8RjsvyLXmTy4dmtTYE3", help="API key for the Mistral AI LLM")


    parser.add_argument("--save_kb", action="store_true", help="Clone the docker repo and save the knowledge base data locally")

    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder model")

    parser.add_argument("--llm_model", type=str, default="mistral-large-latest", help="LLM API model")

    args = parser.parse_args()

    main(args)