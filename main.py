import argparse
import json

import numpy as np
from mistralai import Mistral
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from docker_kb.data_utils import KB
from docker_kb.utils import generate_prompt


def main(args):

    # Encoder initialization
    encoder_model = SentenceTransformer(args.embedding_model)

    # MistralAI - LLM API connection
    client = Mistral(api_key=args.api_key)

    # Knowledgebase initialization
    kb_obj = KB(
        encoder_model=encoder_model,
        data_path=args.data_path,
        load_kb_content=args.load_kb_content,
    )

    # Save the knowledgebase content
    if args.save_kb:
        with open(args.json_path, "w") as file:
            # Convert the embeddings to list so that they can be saved in a .json file
            for item in tqdm(
                kb_obj.database,
                desc=f"Saving knowledgebase content at {args.json_path}",
            ):
                assert isinstance(
                    item["embedding"], np.ndarray
                ), "The embedding should be a np.ndarray variable"
                item["embedding"] = item["embedding"].tolist()
            json.dump(kb_obj.database, file)

    # Fetch query specific context for the LLM
    retrieved_chunks = kb_obj.retrieve_context(
        args.query, top_k=args.top_k, method=args.score_method
    )

    # Given the user query and the retrieved chunks, build the prompt to be fed to the LLM API
    messages, prompt = generate_prompt(
        query=args.query,
        retrieved_chunks=retrieved_chunks,
        one_shot=args.one_shot,
    )

    # Send a request to the LLM model through MistralAI API
    chat_response = client.chat.complete(
        model=args.llm_model, messages=messages
    )

    print(f"Chatbot prompt:\n{prompt}\n\n")
    if chat_response is not None and chat_response.choices is not None:
        print(f"Chatbot answer:\n{chat_response.choices[0].message.content}\n")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model used for encoding the documents' chunks",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="mistral-large-latest",
        help="LLM API model",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default="2nn1vqvgifrwP8RjsvyLXmTy4dmtTYE3",
        help="API key for the Mistral AI LLM",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/data/docker_docs",
        help="The path to the locally saved original documents as found in"
        + "https://github.com/docker/docs/tree/main/content/get-started or the path to a saved"
        + ".json file with the already processed knowledgebase data",
    )

    parser.add_argument(
        "--load_kb_content",
        action="store_true",
        help="Load the knowledgebase content saved locally",
    )

    parser.add_argument(
        "--save_kb",
        action="store_true",
        help="Save the knowledgebase content locally",
    )

    parser.add_argument(
        "--json_path",
        type=str,
        default="./kb.json",
        help="Path to the file with the already processed knowledgebase data",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="What is docker?",
        help="An example user query",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of chunks to be used in the LLM context",
    )

    parser.add_argument(
        "--score_method",
        type=str,
        choices=["cosine", "euclidean"],
        default="cosine",
        help="The metric that is used for comparing the user queries to the knowledge base chunks",
    )

    parser.add_argument(
        "--one_shot",
        action="store_true",
        help="Feed a one-shot example to the LLM",
    )

    args = parser.parse_args()

    main(args)
