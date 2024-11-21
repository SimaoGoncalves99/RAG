import argparse
import markdown
from data_utils import clone_data_repo,parse_md_files,chunk_documents
from sentence_transformers import SentenceTransformer
import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


def main(args):

    #Encoder initialization
    encoder = SentenceTransformer(args.embedding_model)

    if args.create_kb: #TODO Fix this when you are finished with the RAG
        #clone_data_repo(args.repo_url, args.clone_path,args.kb_path)

        kb = parse_md_files(args.kb_path)

        total_chunks = chunk_documents(kb)

        database = {}
        for enum,document in enumerate(total_chunks):
            path = document['path']
            database[path] = []
            for chunk in document['chunks']:
                output = query({
                    "inputs": {
                    "source_sentence": "That is a happy person",
                    "sentences": [
                        "That is a happy dog",
                        "That is a very happy person",
                        "Today is a sunny day"
                    ]
                },
                })
                print('hey')
                # embedding = encoder.encode(chunk.page_content)
                # database[path].append({'chunk':chunk.page_content,'embedding':embedding})


    return




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--kb_path", type=str,default="/mnt/data/docker_docs", help="Increase output verbosity")

    parser.add_argument("--create_kb", action="store_true", help="Clone the docker repo and save the knowledge base data locally")

    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder model")

    args = parser.parse_args()

    main(args)