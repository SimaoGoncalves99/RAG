import argparse
import markdown
from data_utils import KB
from sentence_transformers import SentenceTransformer
import requests
from tqdm import tqdm 
import json

query = "What is docker?"

def main(args):

    #Encoder initialization
    encoder_model = SentenceTransformer(args.embedding_model)

    kb_obj = KB(encoder_model=encoder_model,data_path=args.documents_path,load_kb=args.load_kb)
        
    if args.save_kb: #TODO Convert the embeddings to list before saving!
        with open(args.json_path,"w") as file:
            json.dump(kb_obj.database,file)

    #Fetch context
    context = kb_obj.retrieve_context(query,top_k=3,method='euclidean')

    print(docs)
    


    return




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--documents_path", type=str,default="/mnt/data/docker_docs", help="Increase output verbosity")

    parser.add_argument("--json_path", type=str,default="./kb_json", help="Increase output verbosity")

    parser.add_argument("--load_kb", action="store_true", help="Load the knowledge base saved in memory")

    parser.add_argument("--save_kb", action="store_true", help="Clone the docker repo and save the knowledge base data locally")

    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder model")

    args = parser.parse_args()

    main(args)