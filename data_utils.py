import os
from git import Repo
import shutil
from tqdm import tqdm
import markdown
from langchain.text_splitter import MarkdownHeaderTextSplitter
from pathlib import Path
import json 
import numpy as np
from numpy.linalg import norm

class KB:
    def __init__(self,data_path='',encoder_model=None,load_kb=False):
        self.encoder_model=encoder_model
        self.data_path = data_path
        self.load_kb = load_kb

        if self.load_kb:
            self.database = self.local_kb(self.data_path)
        else:
            self.database = self.create_kb(self.data_path)

    def create_kb(self,data_path):

        md_files = parse_md_files(data_path)

        chunked_docs = chunk_documents(md_files)

        self.database = []
        for doc in chunked_docs:
            path = doc['path']
            for chunk in tqdm(doc['chunks'],desc="Saving embedding data..."):
                embedding = (self.encoder_model).encode(chunk.page_content)
                (self.database).append({'chunk':chunk.page_content,'path':path,'embedding':embedding})

        return self.database
    
    def local_kb(self,data_path):

        if os.path.isfile(data_path) and data_path.endswith('.json'):
            self.database = json.load(data_path)
        else:
            raise RuntimeError("Please provide a valid path to a .json file")
        
        return self.database
    
    def retrieve_context(self,query,top_k=3,method='cosine'):
        
        query_embedding = (self.encoder_model).encode(query)

        # Step 1: Normalize the single embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        results = []
        embedding_matrix = np.array([d["embedding"] for d in self.database])  # Shape: (n, d)

        if method == 'cosine':
            similarities = np.dot(embedding_matrix,query_embedding)/(norm(embedding_matrix, axis=1)*norm(query_embedding))
            
            results = [{"cosine_similarity": similarities[i],"chunk": self.database[i]["chunk"],"path":self.database[i]['path']} for i in range(len(similarities))]
            
            results.sort(key=lambda item: item['cosine_similarity'],reverse=True)
        
        elif method == 'euclidean':
            distances = np.linalg.norm(embedding_matrix - query_embedding, axis=1)
            results = [{"euclidean_distance": distances[i],"chunk": self.database[i]["chunk"],"path":self.database[i]['path']} for i in range(len(distances))]
            results.sort(key=lambda item: item['euclidean_distance'])
        else:
            raise NotImplementedError
        
        retrieved_chunks = results[:top_k]

        return retrieved_chunks
            

        #     ({"cosine_similarity": cosine_similarity,"chunk": d["chunk"],"path":d['path']})
        # elif method == 'euclidean':

def clone_data_repo(repo_url, clone_path,save_dir):
    # Clone the repo sparsely
    sparse_clone_path = os.path.join(clone_path, "docker_docs_repo")
    repo = Repo.clone_from(repo_url, sparse_clone_path, multi_options=["--no-checkout"])
    repo.git.sparse_checkout("init", "--cone")

    folder_path=os.path.join(sparse_clone_path, "content/get-started")

    repo.git.sparse_checkout("set", folder_path)
    
    # Checkout the sparse content
    repo.git.checkout()
    
    # Move the folder to the final destination
    final_path = os.path.join(save_dir, os.path.basename(folder_path))
    os.rename(os.path.join(sparse_clone_path, folder_path), final_path)
    
    # Clean up the temporary repo
    shutil.rmtree(sparse_clone_path)

    print(f"Folder downloaded to: {final_path}")

def parse_md_files(kb_path):
    """Generate markdown documents from data path

    Args:
        kb_path (_type_): _description_

    Returns:
        _type_: _description_
    """

    kb = {}
    for root, dirs, files in os.walk(kb_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
    
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    kb[file_path] = text
        
    return kb

def chunk_documents(kb):
    
    total_chunks = []

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4"), ("#####", "Header 5"),("######", "Header 6")]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    for key in kb:
        sections = text_splitter.split_text(kb[key])

        total_chunks.append({'path':key,'chunks':sections})

    return total_chunks

