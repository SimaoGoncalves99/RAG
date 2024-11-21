import os
from git import Repo
import shutil
import markdown
from langchain.text_splitter import MarkdownHeaderTextSplitter

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