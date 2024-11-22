import os
from tqdm import tqdm
from langchain.text_splitter import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from numpy.linalg import norm
from typing import Union,List,Dict
import re

class KB:
    """
    This class represents a knowledge base.
    It is responsible for several tasks such as:
        - Creating the knowledge base content i.e. to parse documents in markdown format and
            chunk them into several sections;
        - Loading the knowledge base content present in disk;
        - Encoding the chunks using a sentence transformer so that we can have contextual 
            representations for the different chunks;
        - Encoding the users' queries and use it to select the most relevant chunks for answering it.

    Args:
        data_path (str):
            Path to the directory where the markdown documents to be considered for the knowledge base are
            stored or path to the .json file holding the already chunked documents along with their embeddings,
            and file paths.
        encoder_model (SentenceTransformer): 
            The encoder model that will be used for encoding the knowledgebase documents and the user query
        load_kb (bool):
            Flags if we should load the document chunks and their embeddings from the disk or generate them
    """

    def __init__(self, encoder_model:SentenceTransformer, data_path:str='', load_kb_content:bool=False):
        self.encoder_model=encoder_model
        self.data_path = data_path
        self.load_kb_content = load_kb_content
        generate_kb = True

        #Load the knowledge base data locally saved
        if self.load_kb_content:
            if os.path.isfile(self.data_path):
                print(f'Loading knowledgebase...\n')
                self.database = self.local_kb_content(self.data_path)
                generate_kb = False
            else:
                print(f'Data file is inexistant!\n')

        #Generate the knowledgebase content (chunked documents and embeddings)
        if generate_kb:
            print(f'Generating knowledgebase...\n')
            self.database = self.create_kb(self.data_path)


    def create_kb(self, data_path:str)->List[Dict[str,str|np.ndarray]]:
        """Generate all of the knowledge base content.
        Start by parsing the markdown files located at 'data-path' and then 
        chunk them in smaller sections so that the LLM can select the adequate
        context more precisely. The documents' chunks are finally encoded by a 
        sentence transformer and the knowledgebase content is saved in a list
        of dictionaries ('self.database')

        Args:
            data_path (str): Path to the directory where the markdown documents to
            be considered for the knowledge base are stored

        Returns:
            self.database (List[Dict[str,Union[str,np.ndarray]]]): List of dictionaries
            that holds the documents' chunks, their embeddings, and respective file path locations
            (local disk of GiHub location)
        """

        #Parse the markdown documents
        md_files = parse_md_files(data_path)

        #Chunk the documents
        chunked_docs = chunk_documents(md_files)

        #Encode the documents' chunks and save the knowledge base content
        self.database = []
        for doc in chunked_docs:
            path = doc['path']
            for chunk in tqdm(doc['chunks'],desc="Saving embedding data..."):
                embedding = (self.encoder_model).encode(chunk.page_content)
                (self.database).append({'chunk':chunk.page_content,'path':path,'embedding':embedding})

        return self.database


    def local_kb_content(self, data_path:str)->List[Dict[str,str|np.ndarray]]:
        """Load the knowledgebase content from the local disk.

        Args:
            data_path (str): Path to the .json file holding the chunked documents
            along with their embeddings, and file paths.

        Raises:
            RuntimeError: Please provide a valid path to a .json file

        Returns:
            self.database (List[Dict[str,Union[str,np.ndarray]]]): List of dictionaries
            that holds the documents' chunks, their embeddings, and respective file path
            locations (local disk of GiHub location)
        """

        if os.path.isfile(data_path) and data_path.endswith('.json'):
            with open(data_path, "r") as file:
                self.database = json.load(file)
            #Convert embeddings from list to np.ndarray
            for item in (self.database):
                item['embedding'] = np.array(item['embedding'])
        else:
            raise RuntimeError("Please provide a valid path to a .json file")
        
        return self.database


    def retrieve_context(self, query, top_k=10, method='cosine')->List[Dict[str,str|np.ndarray]]:
        """Encode the user's query using the sentence transformer model
        and then either compute cosine similarity against all of the chunks
        embeddings or the euclidean distance. If the cosine similarity is computed,
        sort the chunks by descending order of similarity value. If the euclidean distance
        is computed, sort the distances by ascending order (the chunks that are more similar
        to the query will have their embeddings spatially closer to the query embedding). Select the
        'top_k' chunks which corresponds to selecting k most relevant chunks to the user's query.

        Args:
            query (str): The user's question.
            top_k (int, optional): Number of chunks to be used for LLM context. Defaults to 10.
            method (str, optional): Method used to compute similarity between user query
                and the knowledgebase chunks. Can assume the values 'cosine' for cosine similarity
                or 'euclidean' for euclidean distance. Defaults to 'cosine'.

        Raises:
            NotImplementedError: Select a valid embedding comparison method!

        Returns:
            retrieved_chunks (List[Dict[str,str|np.ndarray]]): List of dictionaries
            that holds the 'top_k' documents' chunks, their embeddings, and respective file path
            locations (local disk of GiHub location)
        """
        
        #Encode the user's query
        query_embedding = (self.encoder_model).encode(query)

        results = []
        #Encode each chunk in the knowledgebase
        embedding_matrix = np.array([d["embedding"] for d in self.database])

        #Compute cosine similarity between user's query embedding and documents' chunks embeddings
        #and select the 'top_k' chunks whose embeddings are most similar to the query's embedding
        if method == 'cosine':
            similarities = np.dot(embedding_matrix,query_embedding)/(norm(embedding_matrix, axis=1)*norm(query_embedding))
            
            results = [{"cosine_similarity": similarities[i],"chunk": self.database[i]["chunk"],"path":self.database[i]['path']} for i in range(len(similarities))]
            
            results.sort(key=lambda item: item['cosine_similarity'],reverse=True)
        
        #Compute euclidean distance between user's query embedding and documents' chunks embeddings
        #and select the 'top_k' chunks whose embeddings are spatially closer to the query's embedding
        elif method == 'euclidean':
            distances = np.linalg.norm(embedding_matrix - query_embedding, axis=1)
            results = [{"euclidean_distance": distances[i],"chunk": self.database[i]["chunk"],"path":self.database[i]['path']} for i in range(len(distances))]
            results.sort(key=lambda item: item['euclidean_distance'])
        else:
            raise NotImplementedError('Select a valid embedding comparison method!')
        
        #Select only the 'top_k' chunks
        retrieved_chunks = results[:top_k]

        return retrieved_chunks


def parse_md_files(kb_path):
    """Generate markdown documents from data path

    Args:
        kb_path (_type_): _description_

    Returns:
        _type_: _description_
    """

    kb = {}
    regex = r"(?s)^---.*?---"
    for root, dirs, files in os.walk(kb_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
    
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    #Delete the initial table from the md_files
                    text = re.sub(regex, "", text).strip()
                    if len(text) != 0:
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

