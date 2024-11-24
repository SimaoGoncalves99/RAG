# docker-basics-rag
Author: Simão Gonçalves; E-mail: simao.campos.goncalves@gmail.com<br>

In this project I implement a system capable of handling requests related to common issues frequently faced by developers regarding containerization (Docker) and related technologies.<br>

To answer the users' questions I implement a [RAG](https://cloud.google.com/use-cases/retrieval-augmented-generation?hl=en) system.<br>

The idea is to have a knowledgebase (KB) to serve as factual grounding for a Large language model (LLM) that is prompted to address the users' questions.<br>
To answer the users' queries, the LLM is given context provided through the KB, so that its' answers are supported by actual documentation instead of being generic.<br>

The pipeline is established as follows:<br>

  - **KB**:<br>
    - The KB is built from the get-started docs of Docker here:https://github.com/docker/docs/tree/main/content/get-started;<br>
    - The documents in the KB are in `.md` format and must be pre-processed so that the LLM can digest them:<br>
        -  Each document is parsed into a string;<br>
        -  Each document's inital table with a general overview is removed;<br>
        -  Each document is chunked accordingly to its sections (`.md` headers denoted by hashes `#`).<br>
  - **Text Encoding**:<br>
    - Each chunk is encoded by a [Sentence Transformer model](https://sbert.net/), along with the user query. For this project I considered [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from HuggingFace.<br>
  - **Context Retrieval**:<br>
    -   The query's embedding is compared to the chunks' embeddings and the `top_k` chunks are returned.<br>
    -   The method used to compared the contextual representations is the **cosine similarity**. Nevertheless,<br>
    other methods can be use like the **euclidean distance** (also implemented).<br>
  - **Answer Generation**:<br>
    -   To answer the user's query I used the [MistralAI client](https://docs.mistral.ai/getting-started/clients/) API.<br>
    -   I prompted the LLM such that:<br>
        -  It has the task of addressing the users' questions;<br>
        -  It uses exclusively the *top_k* chunks to answer the request and not prior knowledge;<br>
        -  It returns the url for the files in the original [Docker repo](https://github.com/docker/docs/tree/main/content/get-started) (can also return local disk location).<br>

## Setup

### Fetching data for the KB
Start of by cloning the https://github.com/docker/docs/ repository sparsely<br>
By using the sparse-checkout, you are allowed to clone only a specific folder or set of folders (tree/main/content/get-started in this case)<br>

```
git clone --no-checkout https://github.com/docker/docs.git ./docs
cd ./docs
git sparse-checkout init --cone
git sparse-checkout set tree/main/content/get-started
git checkout
```
In *./docs* you can now find all the necessary `.md` files for the KB.<br> 

### Environment Setup
Create a virtual environment and install the required packages<br>

```
conda create --name RAG python=3.10.15
conda activate RAG
pip install .
pip install -r requirements.txt
```

## Testing the RAG System

To test the implemented system you can run the *main.py* script.<br>

E.g.<br>
From the project directory, run:<br>

```
python main.py --data_path ./docs --query How\ do\ I\ stop\ a\ docker\ container? --top_k 10 --score_method cosine
```

### Available configurations

  - **--embedding_model**: Model used for encoding the documents' chunks (default="sentence-transformers/all-MiniLM-L6-v2");
  - **--llm_model**: LLM API model (default="mistral-large-latest")
  - **--api_key**: API key for the Mistral AI LLM (default="2nn1vqvgifrwP8RjsvyLXmTy4dmtTYE3")
  - **--data_path"**: The path to the locally saved original documents as found in https://github.com/docker/docs/tree/main/content/get-started or the path to a saved .json file with the already processed knowledgebase data (default="./docs");
  - **--load_kb_content**: Enables loading the knowledgebase content saved locally;
  - **--save_kb**: Enables saving the processed knowledgebase content locally;
  - **--json_path**: Path to the file where the already processed knowledgebase data should be saved (used together with `--save_kb`);
  - **--query**: The user test query (default="What is docker?");
  - **--top_k**: Number of chunks to be used in the LLM context (default=10);
  - **--score_method**:The metric that is used for comparing the user queries to the knowledge base chunks;
  -  **--one_shot**: Optionally, you can feed the LLM with a synthetic one-shot example.

## Running a local API

The file `__main__.py` under the `api` folder defines an API that can be launched from the command line.<br>
After launching, it is possible access the API Swagger and to interact with the chatbot through HTTP post requests.<br>

To launch the API, first define the `API_KEY` and `DATA_PATH` environment variables and then run
`__main__.py`under the *docker-basics-rag/api* folder<br>

```
export API_KEY="2nn1vqvgifrwP8RjsvyLXmTy4dmtTYE3"
export DATA_PATH="./docs"
cd api
python __main__.py
```

If all is done correctly, you can easily access the Swagger through http://127.0.0.1:8000/docs for example.

## Containarized APis

In this project you can find under the *./apis* folder the files:<br>

- `.env`
- `docker-compose.yaml`
- `docker-compose-gcp`
- `dockerfile`

These files are used to containarize the API defined in `__main__.py`<br>

### Local container

By building and running with the `docker-compose.yaml` file we are able to run a local containarized API<br>

In the `.env` file set make sure you have `DOCKER_BASICS_RAG_VER=0.0.1`<br>

```
cd ./api
docker compose -f docker-compose.yaml build  docker_basics_rag --no-cache
docker compose -f docker-compose.yaml up -d --force-recreate  docker_basics_rag 
```

### GCP container

By building the `docker-compose-gcp.yaml` file we are able to build a docker image that can be later used in Google Cloud Platform<br>

In the `.env` file set make sure you have `DOCKER_BASICS_RAG_VER=0.0.2`<br>

```
cd ./api
docker compose -f docker-compose-gcp.yaml build  docker_basics_rag --no-cache
```


