from typing import Any, Dict, List, Tuple, Union

import numpy as np
from mistralai import models


def generate_prompt(
    query: str,
    retrieved_chunks: List[
        Dict[str, str | np.ndarray[Any, np.dtype[np.float32]]]
    ],
    one_shot: bool = False,
) -> Tuple[Union[List[models.Messages], List[models.MessagesTypedDict]], str]:
    """Generate a prompt to be fed to the LLM API.
        Give the LLM a persona with a well defined task (clarify on common issues
        faced by developers regarding containerization (Docker) and related technologies).
        Use the retrieved chunks to supply the LLM with relevant factual grounding
        to the user's query. Additionaly, use the 'one_shot' option to feed the LLM with
        an example input-output pair so that some guidance is provided.

    Args:
        query (str): User's question
        retrieved_chunks (List[Dict[str,str|np.ndarray[Any, np.dtype[np.float32]]]]): List of           dictionaries that holds the 'top_k' documents' chunks, their embeddings, and respective file path locations (local disk of GiHub location)
        one_shot (bool): Flags whether an input-output pair example is supplied to the LLM context

    Returns:
        messages (Union[List[models.Messages], List[models.MessagesTypedDict]): The prompt(s) to generate completions for, encoded as a list of dict with role and content
        prompt (str): The prompt in string format
    """

    context_info = ""
    for enum, chunk in enumerate(retrieved_chunks):
        if enum < len(retrieved_chunks) - 1:
            context_info += f"Document {enum+1}:\n{chunk['chunk']}\nReference:{chunk['path']}\n\n"
        else:
            context_info += f"Document {enum+1}:\n{chunk['chunk']}\nReference:{chunk['path']}\n"

    prompt = f"""You are a chatbot of the company \"Big Company\". Your main goal is to provide support for the company's developers, more specifically, you answer users' questions across various topics, such as containerization (Docker) and related technologies. You are capable of delivering company-specific answers to queries rather than generic answers. You will be answering mostly to common issues.\nYou will be given \"Context Information\" to answer the users' questions. Additionaly, provide in your answer the \"Reference\" of the sources that you consulted, so that the users can better understand what documents support your answer. It is better to direct the user to the source documents than to give information not supported by the \"Context Information\". \"Context Information\"is below.\n---------------------\n{context_info}---------------------\nGiven exclusively the \"Context Information\" and not prior knowledge, answer the query.\nQuery: {query}\nAnswer:"""

    if not one_shot:
        messages: Union[
            List[models.Messages], List[models.MessagesTypedDict]
        ] = [{"role": "user", "content": prompt}]

    else:
        ex_query = "how do i stop a Docker container?"
        ex_context_info = """Document 1:\nThe container continues to run until you stop it.  \n1. Go to the **Containers** view in the Docker Desktop Dashboard.  \n2. Locate the container you\'d like to stop.  \n3. Select the **Delete** action in the Actions column.  \n![A screenshot of Docker Desktop Dashboard showing how to delete the container](images/delete-the-container.webp?border=true)\nReference:https://github.com/docker/docs/tree/main/content/get-started/docker-concepts/running-containers/sharing-local-files.md\n\nDocument 2:\nThe `docker/welcome-to-docker` container continues to run until you stop it.  \n1. Go to the **Containers** view in the Docker Desktop Dashboard.  \n2. Locate the container you\'d like to stop.  \n3. Select the **Stop** action in the **Actions** column.  \n![Screenshot of the Docker Desktop Dashboard with the welcome container selected and being prepared to stop](images/stop-your-container.webp?border)  \n{{< /tab >}}\n{{< tab name="Using the CLI" >}}  \nFollow the instructions to run a container using the CLI:  \n1. Open your CLI terminal and start a container by using the [`docker run`](/reference/cli/docker/container/run/) command:  \n```console\n$ docker run -d -p 8080:80 docker/welcome-to-docker\n```  \nThe output from this command is the full container ID.  \nCongratulations! You just fired up your first container! 🎉\nReference:https://github.com/docker/docs/tree/main/content/get-started/docker-concepts/the-basics/what-is-a-container.md\n\nDocument 3:\n1. Get the ID of the container by using the `docker ps` command.  \n```console\n$ docker ps\n```  \n2. Use the `docker stop` command to stop the container. Replace `<the-container-id>` with the ID from `docker ps`.  \n```console\n$ docker stop <the-container-id>\n```  \n3. Once the container has stopped, you can remove it by using the `docker rm` command.  \n```console\n$ docker rm <the-container-id>\n```  \n> [!NOTE]\n>\n> You can stop and remove a container in a single command by adding the `force` flag to the `docker rm` command. For example: `docker rm -f <the-container-id>`  \n{{< /tab >}}\n{{< tab name="Docker Desktop" >}}\nReference:https://github.com/docker/docs/tree/main/content/get-started/workshop/03_updating_app.md"""
        ex_prompt = f"""You are a chatbot of the company \"Big Company\". Your main goal is to provide support for the company's developers, more specifically, you answer users' questions across various topics, such as containerization (Docker) and related technologies. You are capable of delivering company-specific answers to queries rather than generic answers. You will be answering mostly to common issues.\nYou will be given \"Context Information\" to answer the users' questions. Additionaly, provide in your answer the \"Reference\" of the sources that you consulted, so that the users can better understand what documents support your answer. It is better to direct the user to the source documents than to give information not supported by the \"Context Information\". \"Context Information\"is below.\n---------------------\n{ex_context_info}---------------------\nGiven exclusively the \"Context Information\" and not prior knowledge, answer the query.\nQuery: {ex_query}\nAnswer:"""

        ex_answer = """"To stop a Docker container, you can follow these steps:\n\n### Using Docker Desktop Dashboard:\n1. Go to the **Containers** view in the Docker Desktop Dashboard.\n2. Locate the container you'd like to stop.\n3. Select the **Stop** action in the **Actions** column.\n\n### Using the Command Line Interface (CLI):\n1. Get the ID of the container by using the `docker ps` command:\n   ```console\n   $ docker ps\n   ```\n2. Use the `docker stop` command to stop the container. Replace `<the-container-id>` with the ID from `docker ps`:\n   ```console\n   $ docker stop <the-container-id>\n   ```\n\nReferences:\n- https://github.com/docker/docs/tree/main/content/get-started/docker-concepts/the-basics/what-is-a-container.md\n- https://github.com/docker/docs/tree/main/content/get-started/workshop/03_updating_app.md"""

        messages: Union[
            List[models.Messages], List[models.MessagesTypedDict]
        ] = [
            {"role": "user", "content": ex_prompt},
            {"role": "assistant", "content": ex_answer},
            {"role": "user", "content": prompt},
        ]

    return messages, prompt
