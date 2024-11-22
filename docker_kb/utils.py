def generate_prompt(query,retrieved_chunks):

    context_info = ''
    for enum,chunk in enumerate(retrieved_chunks):
        if enum < len(retrieved_chunks)-1:
            context_info+=f"Document {enum+1}:\n{chunk['chunk']}\nReference:{chunk['path']}\n\n"
        else:
            context_info+=f"Document {enum+1}:\n{chunk['chunk']}\nReference:{chunk['path']}\n"

    prompt  = f"""You are a chatbot of the company \"Big Company\". Your main goal is to provide support for the company's developers, more specifically, you answer users' questions across various topics, such as containerization (Docker) and related technologies. You are capable of delivering company-specific answers to queries rather than generic answers. You will be answering mostly to common issues.\nYou will be given \"Context Information\" to answer the users' questions. Additionaly, provide in your answer the \"Reference\" of the sources that you consulted, so that the users can better understand what documents support your answer. It is better to direct the user to the source documents than to give information not supported by the \"Context Information\". \"Context Information\"is below.\n---------------------\n{context_info}---------------------\nGiven exclusively the \"Context Information\" and not prior knowledge, answer the query.\nQuery: {query}\nAnswer:"""
    
    messages = [
        {
            "role": "user", "content": prompt
        }
    ]

    return messages
    