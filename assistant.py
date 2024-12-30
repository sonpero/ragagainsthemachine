import time as time
from langchain.vectorstores import Chroma
import config as config


# Initialize embeddings and Chroma vector store
persist_directory = "./chroma_persistence"
chroma = Chroma(embedding_function=config.embedded_model, persist_directory=persist_directory)

# Perform similarity search
while True:
    query = input("Enter your question (or 'quit' to exit)")
    if query.lower() in ['exit', 'quit']:
        print("Assistant: Goodbye!")
        break

    similar_documents = chroma.similarity_search(query, k=5)

    # Print results
    page_number = []
    for doc in similar_documents:
        # print("--- Similar Document ---")
        # print(doc.page_content)
        # print("Metadata:", doc.metadata)
        page_number.append(doc.metadata.get("page_number"))
        page_number = list(set(page_number))
        # print(page_number)

    # Retrieve documents by filtering on metadata
    page_number = [page_num for page_num in page_number if page_num > 5]

    # join all the pages to create the content to provide the retrieval
    retrieved_documents = []
    for page in page_number:
        metadata_filters = {
            "page_number": page,
        }

        results = chroma.get(where=metadata_filters)
        for doc in results.get("documents"):
            retrieved_documents.append(doc)
            # print(doc)

    # define the prompt and pass the retrieval chunks
    # Define the prompt
    # Combine the query and the retrieved chunks into a single context with a software development-specific prompt
    context = f"Assistant is a software development expert. The user asked the following question:\n\n'{query}'\n\n"

    # Add the relevant document chunks retrieved from Chroma
    context += "Here are some relevant pieces of information:\n\n"
    context += "\n\n".join([doc for doc in retrieved_documents])

    # The assistant should answer the question using the provided context.
    context += "\n\nPlease provide a detailed and accurate response with Python code examples when applicable, in the context of software development."

    # Pass the context to the LLM and generate a response
    response = config.llm(context)

    # Output the LLM response
    print("LLM Response:", response.content)
    print("---")