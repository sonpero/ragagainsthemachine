import time as time
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain.vectorstores import Chroma


# class to define the split method
class TextSplitter:
    def recursive(self, chunk_size: int, chunk_overlap: int):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "],
            add_start_index=True,
        )

    def semantic(self, embeddings):
        return SemanticChunker(embeddings)


# loading secrets
load_dotenv()

# set LLM config for embedding
embedding_3_small = "text-embedding-3-small"
api_version = "2024-02-15-preview"

ada_002 = "text-embedding-ada-002"
azure_deployment_llm = "gpt-4-32k-last"
azure_deployment_llm_prev = "gpt-4-32k"
azure_deployment_version = "2023-07-01-preview"

# define embedded_model
embedded_model = AzureOpenAIEmbeddings(
    azure_deployment=embedding_3_small,
    openai_api_version=azure_deployment_version,
)

# Define the LLM used for the query
llm = AzureChatOpenAI(
    azure_deployment="gpt-4-32k-last",
    openai_api_version="2023-07-01-preview",
)

# PDF Parsing using Langchain
# Define the PDF file path
pdf_file = "/Users/alex/Documents/VSCode_project/ragagainsthemachine/Clean Code Principles And Patterns Python Edition (Petri Silén).pdf"
book_name = pdf_file.split("/")[-1].split(".")[0]

# Parse the PDF using PyPDFLoader
loader = PyPDFLoader(pdf_file)
documents = loader.load()

# Optionally, print the content of the parsed documents
for doc in documents:
    print(doc.page_content)
    break

# Define the text splitter
text_splitter_recursive = TextSplitter().recursive(chunk_size=500, chunk_overlap=50)
# text_splitter_semantic = TextSplitter().semantic(embedded_model)

# split the documents
split_documents = []
i = 0
for doc in documents:
    i += 1
    print(i)
    chunks = text_splitter_recursive.split_text(doc.page_content)
    chunks_counter = 0
    for chunk in chunks:
        chunks_counter += 1
        split_documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "page_number": doc.metadata.get("page_number", i),
                    "chunk_index": chunks_counter,
                    "book_name": book_name,
                },
            )
        )


# Optionally, print the chunked documents
for chunk in split_documents:
    print("--- Chunk ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
    break

# Initialize embeddings and Chroma vector store
persist_directory = "./chroma_persistence"
chroma = Chroma(embedding_function=embedded_model, persist_directory=persist_directory)

# Index the documents
batch_size = 10
split_batches = [
    split_documents[i : i + batch_size]
    for i in range(0, len(split_documents), batch_size)
]
for batch in split_batches:
    chroma.add_documents(batch)
    time.sleep(1)

# Persist the Chroma vector store
chroma.persist()
