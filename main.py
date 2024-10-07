import langchain
import os
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA

from langchain.vectorstores import Chroma
from langchain.schema import Document


class TextSplitter:
    def recursive(self,chunk_size: int, chunk_overlap: int):
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " "])
    def semantic(self, embeddings):
        return SemanticChunker(embeddings)
    

load_dotenv()

semantic_embedded = "text-embedding-3-small"
api_version="2024-02-15-preview"

azure_deployment_embedding = "text-embedding-ada-002"
azure_deployment_llm = "gpt-4-32k-last"
azure_deployment_llm_prev = "gpt-4-32k"
azure_deployment_version = "2023-07-01-preview"


embedded = AzureOpenAIEmbeddings(
        azure_deployment=azure_deployment_embedding,
        openai_api_version=azure_deployment_version,
    )

semantic_chunker = AzureOpenAIEmbeddings(
        azure_deployment=semantic_embedded,
        openai_api_version=api_version,
    )

# Initialize embeddings and Chroma vector store # Assuming you are using OpenAI embeddings
chroma = Chroma(embedding_function=embedded)  # Initialize with embeddings

# Define the LLM (e.g., OpenAI's GPT-3)
llm = AzureChatOpenAI(
    azure_deployment="gpt-4-32k-last",
    openai_api_version="2023-07-01-preview",
)

# Define the RAG chain (RetrieverQA Chain)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=chroma.as_retriever(),  # Use Chroma as retriever
    return_source_documents=True  # Return source documents along with the response
)

# PDF Parsing using Langchain
# Define the PDF file path
pdf_file = "/Users/alex/Documents/VSCode_project/ragagainsthemachine/Clean Code Principles And Patterns Python Edition (Petri Silén).pdf"

# Parse the PDF using PyPDFLoader
loader = PyPDFLoader(pdf_file)
documents = loader.load()

# Optionally, print the content of the parsed documents
for doc in documents:
    print(doc.page_content)

text_splitter_recursive = TextSplitter().recursive(chunk_size=500, chunk_overlap=50)
text_splitter_semantic = TextSplitter().semantic(semantic_chunker)


# split the documents
split_documents = []
i = 0
for doc in documents:
    i+=1
    print(i)
    chunks = text_splitter_recursive.split_text(doc.page_content)
    split_documents.extend([Document(page_content=chunk) for chunk in chunks])
    if i == 5:
        break

# Optionally, print the chunked documents
for chunk in split_documents:
    print(chunk.page_content)

# Index the documents
chroma.add_documents(split_documents)

# Define the prompt
prompt = "Write a summary of the documents."

# Generate text using the RAG chain
output = rag_chain({"query": prompt})

print(output['result'])  # Outputs the generated result
