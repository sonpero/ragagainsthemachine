import time as time
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

from langchain.vectorstores import Chroma

import config as config


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


class Vectorizer:
    def __init__(self, pdf_document_path, text_splitter, persist_directory):
        self.pdf_document_path = pdf_document_path
        self.book_name = self.pdf_document_path.split("/")[-1].split(".")[0]
        self.text_splitter = text_splitter
        self.persist_directory = persist_directory
        self.documents = None
        self.splitter = None
        self.split_documents = None

    def run(self):
        self.load_documents()
        self.split_the_documents()
        self.index_and_store_documents()
    
    def load_documents(self):
        loader = PyPDFLoader(self.pdf_document_path)
        self.documents = loader.load()
    
    def split_the_documents(self, mode="recursive"):
        if mode == "recursive":
            self.splitter = self.text_splitter.recursive(chunk_size=500, chunk_overlap=50)
        elif mode == "semantic":
            self.splitter = self.text_splitter.semantic(config.embedded_model)

        split_documents = []
        i = 0
        for doc in self.documents:
            i += 1
            chunks = self.splitter.split_text(doc.page_content)
            chunks_counter = 0
            for chunk in chunks:
                chunks_counter += 1
                split_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"page": i, 
                                  "chunk": chunks_counter, 
                                  "book_name": self.book_name},
                    )
                )

        self.split_documents = split_documents

    def index_and_store_documents(self):
        chroma = Chroma(embedding_function=config.embedded_model, persist_directory=self.persist_directory)
        # Index the documents
        batch_size = 10
        split_batches = [
            self.split_documents[i : i + batch_size]
            for i in range(0, len(self.split_documents), batch_size)
        ]
        for batch in split_batches:
            chroma.add_documents(batch)
            time.sleep(1)

        # Persist the Chroma vector store
        chroma.persist()


if __name__ == "__main__":
    # loading secrets
    load_dotenv()
    text_splitter = TextSplitter()

    # Define the PDF file path
    pdf_file = "/Users/alex/Documents/VSCode_project/ragagainsthemachine/clapnq_corpus.pdf"


    # run the vectorizer
    print("Vectorization started")
    vectorizer = Vectorizer(pdf_file, text_splitter, "./chroma_persistence/clapnq")
    vectorizer.run()
    print("Vectorization complete")