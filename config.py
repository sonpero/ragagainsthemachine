import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv

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

azure_config = {
    "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "model_deployment": "gpt-4-32k-last",  # your model deployment name
    "model_name": "gpt-4-32k",
    "embedding_deployment": "text-embedding-3-small",  # your embedding deployment name
    "embedding_name": "text-embedding-3-small",  # your embedding name
}

evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(
    openai_api_version="2023-07-01-preview",
    azure_endpoint=azure_config["base_url"],
    azure_deployment=azure_config["model_deployment"],
    model=azure_config["model_name"],
    validate_base_url=False,
))

# init the embeddings for answer_relevancy, answer_correctness and answer_similarity
evaluator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
    openai_api_version="2023-07-01-preview",
    azure_endpoint=azure_config["base_url"],
    azure_deployment=azure_config["embedding_deployment"],
    model=azure_config["embedding_name"],
))
