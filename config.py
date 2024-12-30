from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
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
