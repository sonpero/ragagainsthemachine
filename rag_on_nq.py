import time as time
import pandas as pd

from langchain.vectorstores import Chroma
import config as config
import utils as utils
from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate

# Initialize embeddings and Chroma vector store
persist_directory = "./chroma_persistence/clapnq"
chroma = Chroma(
    embedding_function=config.embedded_model, persist_directory=persist_directory
)

# load questions and answers from clapnq corpus
file_path = "clapnq_train_answerable.jsonl"
records = utils.read_jsonl(file_path)

i = 0
question = []
answer = []
for record in records:
    i += 1
    # question
    print(record["input"])
    question.append(record["input"])
    # answer
    print(record["output"][0]["answer"])
    answer.append(record["output"][0]["answer"])
    print("-----------------------")
    if i > 5:
        break


# Create EvaluationDataset with Ragas
samples = []
for record in records:
    sample = SingleTurnSample(
        user_input=record["input"],
        retrieved_contexts=[records[0]['passages'][0]['text']],
        response=record["output"][0]["answer"],
        reference=records[0]['passages'][0]['title'],
    )
    samples.append(sample)
    if i > 5:
        break

dataset = EvaluationDataset(samples=samples)
# Perform similarity search
query = question[0]

similar_documents = chroma.similarity_search_with_relevance_scores(query, k=10)
# for item, score in similar_documents:
#     print("--- Similarity Search Results ---")
#     print(f"Item: {item}, Relevance Score: {score}")


# similar_documents = chroma.similarity_search(query, k=5)

# Print results
page_number = []
for doc, score in similar_documents:
    if score > 0.4:
        print("--- Similar Document ---")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        page_number.append(doc.metadata.get("page"))
        page_number = list(set(page_number))
        print(page_number)

# Retrieve documents by filtering on metadata
page_number = [page_num for page_num in page_number if page_num > 0]

# join all the pages to create the content to provide the retrieval
retrieved_documents = []
for page in page_number:
    metadata_filters = {
        "page": page,
    }

    results = chroma.get(where=metadata_filters)
    for doc in results.get("documents"):
        retrieved_documents.append(doc)
        print(doc)

# define the prompt and pass the retrieval chunks
# Define the prompt
# Combine the query and the retrieved chunks into a single context with a software development-specific prompt
context = f"Assistant try to answer the following question with the provided context. The user asked the following question:\n\n'{query}'\n\n"

# Add the relevant document chunks retrieved from Chroma
context += "Here are some relevant pieces of information:\n\n"
context += "\n\n".join([doc for doc in retrieved_documents])

# The assistant should answer the question using the provided context.
context += "\n\nPlease provide a detailed and accurate response."

# Pass the context to the LLM and generate a response
response = config.llm(context)

# Output the LLM response
print(context)
print("LLM Response:", response.content)
print("---")

# Evaluate the LLM response using ragas
metrics = [
    LLMContextRecall(llm=config.evaluator_llm), 
    FactualCorrectness(llm=config.evaluator_llm), 
    Faithfulness(llm=config.evaluator_llm),
    SemanticSimilarity(embeddings=config.evaluator_embeddings)
]

results = evaluate(dataset=dataset, metrics=metrics)
df = results.to_pandas()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df)
print('this is the end')

