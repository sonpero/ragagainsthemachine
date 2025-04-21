from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import SemanticSimilarity, FactualCorrectness
from ragas import evaluate
import config as config
import time

# Instancier les embeddings
embedding_model = config.embedded_model
embedding_wrapper = config.evaluator_embeddings

# Créer l’échantillon à évaluer
samples = SingleTurnSample(
    response="J'aime pas le chocolat.",
    reference="J'aime le chocolat."
)

# Instancier la métrique
dataset = EvaluationDataset(samples=[samples])
# Evaluate the LLM response using ragas
metrics = [
    # LLMContextRecall(llm=config.evaluator_llm),
    # FactualCorrectness(llm=config.evaluator_llm),
    # Faithfulness(llm=config.evaluator_llm),
    SemanticSimilarity(embeddings=config.evaluator_embeddings),
]

t1 = time.time()
results = evaluate(dataset=dataset, metrics=metrics)
t2 = time.time()
print(f"Evaluation time: {t2 - t1:.2f} seconds")
df = results.to_pandas()
print(df)
print('done')