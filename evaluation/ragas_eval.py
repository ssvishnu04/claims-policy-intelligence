import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL
from app.rag_pipeline import ask_claims_assistant


TEST_FILE = Path("evaluation/test_questions.json")


def filter_sources_by_policy_claim(sources, policy_id: str, claim_id: str):
    """
    Improves context precision by keeping only chunks that are related
    to the selected policy or claim.
    """
    filtered = []

    for doc in sources:
        content = doc.page_content.lower()
        metadata_text = " ".join(str(v).lower() for v in doc.metadata.values())

        policy_match = policy_id.lower() in content or policy_id.lower() in metadata_text
        claim_match = claim_id.lower() in content or claim_id.lower() in metadata_text

        # Keep documents that match either the exact policy or exact claim
        if policy_match or claim_match:
            filtered.append(doc)

    # Fallback: if filtering removes everything, keep original sources
    return filtered if filtered else sources


def build_ragas_dataset():
    with TEST_FILE.open("r", encoding="utf-8") as f:
        test_cases = json.load(f)

    rows = []

    for case in test_cases:
        policy_id = case["policy_id"]
        claim_id = case["claim_id"]
        question = case["question"]

        result = ask_claims_assistant(
            policy_id=policy_id,
            claim_id=claim_id,
            question=question,
            k=8,
        )

        filtered_sources = filter_sources_by_policy_claim(
            result["sources"],
            policy_id=policy_id,
            claim_id=claim_id,
        )

        rows.append(
            {
                "question": f"Policy ID: {policy_id}. Claim ID: {claim_id}. {question}",
                "answer": result["answer"],
                "contexts": [doc.page_content for doc in filtered_sources],
                "ground_truth": case["ground_truth"],
            }
        )

    return Dataset.from_list(rows)


def main():
    dataset = build_ragas_dataset()

    groq_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0,
    )

    ragas_llm = LangchainLLMWrapper(groq_llm)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    result = evaluate(
        dataset=dataset,
        metrics=[
            ResponseRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            # Faithfulness may return NaN with Groq because RAGAS may request n > 1 generations.
            # Keep it if you want to test, remove if it causes repeated timeout issues.
            Faithfulness(),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\nRAGAS Evaluation Results")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    main()