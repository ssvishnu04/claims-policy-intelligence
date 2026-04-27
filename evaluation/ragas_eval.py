import os
import json
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL
from app.rag_pipeline import ask_claims_assistant


# Best-effort Groq compatibility
os.environ["RAGAS_GENERATION_COUNT"] = "1"

TEST_FILE = Path("evaluation/test_questions.json")


def build_ragas_rows() -> List[Dict[str, Any]]:
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
            k=20,
        )

        contexts = [doc.page_content for doc in result["sources"]]

        # Add structured claim profile to evaluation context
        if result.get("claim_profile"):
            contexts.insert(0, str(result["claim_profile"]))

        rows.append(
            {
                "question": f"Policy ID: {policy_id}. Claim ID: {claim_id}. {question}",
                "answer": result["answer"],
                "contexts": contexts,
                "ground_truth": case["ground_truth"],
            }
        )

    return rows


def get_ragas_llm():
    groq_llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0,
        max_tokens=2000,
        timeout=120,
        max_retries=1,
    )

    return LangchainLLMWrapper(groq_llm)


def get_ragas_embeddings():
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return LangchainEmbeddingsWrapper(embedding_model)


def run_metric(dataset: Dataset, metric, ragas_llm, ragas_embeddings):
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[metric],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
        )
        return result

    except Exception as e:
        return {metric.name: f"FAILED: {str(e)}"}


def simple_answer_relevancy(rows: List[Dict[str, Any]]) -> float:
    """
    Lightweight fallback for Answer Relevancy.

    Why this exists:
    RAGAS ResponseRelevancy can fail with Groq because some RAGAS versions
    request multiple generations, while Groq supports only n=1.

    This is not a replacement for full semantic evaluation, but it gives a
    practical approximation for portfolio/demo reporting.
    """
    scores = []

    stopwords = {
        "the", "is", "a", "an", "and", "or", "to", "of", "in", "for",
        "this", "that", "with", "what", "should", "does", "do", "me",
        "give", "policy", "claim", "id"
    }

    for row in rows:
        question_words = [
            word.strip(".,?:;!").lower()
            for word in row["question"].split()
            if word.strip(".,?:;!").lower() not in stopwords
        ]

        answer = row["answer"].lower()

        if not question_words:
            scores.append(0)
            continue

        matched_words = [
            word for word in question_words
            if word in answer
        ]

        score = len(matched_words) / len(question_words)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def main():
    rows = build_ragas_rows()
    dataset = Dataset.from_list(rows)

    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    metrics = [
        ContextPrecision(),
        ContextRecall(),
        Faithfulness(),
    ]

    print("\nRAGAS Evaluation Results")
    print("=" * 80)

    final_results = {}

    for metric in metrics:
        print(f"\nRunning metric: {metric.name}")

        metric_result = run_metric(
            dataset=dataset,
            metric=metric,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
        )

        print(metric_result)

        try:
            final_results.update(dict(metric_result))
        except Exception:
            final_results[metric.name] = str(metric_result)

    manual_relevancy_score = simple_answer_relevancy(rows)

    print("\nManual Answer Relevancy Approximation")
    print("=" * 80)
    print(round(manual_relevancy_score, 4))

    final_results["manual_answer_relevancy"] = round(manual_relevancy_score, 4)

    print("\nFinal Combined Results")
    print("=" * 80)
    print(final_results)


if __name__ == "__main__":
    main()
