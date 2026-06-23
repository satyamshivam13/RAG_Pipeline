"""RAGAS evaluation runner for RAG_Pipeline.

Runs the real pipeline over a grounded question set and scores it with RAGAS
(faithfulness, answer relevancy, context precision, context recall).

The judge LLM is read from the same env the pipeline uses (OPENAI_API_KEY /
LLM_BASE_URL / LLM_MODEL), so it works with OpenAI-compatible providers such as
Groq. Embeddings for the relevancy metric use the local sentence-transformers
model so no OpenAI embeddings endpoint is required.

Usage:
    python -m evaluation.run_ragas \
        --corpus evaluation/datasets/ragas_corpus.json \
        --dataset evaluation/datasets/ragas_eval.jsonl \
        --output evaluation/reports/ragas-latest.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _load_corpus(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [p["text"] for p in data["passages"]]


def _load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_judge():
    """Wire RAGAS to the configured (OpenAI-compatible) LLM + a local embedder."""
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("LLM_BASE_URL") or None,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            timeout=120,
            max_retries=2,
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    return judge_llm, judge_emb


def run(corpus_path: Path, dataset_path: Path, output_path: Path, limit: int | None) -> dict:
    from config import PipelineConfig, RuntimeConfig
    from main import RAGPipeline

    passages = _load_corpus(corpus_path)
    rows = _load_dataset(dataset_path)
    if limit:
        rows = rows[:limit]

    print(f"Corpus passages: {len(passages)} | eval questions: {len(rows)}", flush=True)

    pipeline = RAGPipeline(PipelineConfig(runtime=RuntimeConfig(evaluator_mode="sync")))
    n_chunks = pipeline.ingest(passages, source="ragas_corpus")
    print(f"Ingested {n_chunks} chunks", flush=True)

    questions, answers, contexts, references, latencies, consistency = [], [], [], [], [], []
    for row in rows:
        t0 = time.perf_counter()
        result = pipeline.query(row["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved = [rc.chunk.content for rc in result.retrieval] or []
        questions.append(row["question"])
        answers.append(result.answer)
        contexts.append(retrieved)
        references.append(row["ground_truth"])
        consistency.append(result.consistency_score)
        print(f"  {row['id']}: {latencies[-1]:6.0f} ms", flush=True)
    pipeline.close()

    # --- RAGAS scoring ---
    from ragas import evaluate, EvaluationDataset, RunConfig
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    judge_llm, judge_emb = _build_judge()
    dataset = EvaluationDataset.from_list(
        [
            {"user_input": q, "response": a, "retrieved_contexts": c, "reference": r}
            for q, a, c, r in zip(questions, answers, contexts, references)
        ]
    )
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    # Conservative concurrency: free-tier OpenAI-compatible providers (e.g. Groq)
    # rate-limit aggressively; keep workers low and let RAGAS retry on 429.
    run_config = RunConfig(max_workers=3, timeout=180, max_retries=8, max_wait=90)

    print("Running RAGAS (this makes many judge-LLM calls)...", flush=True)
    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
    )

    scores = {k: (round(float(v), 4) if v is not None else None) for k, v in dict(eval_result).items()}
    p95 = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
    report = {
        "evaluator": "ragas",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": os.getenv("LLM_MODEL"),
        "judge_provider_base_url": os.getenv("LLM_BASE_URL"),
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "dataset_size": len(rows),
        "corpus_passages": len(passages),
        "metrics": scores,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "p95": round(p95, 1),
        },
        "pipeline_consistency_mean": round(statistics.mean(consistency), 4),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved -> {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation over the RAG pipeline.")
    parser.add_argument("--corpus", default="evaluation/datasets/ragas_corpus.json")
    parser.add_argument("--dataset", default="evaluation/datasets/ragas_eval.jsonl")
    parser.add_argument("--output", default="evaluation/reports/ragas-latest.json")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples.")
    args = parser.parse_args()
    run(Path(args.corpus), Path(args.dataset), Path(args.output), args.limit)


if __name__ == "__main__":
    main()
