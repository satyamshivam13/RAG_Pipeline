"""RAGAS evaluation runner for RAG_Pipeline.

Runs the real pipeline over a grounded question set and scores it with RAGAS
(faithfulness, answer relevancy, context precision, context recall).

The judge LLM is read from the same env the pipeline uses (OPENAI_API_KEY /
LLM_BASE_URL / LLM_MODEL), so it works with OpenAI-compatible providers such as
Groq. Embeddings for the relevancy metric use the local sentence-transformers
model so no OpenAI embeddings endpoint is required.

The run has two stages that each spend judge-LLM tokens:

  1. Pipeline stage  - answers every question through the real pipeline.
  2. RAGAS scoring   - many judge-LLM calls to score those answers.

On free tiers with a daily token cap (e.g. Groq's 100k TPD) the two stages
together can exceed the budget. To avoid re-spending the pipeline budget on a
retry, stage 1 caches its results to disk *before* scoring begins. A retry with
``--resume`` skips the pipeline entirely and scores from the cache, so only the
RAGAS calls consume quota.

Usage:
    # Full run (pipeline + scoring):
    python -m evaluation.run_ragas \
        --corpus evaluation/datasets/ragas_corpus.json \
        --dataset evaluation/datasets/ragas_eval.jsonl \
        --output evaluation/reports/ragas-latest.json

    # Stage 1 only - answer questions and cache, spend no scoring tokens:
    python -m evaluation.run_ragas --pipeline-only

    # Stage 2 only - score the cached answers (after a quota reset):
    python -m evaluation.run_ragas --resume
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
    from pydantic import SecretStr

    openai_api_key = os.getenv("OPENAI_API_KEY")
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("LLM_BASE_URL") or None,
            api_key=SecretStr(openai_api_key) if openai_api_key else None,
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


def run_pipeline_stage(corpus_path: Path, dataset_path: Path, cache_path: Path, limit: int | None) -> dict:
    """Answer every question through the real pipeline and cache the results.

    Written to ``cache_path`` *before* any scoring happens so a later
    ``--resume`` run can score without re-spending the pipeline token budget.
    """
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

    cached = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "corpus_passages": len(passages),
        "dataset_size": len(rows),
        "questions": questions,
        "answers": answers,
        "contexts": contexts,
        "references": references,
        "latencies_ms": latencies,
        "consistency": consistency,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    print(f"Pipeline results cached -> {cache_path}", flush=True)
    return cached


def _extract_scores(eval_result) -> dict:
    """Pull nan-safe mean scores out of a RAGAS EvaluationResult.

    ``dict(eval_result)`` is wrong: the object has no mapping protocol, so
    ``dict()`` falls back to sequence iteration and raises ``KeyError: 0``.
    The aggregated means live in ``_repr_dict`` (the same values ``__repr__``
    prints). Metrics whose jobs all failed come back as NaN -> reported as None.
    """
    repr_dict = getattr(eval_result, "_repr_dict", None)
    if repr_dict is None:  # pragma: no cover - defensive fallback
        scores_list = getattr(eval_result, "scores", [])
        keys = scores_list[0].keys() if scores_list else []
        repr_dict = {
            k: statistics.fmean(
                [d[k] for d in scores_list if d.get(k) is not None and d.get(k) == d.get(k)] or [float("nan")]
            )
            for k in keys
        }
    scores: dict = {}
    for k, v in repr_dict.items():
        # NaN != NaN; treat NaN / None as "not scored".
        scores[k] = round(float(v), 4) if v is not None and v == v else None
    return scores


def _score_coverage(eval_result) -> dict:
    """Per-metric count of samples that actually scored (non-NaN)."""
    scores_list = getattr(eval_result, "scores", []) or []
    coverage: dict = {}
    for sample in scores_list:
        for k, v in sample.items():
            ok = v is not None and v == v
            coverage[k] = coverage.get(k, 0) + (1 if ok else 0)
    return coverage


def run_scoring_stage(cached: dict, output_path: Path, limit: int | None = None) -> dict:
    """Score cached pipeline answers with RAGAS and write the report.

    Robust to partial failure: whatever metrics completed are written out with
    a coverage breakdown instead of crashing and losing everything.

    ``limit`` scores only the first N cached samples. The full 4-metric eval
    over all 22 questions costs ~260k judge tokens, which overflows small daily
    caps (e.g. Groq free tier's 100k TPD); limiting to ~8 keeps a *complete*
    4-metric run inside one daily window.
    """
    from ragas import evaluate, EvaluationDataset, RunConfig
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    questions = cached["questions"]
    answers = cached["answers"]
    contexts = cached["contexts"]
    references = cached["references"]
    latencies = cached["latencies_ms"]
    consistency = cached["consistency"]

    if not questions:
        print("No cached samples to score; nothing to do.", flush=True)
        return {}

    if limit:
        questions = questions[:limit]
        answers = answers[:limit]
        contexts = contexts[:limit]
        references = references[:limit]
        latencies = latencies[:limit]
        consistency = consistency[:limit]
        print(f"Scoring first {len(questions)} of {cached['dataset_size']} cached samples (--limit).", flush=True)

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

    scores = _extract_scores(eval_result)
    coverage = _score_coverage(eval_result)
    n = len(questions)
    complete = all(scores.get(m) is not None and coverage.get(m, 0) == n for m in scores)

    p95 = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
    report = {
        "evaluator": "ragas",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE" if complete else "PARTIAL - some metric jobs did not finish (see coverage)",
        "judge_model": os.getenv("LLM_MODEL"),
        "judge_provider_base_url": os.getenv("LLM_BASE_URL"),
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "dataset_size": n,
        "corpus_passages": cached.get("corpus_passages"),
        "metrics": scores,
        "metric_coverage": {m: f"{coverage.get(m, 0)}/{n}" for m in scores},
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


def run(
    corpus_path: Path,
    dataset_path: Path,
    output_path: Path,
    limit: int | None,
    cache_path: Path,
    resume: bool = False,
    pipeline_only: bool = False,
) -> dict | None:
    if resume:
        if not cache_path.exists():
            raise SystemExit(
                f"--resume needs a pipeline cache at {cache_path}, but none exists. "
                f"Run the pipeline stage first (e.g. --pipeline-only)."
            )
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Resuming from cached pipeline results ({cached['dataset_size']} questions).", flush=True)
    else:
        cached = run_pipeline_stage(corpus_path, dataset_path, cache_path, limit)

    if pipeline_only:
        print("Pipeline stage done and cached; skipping RAGAS scoring (--pipeline-only).", flush=True)
        return None

    # In the non-resume path the pipeline stage already limited the cache, so
    # re-applying the limit here is a harmless no-op; on --resume it slices the
    # full cache down to the first N samples.
    return run_scoring_stage(cached, output_path, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation over the RAG pipeline.")
    parser.add_argument("--corpus", default="evaluation/datasets/ragas_corpus.json")
    parser.add_argument("--dataset", default="evaluation/datasets/ragas_eval.jsonl")
    parser.add_argument("--output", default="evaluation/reports/ragas-latest.json")
    parser.add_argument(
        "--cache",
        default="evaluation/reports/ragas-pipeline-cache.json",
        help="Where pipeline answers/contexts are cached between the two stages.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip the pipeline stage and score from the cache (spends only RAGAS tokens).",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Run the pipeline stage and cache results, then stop before RAGAS scoring.",
    )
    args = parser.parse_args()
    run(
        Path(args.corpus),
        Path(args.dataset),
        Path(args.output),
        args.limit,
        Path(args.cache),
        resume=args.resume,
        pipeline_only=args.pipeline_only,
    )


if __name__ == "__main__":
    main()
