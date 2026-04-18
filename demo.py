"""
End-to-end demo of the Multi-Agent RAG Pipeline.
Run: python demo.py
"""

import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

from config import PipelineConfig
from main import RAGPipeline

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-24s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)

console = Console()

# ── Sample Knowledge Base ────────────────────────────────────────────
KNOWLEDGE_BASE = [
    """
    Quantum Computing Fundamentals:
    Quantum computers use qubits instead of classical bits. Unlike classical bits
    that exist as 0 or 1, qubits can exist in superposition — simultaneously
    representing both states. This property, combined with quantum entanglement,
    allows quantum computers to process certain calculations exponentially faster
    than classical computers. IBM's Eagle processor, released in 2021, features
    127 qubits. Google's Sycamore processor demonstrated quantum supremacy in 2019
    by completing a calculation in 200 seconds that would take a classical
    supercomputer approximately 10,000 years. Current quantum computers require
    extreme cooling to near absolute zero (-273.15°C) to maintain qubit coherence.
    Quantum error correction remains one of the biggest challenges, as qubits are
    extremely sensitive to environmental noise.
    """,

    """
    Machine Learning Model Training:
    Training large language models requires massive computational resources.
    GPT-3, released by OpenAI in 2020, was trained on approximately 570GB of
    text data and has 175 billion parameters. The training process consumed an
    estimated 3.14 × 10^23 FLOPS of compute. Modern training techniques include
    distributed training across thousands of GPUs, mixed-precision training to
    reduce memory usage, and gradient checkpointing. Transfer learning allows
    pre-trained models to be fine-tuned on specific tasks with much less data.
    The cost of training GPT-3 was estimated at $4.6 million in cloud compute
    costs. Techniques like LoRA (Low-Rank Adaptation) have since reduced
    fine-tuning costs by orders of magnitude.
    """,

    """
    Renewable Energy Technologies:
    Solar photovoltaic (PV) technology has seen dramatic cost reductions, with
    the levelized cost of solar electricity dropping by 89% between 2010 and 2022.
    Modern solar panels achieve efficiency rates of 20-23% for commercial panels,
    while laboratory cells have reached over 47% efficiency using multi-junction
    designs. Wind energy capacity globally reached 837 GW by the end of 2022.
    Offshore wind farms can generate electricity at capacity factors of 40-50%,
    significantly higher than onshore installations. Battery storage technology,
    particularly lithium-ion, has seen costs decline by 97% since 1991. Emerging
    technologies include perovskite solar cells, solid-state batteries, and green
    hydrogen production through electrolysis powered by renewable sources.
    """,

    """
    CRISPR Gene Editing:
    CRISPR-Cas9, discovered by Jennifer Doudna and Emmanuelle Charpentier (who
    received the 2020 Nobel Prize in Chemistry), is a revolutionary gene-editing
    technology. It works by using a guide RNA to direct the Cas9 enzyme to a
    specific location in the genome, where it makes a precise cut. This allows
    scientists to delete, modify, or insert genetic material. Clinical trials are
    underway for treating sickle cell disease, beta-thalassemia, and certain
    cancers. In 2023, the FDA approved Casgevy, the first CRISPR-based therapy,
    for treating sickle cell disease. Ethical concerns include the potential for
    germline editing (heritable changes) and off-target effects where unintended
    parts of the genome are modified.
    """,

    """
    Climate Change and Ocean Acidification:
    The world's oceans have absorbed approximately 30% of anthropogenic CO2
    emissions since the industrial revolution, leading to ocean acidification.
    The average pH of ocean surface water has decreased from approximately 8.21
    to 8.10 since the pre-industrial era — a 26% increase in acidity. This
    threatens marine organisms that build calcium carbonate shells or skeletons,
    including corals, mollusks, and some plankton species. Coral bleaching events
    have increased in frequency, with the Great Barrier Reef experiencing mass
    bleaching in 2016, 2017, 2020, and 2022. Ocean temperatures have risen by
    approximately 0.88°C compared to the 1850-1900 average. Deep ocean warming
    is accelerating, which has implications for sea level rise through thermal
    expansion.
    """,
]

DEMO_QUERIES = [
    "How does quantum computing achieve speedups over classical computing?",
    "What are the costs associated with training large language models?",
    "What is the current state of CRISPR gene editing therapy approvals?",
    "How has solar energy cost changed over the past decade?",
    "What is the relationship between CO2 emissions and ocean pH levels?",
    # This one should trigger low-relevance guardrail for most chunks:
    "What is the population of Tokyo?",
]


def display_result(result, console: Console) -> None:
    """Pretty-print a PipelineResult using Rich."""

    # ── Header ──────────────────────────────────────────────────
    console.print()
    console.rule(f"[bold cyan]Query: {result.query}", style="cyan")

    # ── Answer ──────────────────────────────────────────────────
    reliability_color = "green" if result.is_reliable else "red"
    reliability_icon = "✅" if result.is_reliable else "⚠️"

    console.print(Panel(
        Markdown(result.answer),
        title=f"{reliability_icon} Answer (consistency: {result.consistency_score:.0%})",
        border_style=reliability_color,
        padding=(1, 2),
    ))

    # ── Retrieval Summary ───────────────────────────────────────
    table = Table(
        title="📦 Retrieval & Guardrail Summary",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Chunks Retrieved", str(len(result.retrieval)))
    table.add_row("Chunks After Guardrail", str(len(result.guardrail.filtered_chunks)))
    table.add_row("Chunks Removed", str(len(result.guardrail.removed_chunks)))
    table.add_row(
        "Safety Flags",
        ", ".join(result.guardrail.safety_flags) if result.guardrail.safety_flags else "None"
    )
    table.add_row("Guardrail Time", f"{result.guardrail.processing_time_ms:.0f}ms")
    table.add_row("Generation Time", f"{result.generation.processing_time_ms:.0f}ms")
    table.add_row("Evaluation Time", f"{result.evaluation.processing_time_ms:.0f}ms")
    table.add_row("Total Pipeline Time", f"{result.total_time_ms:.0f}ms")

    console.print(table)

    # ── Claim-level Evaluation ──────────────────────────────────
    if result.evaluation.claims:
        claim_table = Table(
            title="🔍 Factual Consistency — Claim Breakdown",
            box=box.ROUNDED,
            show_lines=True,
        )
        claim_table.add_column("#", width=3)
        claim_table.add_column("Claim", max_width=50)
        claim_table.add_column("Verdict", width=20)
        claim_table.add_column("Reasoning", max_width=40)

        verdict_colors = {
            "supported": "green",
            "partially_supported": "yellow",
            "not_supported": "red",
            "contradicted": "bold red",
        }

        for i, claim in enumerate(result.evaluation.claims, 1):
            color = verdict_colors.get(claim.verdict.value, "white")
            claim_table.add_row(
                str(i),
                claim.claim[:80] + ("..." if len(claim.claim) > 80 else ""),
                f"[{color}]{claim.verdict.value}[/{color}]",
                claim.reasoning[:60] + ("..." if len(claim.reasoning) > 60 else ""),
            )

        console.print(claim_table)

    # ── Evaluator Summary ───────────────────────────────────────
    if result.evaluation.summary:
        console.print(Panel(
            result.evaluation.summary,
            title="📋 Evaluator Summary",
            border_style="blue",
        ))

    # ── Guardrail Removed Chunks ────────────────────────────────
    if result.guardrail.removed_chunks:
        removed_table = Table(
            title="🚫 Chunks Removed by Guardrail",
            box=box.SIMPLE,
        )
        removed_table.add_column("Chunk ID", width=12)
        removed_table.add_column("Score", width=8)
        removed_table.add_column("Reason")

        for rc in result.guardrail.removed_chunks:
            removed_table.add_row(
                rc.chunk_id[:12],
                f"{rc.relevance_score:.2f}",
                rc.reasoning[:80],
            )

        console.print(removed_table)


def main():
    console.print(Panel(
        "[bold]Multi-Agent RAG Pipeline Demo[/bold]\n\n"
        "Components:\n"
        "  1. 📦 Retriever  — FAISS vector search with MMR\n"
        "  2. 🛡️  Guardrail  — LLM-based relevance & safety filter\n"
        "  3. 🤖 Generator  — Grounded answer generation\n"
        "  4. ✅ Evaluator  — Factual consistency scoring",
        title="🚀 RAG Pipeline",
        border_style="bright_blue",
        padding=(1, 2),
    ))

    # ── Initialize ──────────────────────────────────────────────
    console.print("\n[bold yellow]Initializing pipeline...[/bold yellow]")
    config = PipelineConfig()
    pipeline = RAGPipeline(config)

    try:
        # ── Ingest ──────────────────────────────────────────────────
        console.print("[bold yellow]Ingesting knowledge base...[/bold yellow]")
        n_chunks = pipeline.ingest(KNOWLEDGE_BASE, source="demo_kb")
        console.print(f"[green]✓ Ingested {len(KNOWLEDGE_BASE)} documents → {n_chunks} chunks[/green]\n")

        # ── Query Loop ──────────────────────────────────────────────
        for query in DEMO_QUERIES:
            result = pipeline.query(query)
            display_result(result, console)
            console.print()

        # ── Interactive Mode ────────────────────────────────────────
        console.print(Panel(
            "Type a question and press Enter. Type 'quit' to exit.",
            title="💬 Interactive Mode",
            border_style="green",
        ))

        while True:
            try:
                question = console.input("[bold green]Question>[/bold green] ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    break
                if not question:
                    continue

                result = pipeline.query(question)
                display_result(result, console)

            except KeyboardInterrupt:
                break

        console.print("\n[bold]Goodbye! 👋[/bold]")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()