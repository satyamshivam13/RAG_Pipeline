"""
Central configuration for the entire RAG pipeline.
Every tunable knob lives here - no magic numbers scattered across files.
"""

from dataclasses import dataclass, field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384  # Must match model output dim
    device: str = "cpu"  # "cpu" | "cuda"
    batch_size: int = 64
    normalize: bool = True


@dataclass(frozen=True)
class VectorStoreConfig:
    index_type: str = "flat"  # "flat" | "ivf" | "hnsw"
    n_lists: int = 100  # Only for IVF
    n_probe: int = 10  # Only for IVF
    persist_dir: str = "./vector_store_data"


@dataclass(frozen=True)
class RetrieverConfig:
    top_k: int = 10  # Initial retrieval count
    similarity_threshold: float = 0.3  # Min cosine similarity
    use_mmr: bool = True  # Maximal Marginal Relevance
    mmr_lambda: float = 0.7  # Diversity vs relevance trade-off
    mmr_top_k: int = 5  # Final count after MMR


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 512  # Characters per chunk
    chunk_overlap: int = 64  # Overlapping characters
    min_chunk_size: int = 50  # Discard tiny tail chunks


@dataclass(frozen=True)
class LLMConfig:
    """Shared LLM configuration. Each agent can override model/temperature."""

    provider: str = "openai"  # "openai" | "local"
    api_key: Optional[str] = field(default=None)
    base_url: Optional[str] = None  # For local / vLLM / Ollama
    default_model: str = "gpt-4o-mini"
    max_retries: int = 3
    timeout: int = 60

    def __post_init__(self):
        # Frozen dataclass workaround for setting api_key from env
        if self.api_key is None:
            object.__setattr__(self, "api_key", os.getenv("OPENAI_API_KEY"))
        if self.base_url is None:
            object.__setattr__(self, "base_url", os.getenv("LLM_BASE_URL"))
        if os.getenv("LLM_MODEL"):
            object.__setattr__(self, "default_model", os.getenv("LLM_MODEL"))


@dataclass(frozen=True)
class GuardrailConfig:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0  # Deterministic for safety
    relevance_threshold: float = 0.6  # 0-1 score; below = irrelevant
    max_tokens: int = 1024


@dataclass(frozen=True)
class GeneratorConfig:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.3
    max_tokens: int = 2048
    system_prompt: str = (
        "You are a precise, helpful assistant. Answer the user's question "
        "using ONLY the provided context. If the context doesn't contain "
        "enough information, say so explicitly. Never fabricate facts."
    )


@dataclass(frozen=True)
class EvaluatorConfig:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0
    max_tokens: int = 1024
    consistency_threshold: float = 0.7  # Below = flag as unreliable


@dataclass(frozen=True)
class RuntimeConfig:
    # Keep guardrail available for compatibility, but disabled by default in runtime path.
    use_guardrail: bool = False
    # Phase 1 default: deferred evaluation. Set "sync" for deterministic debugging/tests.
    evaluator_mode: str = "deferred"  # "deferred" | "sync"


@dataclass(frozen=True)
class PipelineConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
