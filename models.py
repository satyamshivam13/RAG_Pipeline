"""
Pydantic models for every data structure flowing through the pipeline.
Strict typing catches integration bugs at boundaries.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import uuid
import time


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: str = "unknown"
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    source: str = "unknown"
    chunk_index: int = 0
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    similarity_score: float = Field(ge=0.0, le=1.0)


class RelevanceVerdict(str, Enum):
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"


class ChunkRelevanceResult(BaseModel):
    chunk_id: str
    verdict: RelevanceVerdict
    relevance_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class GuardrailOutput(BaseModel):
    query: str
    original_count: int
    filtered_chunks: list[RetrievedChunk]
    removed_chunks: list[ChunkRelevanceResult]
    accepted_chunks: list[ChunkRelevanceResult]
    safety_flags: list[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0


class GeneratorOutput(BaseModel):
    answer: str
    query: str
    context_used: list[RetrievedChunk]
    model: str = ""
    processing_time_ms: float = 0.0
    context_truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    context_token_estimate: int = 0


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"


class ClaimEvaluation(BaseModel):
    claim: str
    verdict: ClaimVerdict
    supporting_evidence: str = ""
    reasoning: str = ""


class EvaluatorOutput(BaseModel):
    overall_consistency_score: float = Field(ge=0.0, le=1.0)
    is_reliable: bool
    claims: list[ClaimEvaluation]
    summary: str = ""
    processing_time_ms: float = 0.0


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineResult(BaseModel):
    query: str
    answer: str
    is_reliable: bool
    consistency_score: float

    retrieval: list[RetrievedChunk]
    guardrail: Optional[GuardrailOutput] = None
    generation: GeneratorOutput
    evaluation: EvaluatorOutput

    evaluation_status: EvaluationStatus = EvaluationStatus.COMPLETED
    evaluation_error: Optional[str] = None
    evaluation_deferred: bool = False

    total_time_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)
