from config import RetrieverConfig
from models import Chunk, RetrievedChunk
from retriever import Retriever


class _StubEmbeddingModel:
    def embed_query(self, query):
        return [1.0, 0.0, 0.0, 0.0]


class _StubVectorStore:
    def __init__(self, mmr_results, search_results):
        self._mmr_results = mmr_results
        self._search_results = search_results

    def mmr_search(self, **_kwargs):
        return list(self._mmr_results)

    def search(self, **_kwargs):
        return list(self._search_results)


def _chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc-1", content=content, source="src", chunk_index=0)


def test_retriever_preserves_mmr_order():
    c1 = RetrievedChunk(chunk=_chunk("c1", "first"), similarity_score=0.2)
    c2 = RetrievedChunk(chunk=_chunk("c2", "second"), similarity_score=0.9)
    store = _StubVectorStore([c1, c2], [c2, c1])
    retriever = Retriever(RetrieverConfig(use_mmr=True), _StubEmbeddingModel(), store)

    results = retriever.retrieve("query")

    assert [r.chunk.id for r in results] == ["c1", "c2"]


def test_retriever_sorts_similarity_search_results():
    c1 = RetrievedChunk(chunk=_chunk("c1", "first"), similarity_score=0.2)
    c2 = RetrievedChunk(chunk=_chunk("c2", "second"), similarity_score=0.9)
    store = _StubVectorStore([c1, c2], [c1, c2])
    retriever = Retriever(RetrieverConfig(use_mmr=False), _StubEmbeddingModel(), store)

    results = retriever.retrieve("query")

    assert [r.chunk.id for r in results] == ["c2", "c1"]
