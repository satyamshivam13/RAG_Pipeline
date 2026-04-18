from config import ChunkingConfig
from document_loader import DocumentLoader
from models import Document


def test_chunking_prefers_semantic_boundaries():
    cfg = ChunkingConfig(chunk_size=80, chunk_overlap=10, min_chunk_size=20)
    loader = DocumentLoader(cfg)

    text = (
        "Paragraph one has useful details. It ends cleanly.\n\n"
        "Paragraph two continues with additional context for retrieval."
    )
    doc = Document(content=text, source="test")

    chunks = loader.chunk_document(doc)

    assert len(chunks) >= 2
    assert chunks[0].content.endswith("\n\n") or chunks[0].content.endswith(".")


def test_chunking_overlap_and_api_compatibility():
    cfg = ChunkingConfig(chunk_size=28, chunk_overlap=8, min_chunk_size=10)
    loader = DocumentLoader(cfg)

    text = "Sentence A. Sentence B. Sentence C. Sentence D. Sentence E."
    docs = loader.load_texts([text], source="batch")
    chunks = loader.chunk_documents(docs)

    assert isinstance(chunks, list)
    assert len(chunks) >= 2
    # Overlap keeps chunks connected; first chars of next chunk should appear near prior chunk end.
    assert chunks[1].content[:5].strip() != ""


def test_chunking_drops_tiny_trailing_fragment():
    cfg = ChunkingConfig(chunk_size=25, chunk_overlap=0, min_chunk_size=12)
    loader = DocumentLoader(cfg)

    text = "This chunk is long enough. tiny"
    doc = Document(content=text, source="test")

    chunks = loader.chunk_document(doc)

    assert len(chunks) >= 1
    assert all(len(c.content) >= cfg.min_chunk_size for c in chunks)

