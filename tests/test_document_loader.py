from chunking import TiktokenTokenizer
from config import ChunkingConfig
from document_loader import DocumentLoader
from models import Document


def _cfg(**overrides):
    return ChunkingConfig(tokenizer_encoding="regex", **overrides)


def test_semantic_chunking_prefers_paragraph_and_sentence_boundaries():
    cfg = _cfg(chunk_size=18, chunk_overlap=3, min_chunk_size=4)
    loader = DocumentLoader(cfg)

    text = (
        "Paragraph one has useful details. It ends cleanly.\n\n"
        "Paragraph two continues with additional context for retrieval."
    )
    doc = Document(content=text, source="test")

    chunks = loader.chunk_document(doc)

    assert len(chunks) >= 2
    assert chunks[0].content.endswith(".")
    assert chunks[0].metadata["chunking"]["strategy"] == "semantic"
    assert chunks[0].metadata["chunking"]["metrics"]["ends_at_semantic_boundary"] is True


def test_token_budget_and_overlap_are_enforced():
    cfg = _cfg(chunk_size=16, chunk_overlap=4, min_chunk_size=4)
    loader = DocumentLoader(cfg)
    tokenizer = TiktokenTokenizer(cfg.tokenizer_model, cfg.tokenizer_encoding)

    text = " ".join(f"retrieval sentence {i}." for i in range(12))
    docs = loader.load_texts([text], source="batch")
    chunks = loader.chunk_documents(docs)

    assert isinstance(chunks, list)
    assert len(chunks) >= 2
    assert all(tokenizer.count(chunk.content) <= cfg.chunk_size for chunk in chunks)
    assert chunks[1].metadata["chunking"]["metrics"]["overlap_tokens"] <= cfg.chunk_overlap


def test_chunking_merges_or_drops_tiny_trailing_fragment():
    cfg = _cfg(chunk_size=12, chunk_overlap=0, min_chunk_size=6)
    loader = DocumentLoader(cfg)

    text = "This chunk is long enough for retrieval. tiny"
    doc = Document(content=text, source="test")

    chunks = loader.chunk_document(doc)

    assert len(chunks) >= 1
    assert chunks[-1].content != "tiny"


def test_oversized_document_is_split_without_exceeding_token_budget():
    cfg = _cfg(chunk_size=24, chunk_overlap=5, min_chunk_size=5)
    loader = DocumentLoader(cfg)
    tokenizer = TiktokenTokenizer(cfg.tokenizer_model, cfg.tokenizer_encoding)

    text = " ".join(
        [
            "A long policy paragraph explains retention, access controls, audit logs, and recovery objectives."
            for _ in range(40)
        ]
    )
    doc = Document(content=text, source="oversized")

    chunks = loader.chunk_document(doc)

    assert len(chunks) > 10
    assert all(tokenizer.count(chunk.content) <= cfg.chunk_size for chunk in chunks)
    assert all(chunk.metadata["chunking"]["metrics"]["oversized"] is False for chunk in chunks)


def test_multilingual_content_uses_sentence_boundaries():
    cfg = _cfg(chunk_size=20, chunk_overlap=3, min_chunk_size=4)
    loader = DocumentLoader(cfg)

    text = (
        "English retrieval works across scripts. "
        "हिंदी वाक्य भी सही तरह से अलग होता है। "
        "中文句子应该保持完整。 "
        "¿La busqueda semantica funciona? Si."
    )
    doc = Document(content=text, source="multilingual")

    chunks = loader.chunk_document(doc)

    assert len(chunks) >= 2
    assert any("हिंदी" in chunk.content for chunk in chunks)
    assert any("中文" in chunk.content for chunk in chunks)
    assert all(chunk.metadata["chunking"]["metrics"]["sentence_count"] >= 1 for chunk in chunks)


def test_edge_cases_do_not_emit_empty_or_budget_busting_chunks():
    cfg = _cfg(chunk_size=8, chunk_overlap=2, min_chunk_size=3)
    loader = DocumentLoader(cfg)
    tokenizer = TiktokenTokenizer(cfg.tokenizer_model, cfg.tokenizer_encoding)

    assert loader.chunk_document(Document(content="   \n\n  ", source="empty")) == []

    no_spaces = "supercalifragilisticexpialidocious" * 10
    chunks = loader.chunk_document(Document(content=no_spaces, source="edge"))

    assert chunks
    assert all(chunk.content.strip() for chunk in chunks)
    assert all(tokenizer.count(chunk.content) <= cfg.chunk_size for chunk in chunks)


def test_token_aware_strategy_fits_context_better_than_legacy_character_chunks():
    text = " ".join(["internationalization", "retrieval", "quality", "semantic"] * 80)
    token_cfg = _cfg(chunk_size=40, chunk_overlap=5, min_chunk_size=5)
    legacy_cfg = _cfg(
        chunk_size=40,
        chunk_overlap=5,
        min_chunk_size=5,
        strategy="legacy_char",
    )
    tokenizer = TiktokenTokenizer(token_cfg.tokenizer_model, token_cfg.tokenizer_encoding)

    token_chunks = DocumentLoader(token_cfg).chunk_document(Document(content=text, source="token"))
    legacy_chunks = DocumentLoader(legacy_cfg).chunk_document(Document(content=text, source="legacy"))

    token_max = max(tokenizer.count(chunk.content) for chunk in token_chunks)
    legacy_max = max(tokenizer.count(chunk.content) for chunk in legacy_chunks)

    assert token_max <= token_cfg.chunk_size
    assert legacy_max != token_cfg.chunk_size
    assert len(token_chunks) < len(legacy_chunks)
