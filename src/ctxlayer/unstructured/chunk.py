"""Chunks: a head chunk (title + summary) and overlapping body windows per document."""
from dataclasses import dataclass

from ctxlayer.unstructured._names import BODY_KIND, HEAD_KIND
from ctxlayer.unstructured.documents import Document

_WORDS_PER_TOKEN = 0.75


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    date: str
    text: str
    kind: str  # head | body


def chunk_documents(docs: list[Document], tokens: int, overlap: int) -> list[Chunk]:
    """Per document: chunk `<doc_id>#0` = "title\\nsummary", then body windows `#1`, `#2`, ...

    A window holds int(tokens * 0.75) words and starts int((tokens - overlap) * 0.75) words
    after the previous one, e.g. tokens=500, overlap=60 -> 375-word windows sharing 45 words.
    """
    size = int(tokens * _WORDS_PER_TOKEN)
    shared = int(overlap * _WORDS_PER_TOKEN)
    if shared >= size:
        raise ValueError(f"chunk overlap ({overlap}) must be smaller than chunk size ({tokens})")

    chunks = []
    for doc in docs:
        chunks.append(Chunk(
            f"{doc.doc_id}#0", doc.doc_id, doc.date, f"{' '.join(doc.title.split())}\n{doc.summary}", HEAD_KIND,
        ))

        words = doc.body.split()
        for number, start in enumerate(range(0, len(words), size - shared), start=1):
            window = words[start : start + size]
            chunks.append(Chunk(f"{doc.doc_id}#{number}", doc.doc_id, doc.date, " ".join(window), BODY_KIND))
            if start + size >= len(words):
                break
    return chunks
