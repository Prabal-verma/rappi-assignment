"""Retrieval over the buying team's written policy.

Business rules that are *enforced* live in constraints.py. What lives here
is the softer knowledge a good buyer carries: how to read a demand spike,
when a transfer beats a purchase, what to do with a supplier who keeps
short-shipping. The agent retrieves it at decision time and cites it.

The retriever is lexical (BM25 over heading-delimited chunks). That is a
considered choice, not a shortcut: it needs no embedding service, so the
whole system runs offline and evaluations are perfectly reproducible, and
policy language is keyword-dense enough that lexical matching does well on
it. `Retriever` is an interface — swapping in pgvector means implementing
`search` and nothing else. See docs/architecture.md for the trade-off.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "is", "are", "be", "for",
    "on", "with", "as", "by", "at", "it", "this", "that", "from", "we", "you",
    "should", "must", "may", "can", "if", "when", "than", "then", "not", "do",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_]*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str
    tokens: list[str]

    @property
    def citation(self) -> str:
        return f"{self.doc_id}#{self.section}"


@dataclass
class SearchHit:
    citation: str
    title: str
    section: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return {
            "citation": self.citation,
            "document": self.title,
            "section": self.section,
            "text": self.text,
            "relevance": round(self.score, 3),
        }


class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> list[SearchHit]: ...


class BM25Retriever:
    """Okapi BM25 over markdown chunks split on level-2 headings."""

    def __init__(self, directory: Path | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.directory = directory or KNOWLEDGE_DIR
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._df: Counter[str] = Counter()
        self._avg_len: float = 1.0
        self._load()

    def _load(self) -> None:
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            title = raw.splitlines()[0].lstrip("# ").strip() if raw.strip() else path.stem
            # Split on '## ' headings; keep the preamble as an Overview chunk.
            parts = re.split(r"^## +", raw, flags=re.MULTILINE)
            preamble, sections = parts[0], parts[1:]
            if preamble.strip():
                self._add(path.stem, title, "overview", preamble)
            for section in sections:
                lines = section.splitlines()
                heading = lines[0].strip() if lines else "section"
                body = "\n".join(lines[1:]).strip()
                if body:
                    self._add(path.stem, title, heading, f"{heading}\n{body}")

        if self.chunks:
            self._avg_len = sum(len(c.tokens) for c in self.chunks) / len(self.chunks)
            for chunk in self.chunks:
                for term in set(chunk.tokens):
                    self._df[term] += 1

    def _add(self, doc_id: str, title: str, section: str, text: str) -> None:
        clean = text.strip()
        self.chunks.append(
            Chunk(
                doc_id=doc_id,
                title=title,
                section=section.lower().replace(" ", "-"),
                text=clean,
                tokens=tokenize(clean),
            )
        )

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 4) -> list[SearchHit]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []

        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            tf = Counter(chunk.tokens)
            length = len(chunk.tokens) or 1
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            SearchHit(
                citation=chunk.citation,
                title=chunk.title,
                section=chunk.section,
                text=chunk.text,
                score=score,
            )
            for score, chunk in scored[:k]
        ]


_retriever: BM25Retriever | None = None


def get_retriever() -> BM25Retriever:
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever()
    return _retriever
