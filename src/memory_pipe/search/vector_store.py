"""Vector store for semantic search (optional dependency)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """A single vector entry for semantic search."""

    id: str
    content: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """Simple in-memory vector store for semantic search.

    Uses TF-IDF-like cosine similarity when no embedding model is available.
    For production use, integrate with an embedding API or local model.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self._entries: dict[str, VectorEntry] = {}
        self._idf: dict[str, float] = {}
        self._doc_freq: dict[str, int] = {}
        self._doc_count: int = 0

    def add(self, entry: VectorEntry) -> None:
        """Add a vector entry."""
        self._entries[entry.id] = entry
        self._update_idf(entry.content)

    def _update_idf(self, content: str) -> None:
        """Update IDF values for terms in content."""
        self._doc_count += 1
        terms = self._tokenize(content)
        for term in terms:
            self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

        for term in terms:
            df = self._doc_freq.get(term, 1)
            # Add 1 smoothing to avoid log(1) = 0 with single document
            self._idf[term] = math.log((self._doc_count + 1) / (df + 1)) + 1

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer."""
        import re
        return re.findall(r'\b[a-z]+\b', text.lower())

    def _tfidf_vector(self, text: str) -> list[float]:
        """Convert text to TF-IDF vector."""
        terms = self._tokenize(text)
        if not terms:
            return [0.0] * self.dimension

        term_counts: dict[str, int] = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        # Get all terms from vocabulary
        all_terms = sorted(set(self._idf.keys()))
        vocab_size = len(all_terms)

        if vocab_size == 0:
            return [0.0] * min(self.dimension, 10)

        # Create vector
        vector = [0.0] * self.dimension
        for i, term in enumerate(all_terms):
            if i >= self.dimension:
                break
            tf = term_counts.get(term, 0) / len(terms)
            idf = self._idf.get(term, 0)
            vector[i] = tf * idf

        # Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Search for similar entries by cosine similarity.

        Returns:
            List of (entry_id, similarity_score) tuples, sorted by score descending.
        """
        query_vector = self._tfidf_vector(query)
        if not any(v != 0 for v in query_vector):
            return []

        scores: list[tuple[str, float]] = []

        for entry_id, entry in self._entries.items():
            # Compute cosine similarity
            entry_vec = entry.vector[:len(query_vector)]
            dot = sum(q * e for q, e in zip(query_vector, entry_vec, strict=False))
            q_norm = math.sqrt(sum(v * v for v in query_vector))
            e_norm = math.sqrt(sum(v * v for v in entry.vector))

            if q_norm > 0 and e_norm > 0:
                similarity = dot / (q_norm * e_norm)
                scores.append((entry_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    @property
    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        self._idf.clear()
        self._doc_freq.clear()
        self._doc_count = 0
