"""Tests for MemoryPipe models."""

import time

from memory_pipe.storage.models import (
    ContextEntry,
    ConversationTurn,
    ImportanceLevel,
    MemoryItem,
    MemoryScore,
    MemoryType,
)


class TestMemoryScore:
    """Tests for MemoryScore model."""

    def test_default_importance(self):
        score = MemoryScore()
        assert score.importance == ImportanceLevel.MEDIUM

    def test_default_confidence(self):
        score = MemoryScore()
        assert score.confidence == 0.5

    def test_effective_score_basic(self):
        score = MemoryScore(importance=ImportanceLevel.HIGH)
        assert score.effective_score == 0.75

    def test_effective_score_critical(self):
        score = MemoryScore(importance=ImportanceLevel.CRITICAL)
        assert score.effective_score == 1.0

    def test_effective_score_low(self):
        score = MemoryScore(importance=ImportanceLevel.LOW)
        assert score.effective_score == 0.25

    def test_access_bonus(self):
        score = MemoryScore(importance=ImportanceLevel.MEDIUM, access_count=10)
        assert score.effective_score == 0.7  # 0.5 + 0.2 (capped)

    def test_is_expired_with_ttl(self):
        score = MemoryScore(
            ttl_hours=1.0,
            last_accessed=time.time() - 7200,  # 2 hours ago
        )
        assert score.is_expired is True

    def test_not_expired_within_ttl(self):
        score = MemoryScore(
            ttl_hours=24.0,
            last_accessed=time.time() - 3600,  # 1 hour ago
        )
        assert score.is_expired is False

    def test_never_expires_without_ttl(self):
        score = MemoryScore(ttl_hours=None)
        assert score.is_expired is False

    def test_invalid_confidence_low(self):
        try:
            MemoryScore(confidence=-0.1)
            raise AssertionError("Should have raised")
        except Exception:
            pass

    def test_invalid_confidence_high(self):
        try:
            MemoryScore(confidence=1.1)
            raise AssertionError("Should have raised")
        except Exception:
            pass


class TestMemoryItem:
    """Tests for MemoryItem model."""

    def test_defaults(self):
        item = MemoryItem(content="test fact")
        assert item.content == "test fact"
        assert item.memory_type == MemoryType.FACT
        assert len(item.tags) == 0
        assert len(item.id) == 12

    def test_custom_type(self):
        item = MemoryItem(content="preference", memory_type=MemoryType.PREFERENCE)
        assert item.memory_type == MemoryType.PREFERENCE

    def test_update_access(self):
        item = MemoryItem(content="test")
        assert item.score.access_count == 0
        assert item.score.last_accessed is None

        item.update_access()
        assert item.score.access_count == 1
        assert item.score.last_accessed is not None

    def test_to_context_string(self):
        item = MemoryItem(
            content="test fact",
            memory_type=MemoryType.FACT,
            tags=["test", "example"],
            score=MemoryScore(importance=ImportanceLevel.HIGH),
        )
        ctx = item.to_context_string()
        assert "test fact" in ctx
        assert "test" in ctx
        assert "high" in ctx

    def test_to_context_string_truncation(self):
        long_content = "x" * 1000
        item = MemoryItem(content=long_content)
        ctx = item.to_context_string(max_length=100)
        assert len(ctx) <= 100

    def test_to_context_string_short(self):
        item = MemoryItem(content="short")
        ctx = item.to_context_string()
        assert len(ctx) <= 500


class TestConversationTurn:
    """Tests for ConversationTurn model."""

    def test_defaults(self):
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"
        assert turn.timestamp is not None

    def test_with_metadata(self):
        turn = ConversationTurn(
            role="assistant",
            content="hi there",
            metadata={"model": "test"},
        )
        assert turn.metadata == {"model": "test"}


class TestContextEntry:
    """Tests for ContextEntry model."""

    def test_defaults(self):
        entry = ContextEntry(category="facts", content="test")
        assert entry.category == "facts"
        assert entry.content == "test"
        assert entry.relevance_score == 1.0

    def test_to_string(self):
        entry = ContextEntry(category="facts", content="important fact")
        assert entry.to_string() == "[facts] important fact"

    def test_custom_relevance(self):
        entry = ContextEntry(category="facts", content="test", relevance_score=0.5)
        assert entry.relevance_score == 0.5
