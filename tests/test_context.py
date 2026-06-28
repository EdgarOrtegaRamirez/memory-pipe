"""Tests for MemoryPipe context builder."""

from memory_pipe.engine.context import ContextBuilder
from memory_pipe.search.vector_store import VectorStore
from memory_pipe.storage.database import MemoryDatabase
from memory_pipe.storage.models import (
    ContextEntry,
    ConversationTurn,
    ImportanceLevel,
    MemoryItem,
    MemoryScore,
    MemoryType,
)


class TestContextBuilder:
    """Tests for the ContextBuilder."""

    def setup_method(self):
        self.db = MemoryDatabase(":memory:")
        self.vector_store = VectorStore(dimension=32)
        self.builder = ContextBuilder(
            db=self.db,
            vector_store=self.vector_store,
            max_context_length=1000,
        )

    def teardown_method(self):
        self.db.close()

    def test_build_empty_context(self):
        context = self.builder.build_context()
        assert isinstance(context, str)

    def test_build_context_with_facts(self):
        self.db.add_memory(MemoryItem(
            content="User likes Python",
            memory_type=MemoryType.PREFERENCE,
            score=MemoryScore(importance=ImportanceLevel.HIGH),
        ))
        context = self.builder.build_context()
        assert "Python" in context

    def test_build_context_with_query(self):
        self.db.add_memory(MemoryItem(
            content="User is working on a Python project",
            memory_type=MemoryType.CONTEXT,
        ))
        context = self.builder.build_context(user_query="Python")
        assert "Python" in context

    def test_build_context_truncation(self):
        self.db.add_memory(MemoryItem(
            content="x" * 1000,
            memory_type=MemoryType.FACT,
        ))
        self.builder.max_context_length = 100
        context = self.builder.build_context()
        assert len(context) <= 100 + 50  # Allow for truncation marker

    def test_build_context_with_conversation(self):
        self.db.add_conversation_turn("session1", ConversationTurn(role="user", content="Hello"))
        self.db.add_conversation_turn("session1", ConversationTurn(role="assistant", content="Hi!"))
        context = self.builder.build_context(session_id="session1")
        assert "Hello" in context or "Hi" in context

    def test_context_with_high_importance_priority(self):
        self.db.add_memory(MemoryItem(
            content="low importance",
            memory_type=MemoryType.FACT,
            score=MemoryScore(importance=ImportanceLevel.LOW),
        ))
        self.db.add_memory(MemoryItem(
            content="high importance",
            memory_type=MemoryType.FACT,
            score=MemoryScore(importance=ImportanceLevel.HIGH),
        ))
        context = self.builder.build_context()
        # High importance should appear
        assert "high importance" in context

    def test_extract_and_store(self):
        result = self.builder.extract_and_store(
            "My name is Test. I love Python.",
            tags=["test-tag"],
        )
        assert result.total_facts >= 1

        memories = self.db.list_memories()
        assert len(memories) >= 1
        assert "test-tag" in memories[0].tags


class TestContextBuilderNoVectorStore:
    """Tests for ContextBuilder without vector store."""

    def setup_method(self):
        self.db = MemoryDatabase(":memory:")
        self.builder = ContextBuilder(db=self.db)

    def teardown_method(self):
        self.db.close()

    def test_build_context_without_vector_store(self):
        self.db.add_memory(MemoryItem(
            content="test fact",
            memory_type=MemoryType.FACT,
            score=MemoryScore(importance=ImportanceLevel.HIGH),
        ))
        context = self.builder.build_context()
        assert isinstance(context, str)


class TestExtractAndStoreIntegration:
    """Integration tests for extraction + storage."""

    def setup_method(self):
        self.db = MemoryDatabase(":memory:")
        self.builder = ContextBuilder(db=self.db)

    def teardown_method(self):
        self.db.close()

    def test_extract_user_turns_only(self):
        turns = [
            ConversationTurn(role="user", content="My name is Alice."),
            ConversationTurn(role="assistant", content="Hello Alice!"),
        ]
        result = self.builder.add_conversation("session1", turns)
        assert result.total_facts >= 1

    def test_extract_and_store_preserves_tags(self):
        result = self.builder.extract_and_store(
            "I prefer dark mode.",
            tags=["settings"],
        )
        assert result.total_facts >= 1
        memories = self.db.list_memories(tags=["settings"])
        assert len(memories) >= 1


class TestContextEntryFormatting:
    """Tests for context entry string formatting."""

    def test_context_entry_to_string(self):
        entry = ContextEntry(category="facts", content="test content", relevance_score=0.8)
        assert entry.to_string() == "[facts] test content"

    def test_context_entry_relevance_bounds(self):
        entry = ContextEntry(category="facts", content="test", relevance_score=0.0)
        assert entry.relevance_score == 0.0

        entry = ContextEntry(category="facts", content="test", relevance_score=1.0)
        assert entry.relevance_score == 1.0
