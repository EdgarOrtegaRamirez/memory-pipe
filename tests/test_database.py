"""Tests for MemoryPipe database."""

import time
from memory_pipe.storage.database import MemoryDatabase
from memory_pipe.storage.models import (
    ConversationTurn,
    ContextEntry,
    ImportanceLevel,
    MemoryItem,
    MemoryScore,
    MemoryType,
)


class TestMemoryDatabase:
    """Tests for MemoryDatabase CRUD operations."""

    def setup_method(self):
        self.db = MemoryDatabase(":memory:")

    def teardown_method(self):
        self.db.close()

    def test_add_and_get_memory(self):
        item = MemoryItem(content="test fact", memory_type=MemoryType.FACT)
        item_id = self.db.add_memory(item)

        retrieved = self.db.get_memory(item_id)
        assert retrieved is not None
        assert retrieved.id == item_id
        assert retrieved.content == "test fact"
        assert retrieved.memory_type == MemoryType.FACT

    def test_get_nonexistent_memory(self):
        assert self.db.get_memory("nonexistent") is None

    def test_update_memory(self):
        item = MemoryItem(content="original", memory_type=MemoryType.NOTE)
        item_id = self.db.add_memory(item)

        updated = self.db.update_memory(item_id, content="updated")
        assert updated is not None
        assert updated.content == "updated"
        assert updated.updated_at >= item.updated_at

    def test_delete_memory(self):
        item = MemoryItem(content="to delete")
        item_id = self.db.add_memory(item)

        assert self.db.delete_memory(item_id) is True
        assert self.db.get_memory(item_id) is None

    def test_delete_nonexistent_memory(self):
        assert self.db.delete_memory("nonexistent") is False

    def test_list_memories(self):
        for i in range(5):
            self.db.add_memory(MemoryItem(content=f"fact {i}"))

        memories = self.db.list_memories(limit=10)
        assert len(memories) == 5

    def test_list_memories_with_type_filter(self):
        self.db.add_memory(MemoryItem(content="fact", memory_type=MemoryType.FACT))
        self.db.add_memory(MemoryItem(content="preference", memory_type=MemoryType.PREFERENCE))
        self.db.add_memory(MemoryItem(content="another fact", memory_type=MemoryType.FACT))

        facts = self.db.list_memories(memory_type=MemoryType.FACT)
        assert len(facts) == 2

        prefs = self.db.list_memories(memory_type=MemoryType.PREFERENCE)
        assert len(prefs) == 1

    def test_list_memories_with_importance_filter(self):
        self.db.add_memory(MemoryItem(content="low", score=MemoryScore(importance=ImportanceLevel.LOW)))
        self.db.add_memory(MemoryItem(content="high", score=MemoryScore(importance=ImportanceLevel.HIGH)))

        low = self.db.list_memories(importance=ImportanceLevel.LOW)
        assert len(low) == 1

        high = self.db.list_memories(importance=ImportanceLevel.HIGH)
        assert len(high) == 1

    def test_list_memories_with_tags(self):
        item1 = MemoryItem(content="tagged fact", tags=["python", "coding"])
        item2 = MemoryItem(content="untagged fact")
        item3 = MemoryItem(content="python fact", tags=["python"])

        self.db.add_memory(item1)
        self.db.add_memory(item2)
        self.db.add_memory(item3)

        tagged = self.db.list_memories(tags=["python"])
        assert len(tagged) == 2

    def test_list_memories_pagination(self):
        for i in range(10):
            self.db.add_memory(MemoryItem(content=f"fact {i}"))

        first_page = self.db.list_memories(limit=5, offset=0)
        assert len(first_page) == 5

        second_page = self.db.list_memories(limit=5, offset=5)
        assert len(second_page) == 5

    def test_list_memories_default_limit(self):
        for i in range(60):
            self.db.add_memory(MemoryItem(content=f"fact {i}"))

        default = self.db.list_memories()
        assert len(default) == 50  # default limit

    def test_search_memories(self):
        self.db.add_memory(MemoryItem(content="I love Python programming"))
        self.db.add_memory(MemoryItem(content="I dislike spicy food"))

        results = self.db.search_memories("Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_search_memories_multiple_results(self):
        self.db.add_memory(MemoryItem(content="I love Python"))
        self.db.add_memory(MemoryItem(content="Python is great"))
        self.db.add_memory(MemoryItem(content="I hate bugs"))

        results = self.db.search_memories("Python")
        assert len(results) == 2

    def test_search_memories_no_results(self):
        self.db.add_memory(MemoryItem(content="completely different topic"))
        results = self.db.search_memories("nonexistentword")
        assert len(results) == 0

    def test_memory_ordering(self):
        for i in range(3):
            self.db.add_memory(MemoryItem(content=f"fact {i}"))

        memories = self.db.list_memories()
        assert memories[0].content == "fact 2"  # Most recent first

    def test_get_stats(self):
        self.db.add_memory(MemoryItem(content="fact", memory_type=MemoryType.FACT))
        self.db.add_memory(MemoryItem(content="pref", memory_type=MemoryType.PREFERENCE))

        stats = self.db.get_stats()
        assert stats["total_memories"] == 2
        assert stats["by_type"]["fact"] == 1
        assert stats["by_type"]["preference"] == 1

    def test_memory_with_tags_and_metadata(self):
        item = MemoryItem(
            content="complex fact",
            tags=["tag1", "tag2"],
            metadata={"source": "test", "category": "example"},
        )
        item_id = self.db.add_memory(item)

        retrieved = self.db.get_memory(item_id)
        assert retrieved is not None
        assert retrieved.tags == ["tag1", "tag2"]
        assert retrieved.metadata == {"source": "test", "category": "example"}

    def test_update_memory_tags(self):
        item = MemoryItem(content="test", tags=["original"])
        item_id = self.db.add_memory(item)

        updated = self.db.update_memory(item_id, tags=["new", "tags"])
        assert updated.tags == ["new", "tags"]


class TestConversationStorage:
    """Tests for conversation turn storage."""

    def setup_method(self):
        self.db = MemoryDatabase(":memory:")

    def teardown_method(self):
        self.db.close()

    def test_add_and_get_conversation(self):
        session_id = "test_session_1"
        turns = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi there!"),
        ]

        for turn in turns:
            self.db.add_conversation_turn(session_id, turn)

        retrieved = self.db.get_conversation(session_id)
        assert len(retrieved) == 2
        assert retrieved[0].role == "user"
        assert retrieved[1].role == "assistant"

    def test_conversation_ordering(self):
        session_id = "test_session_2"
        import time

        for i in range(5):
            turn = ConversationTurn(role="user", content=f"Message {i}")
            self.db.add_conversation_turn(session_id, turn)
            time.sleep(0.01)

        retrieved = self.db.get_conversation(session_id)
        assert retrieved[0].content == "Message 0"  # First in
        assert retrieved[-1].content == "Message 4"  # Last in

    def test_delete_conversation(self):
        session_id = "test_session_3"
        for i in range(3):
            self.db.add_conversation_turn(session_id, ConversationTurn(role="user", content=f"msg {i}"))

        deleted = self.db.delete_conversation(session_id)
        assert deleted == 3

        remaining = self.db.get_conversation(session_id)
        assert len(remaining) == 0

    def test_conversation_with_metadata(self):
        session_id = "test_session_4"
        turn = ConversationTurn(
            role="user",
            content="test",
            metadata={"model": "test-model", "tokens": 100},
        )
        self.db.add_conversation_turn(session_id, turn)

        retrieved = self.db.get_conversation(session_id)
        assert retrieved[0].metadata == {"model": "test-model", "tokens": 100}


class TestContextStorage:
    """Tests for context entry storage."""

    def setup_method(self):
        self.db = MemoryDatabase(":memory:")

    def teardown_method(self):
        self.db.close()

    def test_add_and_get_context(self):
        session_id = "ctx_session_1"
        entry = ContextEntry(
            category="facts",
            content="important fact",
            relevance_score=0.9,
        )
        self.db.add_context_entry(session_id, entry)

        retrieved = self.db.get_context(session_id)
        assert len(retrieved) == 1
        assert retrieved[0].category == "facts"
        assert retrieved[0].relevance_score == 0.9

    def test_context_sorted_by_relevance(self):
        session_id = "ctx_session_2"
        self.db.add_context_entry(session_id, ContextEntry(category="c", content="low", relevance_score=0.1))
        self.db.add_context_entry(session_id, ContextEntry(category="a", content="high", relevance_score=0.9))
        self.db.add_context_entry(session_id, ContextEntry(category="b", content="medium", relevance_score=0.5))

        retrieved = self.db.get_context(session_id)
        assert retrieved[0].relevance_score == 0.9
        assert retrieved[1].relevance_score == 0.5
        assert retrieved[2].relevance_score == 0.1


class TestCleanup:
    """Tests for expired memory cleanup."""

    def setup_method(self):
        self.db = MemoryDatabase(":memory:")

    def teardown_method(self):
        self.db.close()

    def test_cleanup_expired_memories(self):
        # Add a permanent memory
        self.db.add_memory(MemoryItem(content="permanent", score=MemoryScore(ttl_hours=None)))
        # Add an expired memory
        expired = MemoryItem(content="expired", score=MemoryScore(ttl_hours=1.0, last_accessed=time.time() - 7200))
        self.db.add_memory(expired)

        count = self.db.cleanup_expired()
        assert count == 1

        remaining = self.db.list_memories()
        assert len(remaining) == 1
        assert remaining[0].content == "permanent"

    def test_cleanup_no_expired(self):
        self.db.add_memory(MemoryItem(content="valid", score=MemoryScore(ttl_hours=24.0, last_accessed=time.time())))
        count = self.db.cleanup_expired()
        assert count == 0

    def test_get_expired_memories(self):
        from memory_pipe.storage.models import MemoryItem as MI, MemoryScore as MS

        self.db.add_memory(MI(content="expired1", score=MS(ttl_hours=1.0, last_accessed=time.time() - 7200)))
        self.db.add_memory(MI(content="expired2", score=MS(ttl_hours=0.5, last_accessed=time.time() - 3600)))
        self.db.add_memory(MI(content="valid", score=MS(ttl_hours=24.0, last_accessed=time.time())))

        expired = self.db.get_expired_memories()
        assert len(expired) == 2
        assert set(m.content for m in expired) == {"expired1", "expired2"}
