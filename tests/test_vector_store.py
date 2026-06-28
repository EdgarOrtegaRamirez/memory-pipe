"""Tests for MemoryPipe vector store."""

from memory_pipe.search.vector_store import VectorEntry, VectorStore


class TestVectorStore:
    """Tests for the VectorStore."""

    def setup_method(self):
        self.store = VectorStore(dimension=64)

    def test_add_and_count(self):
        entry = VectorEntry(id="1", content="hello world", vector=[0.1] * 64)
        self.store.add(entry)
        assert self.store.count == 1

    def test_add_multiple(self):
        for i in range(5):
            self.store.add(VectorEntry(id=str(i), content=f"document {i}", vector=[0.1] * 64))
        assert self.store.count == 5

    def test_search_returns_results(self):
        # Add entries first to build vocabulary
        self.store.add(VectorEntry(id="1", content="I love Python programming", vector=[0.5] * 64))
        self.store.add(VectorEntry(id="2", content="I hate bugs", vector=[0.5] * 64))

        results = self.store.search("Python", limit=5)
        assert len(results) >= 1
        assert results[0][0] == "1"

    def test_search_returns_empty_for_no_match(self):
        results = self.store.search("completely unrelated topic", limit=5)
        # With no vocabulary, search returns empty
        assert len(results) == 0

    def test_search_returns_multiple(self):
        self.store.add(VectorEntry(id="1", content="I love Python programming", vector=[0.5] * 64))
        self.store.add(VectorEntry(id="2", content="Python is great for coding", vector=[0.5] * 64))
        self.store.add(VectorEntry(id="3", content="I like JavaScript", vector=[0.5] * 64))

        results = self.store.search("Python", limit=5)
        assert len(results) >= 2

    def test_search_sorts_by_similarity(self):
        self.store.add(VectorEntry(id="1", content="I love Python programming", vector=[0.5] * 64))
        self.store.add(VectorEntry(id="2", content="Python is great for coding", vector=[0.5] * 64))

        results = self.store.search("Python", limit=5)
        assert len(results) >= 2
        # First result should have highest score
        assert results[0][1] >= results[1][1]

    def test_search_with_limit(self):
        for i in range(10):
            self.store.add(VectorEntry(id=str(i), content=f"doc {i}", vector=[0.1] * 64))

        results = self.store.search("doc", limit=3)
        assert len(results) <= 3

    def test_remove_entry(self):
        self.store.add(VectorEntry(id="1", content="test", vector=[0.1] * 64))
        self.store.add(VectorEntry(id="2", content="test2", vector=[0.1] * 64))
        assert self.store.count == 2

        self.store.remove("1")
        assert self.store.count == 1

    def test_remove_nonexistent(self):
        assert self.store.remove("nonexistent") is False

    def test_clear(self):
        for i in range(5):
            self.store.add(VectorEntry(id=str(i), content=f"doc {i}", vector=[0.1] * 64))
        assert self.store.count == 5

        self.store.clear()
        assert self.store.count == 0

    def test_search_with_empty_query(self):
        self.store.add(VectorEntry(id="1", content="test", vector=[0.1] * 64))
        results = self.store.search("")
        assert len(results) == 0

    def test_tfidf_vector_normalization(self):
        self.store.add(VectorEntry(id="1", content="hello world test", vector=[0.1] * 64))
        vector = self.store._tfidf_vector("hello world")
        # Vector should have some non-zero values
        assert any(v != 0 for v in vector)


class TestVectorEntry:
    """Tests for VectorEntry dataclass."""

    def test_defaults(self):
        entry = VectorEntry(id="1", content="test", vector=[0.1, 0.2])
        assert entry.id == "1"
        assert entry.content == "test"
        assert entry.vector == [0.1, 0.2]
        assert entry.metadata == {}

    def test_with_metadata(self):
        entry = VectorEntry(
            id="1",
            content="test",
            vector=[0.1, 0.2],
            metadata={"type": "fact"},
        )
        assert entry.metadata == {"type": "fact"}
