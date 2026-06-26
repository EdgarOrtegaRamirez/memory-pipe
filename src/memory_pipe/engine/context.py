"""Context builder for MemoryPipe."""

from __future__ import annotations

import logging
from typing import Optional

from memory_pipe.engine.extractor import ExtractionResult, FactExtractor
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

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds context strings from stored memories for AI agent sessions."""

    def __init__(
        self,
        db: MemoryDatabase,
        vector_store: Optional[VectorStore] = None,
        max_context_length: int = 2000,
    ):
        self.db = db
        self.vector_store = vector_store
        self.max_context_length = max_context_length
        self.extractor = FactExtractor()

    def build_context(
        self,
        user_query: Optional[str] = None,
        session_id: Optional[str] = None,
        include_facts: bool = True,
        include_preferences: bool = True,
        include_recent: bool = True,
        recent_count: int = 10,
    ) -> str:
        """Build a context string from stored memories.

        Args:
            user_query: Optional query to prioritize relevant memories.
            session_id: Optional session ID for conversation context.
            include_facts: Whether to include factual memories.
            include_preferences: Whether to include preferences.
            include_recent: Whether to include recent conversations.
            recent_count: Number of recent items to include.

        Returns:
            Formatted context string for LLM prompts.
        """
        sections: list[str] = []

        # Add user query context if provided
        if user_query:
            sections.append(f"User Query: {user_query}")

        # Get relevant memories
        memories = []

        if user_query and self.vector_store:
            # Use vector search for relevance
            results = self.vector_store.search(user_query, limit=20)
            for entry_id, score in results:
                mem = self.db.get_memory(entry_id)
                if mem and mem.score.importance != ImportanceLevel.LOW:
                    memories.append((mem, score))
        else:
            # Fall back to importance-based retrieval
            if include_facts:
                facts = self.db.list_memories(
                    memory_type=MemoryType.FACT,
                    importance=ImportanceLevel.HIGH,
                    limit=recent_count,
                )
                memories.extend([(m, 0.8) for m in facts])

            if include_preferences:
                prefs = self.db.list_memories(
                    memory_type=MemoryType.PREFERENCE,
                    limit=recent_count,
                )
                memories.extend([(m, 0.7) for m in prefs])

        # Sort by relevance score
        memories.sort(key=lambda x: x[1], reverse=True)

        # Build context sections
        if memories:
            seen_types: set[str] = set()
            for mem, score in memories:
                if score < 0.3:
                    break
                if len(sections) > 0:
                    sections.append("")

                category = mem.memory_type.value.upper()
                if category not in seen_types:
                    sections.append(f"## {category}")
                    seen_types.add(category)

                sections.append(mem.to_context_string())

        # Add recent conversation context
        if include_recent and session_id:
            turns = self.db.get_conversation(session_id, limit=recent_count)
            if turns:
                if sections:
                    sections.append("")
                sections.append("## RECENT CONVERSATION")
                for turn in turns[-5:]:  # Last 5 turns
                    role = turn.role.upper()
                    content = turn.content[:200]
                    sections.append(f"{role}: {content}")

        context = "\n".join(sections)

        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[: self.max_context_length - 50] + "\n\n[... context truncated ...]"

        logger.info("Built context with %d sections, %d chars", len(sections), len(context))
        return context

    def extract_and_store(self, text: str, tags: Optional[list[str]] = None) -> ExtractionResult:
        """Extract facts from text and store them in the database.

        Args:
            text: Text to extract facts from.
            tags: Optional tags to add to extracted memories.

        Returns:
            ExtractionResult with all extracted items.
        """
        result = self.extractor.extract(text)

        for fact in result.facts:
            if tags:
                fact.tags.extend(tags)
            self.db.add_memory(fact)
            # Also add to vector store if available
            if self.vector_store:
                from memory_pipe.search.vector_store import VectorEntry
                vector = self.vector_store._tfidf_vector(fact.content)
                self.vector_store.add(VectorEntry(
                    id=fact.id,
                    content=fact.content,
                    vector=vector,
                    metadata={"type": "fact", "tags": fact.tags},
                ))

        for pref in result.preferences:
            if tags:
                pref.tags.extend(tags)
            self.db.add_memory(pref)
            if self.vector_store:
                from memory_pipe.search.vector_store import VectorEntry
                vector = self.vector_store._tfidf_vector(pref.content)
                self.vector_store.add(VectorEntry(
                    id=pref.id,
                    content=pref.content,
                    vector=vector,
                    metadata={"type": "preference", "tags": pref.tags},
                ))

        for ctx in result.context:
            if tags:
                ctx.tags.extend(tags)
            self.db.add_memory(ctx)
            if self.vector_store:
                from memory_pipe.search.vector_store import VectorEntry
                vector = self.vector_store._tfidf_vector(ctx.content)
                self.vector_store.add(VectorEntry(
                    id=ctx.id,
                    content=ctx.content,
                    vector=vector,
                    metadata={"type": "context", "tags": ctx.tags},
                ))

        return result

    def add_conversation(
        self,
        session_id: str,
        turns: list[ConversationTurn],
        extract_facts: bool = True,
    ) -> ExtractionResult:
        """Add conversation turns and optionally extract facts.

        Args:
            session_id: Session identifier.
            turns: List of conversation turns.
            extract_facts: Whether to extract facts from user turns.

        Returns:
            ExtractionResult if extract_facts is True, empty result otherwise.
        """
        for turn in turns:
            self.db.add_conversation_turn(session_id, turn)

        if extract_facts:
            # Extract facts from user turns
            user_turns = [
                {"role": t.role, "content": t.content}
                for t in turns
                if t.role == "user"
            ]
            return self.extractor.extract_from_conversation(user_turns)

        return ExtractionResult()
