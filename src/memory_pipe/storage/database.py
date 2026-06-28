"""SQLite-backed memory database for MemoryPipe."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid

import sqlite_utils

from memory_pipe.storage.models import (
    ContextEntry,
    ConversationTurn,
    ImportanceLevel,
    MemoryItem,
    MemoryScore,
    MemoryType,
)


class MemoryDatabase:
    """SQLite-backed storage for memory items."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._db = sqlite_utils.Database(db_path)
        self._db.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        """Initialize database tables."""
        self._db["memories"].create(
            {
                "id": str,
                "content": str,
                "memory_type": str,
                "importance": str,
                "confidence": float,
                "access_count": int,
                "last_accessed": float,
                "ttl_hours": float,
                "tags": str,
                "metadata": str,
                "created_at": float,
                "updated_at": float,
            },
            pk="id",
            if_not_exists=True,
        )

        self._db["conversations"].create(
            {
                "id": str,
                "session_id": str,
                "role": str,
                "content": str,
                "timestamp": float,
                "metadata": str,
            },
            if_not_exists=True,
        )

        self._db["contexts"].create(
            {
                "id": str,
                "session_id": str,
                "category": str,
                "content": str,
                "relevance_score": float,
                "created_at": float,
            },
            if_not_exists=True,
        )

        self._db["memories"].create_index(["memory_type"], if_not_exists=True)
        self._db["memories"].create_index(["importance"], if_not_exists=True)
        self._db["memories"].create_index(["created_at"], if_not_exists=True)
        self._db["memories"].create_index(["tags"], if_not_exists=True)
        self._db["conversations"].create_index(["session_id"], if_not_exists=True)
        self._db["conversations"].create_index(["timestamp"], if_not_exists=True)
        self._db["contexts"].create_index(["session_id"], if_not_exists=True)

    # ---- Memory CRUD ----

    def add_memory(self, item: MemoryItem) -> str:
        """Add a memory item and return its ID."""
        self._db["memories"].insert(
            {
                "id": item.id,
                "content": item.content,
                "memory_type": item.memory_type.value,
                "importance": item.score.importance.value,
                "confidence": item.score.confidence,
                "access_count": item.score.access_count,
                "last_accessed": item.score.last_accessed,
                "ttl_hours": item.score.ttl_hours,
                "tags": json.dumps(item.tags),
                "metadata": json.dumps(item.metadata),
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            },
            alter=True,
        )
        return item.id

    def get_memory(self, memory_id: str) -> MemoryItem | None:
        """Get a memory item by ID."""
        try:
            row = self._db["memories"].get(memory_id)
        except sqlite_utils.db.NotFoundError:
            return None
        if row is None:
            return None
        return self._row_to_memory(row)

    def update_memory(self, memory_id: str, **kwargs) -> MemoryItem | None:
        """Update fields of a memory item."""
        updates: dict = {}
        for key, value in kwargs.items():
            if key == "score":
                if isinstance(value, MemoryScore):
                    updates["importance"] = value.importance.value
                    updates["confidence"] = value.confidence
                    updates["access_count"] = value.access_count
                    updates["last_accessed"] = value.last_accessed
                    updates["ttl_hours"] = value.ttl_hours
            elif key == "tags":
                updates["tags"] = json.dumps(value)
            elif key == "metadata":
                updates["metadata"] = json.dumps(value)
            elif key == "content":
                updates["content"] = value
            elif key == "memory_type":
                updates["memory_type"] = value.value if isinstance(value, MemoryType) else value
            elif key == "updated_at":
                updates["updated_at"] = value

        if updates:
            updates["updated_at"] = time.time()
            try:
                self._db["memories"].update(memory_id, updates)
            except sqlite_utils.db.NotFoundError:
                return None

        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item. Returns True if deleted."""
        try:
            self._db["memories"].delete(memory_id)
            return True
        except sqlite_utils.db.NotFoundError:
            return False

    def list_memories(
        self,
        memory_type: MemoryType | None = None,
        importance: ImportanceLevel | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
        include_expired: bool = False,
    ) -> list[MemoryItem]:
        """List memory items with optional filters."""
        conditions = []
        params: list = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)
        if importance:
            conditions.append("importance = ?")
            params.append(importance.value)
        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM memories
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = list(self._db.execute(query, params))
        memories = [self._row_to_memory(dict(row)) for row in rows]

        if not include_expired:
            memories = [m for m in memories if not m.score.is_expired]

        return memories

    def search_memories(self, query: str, limit: int = 20) -> list[MemoryItem]:
        """Simple text search across memory content and tags."""
        search_pattern = f"%{query}%"
        rows = list(
            self._db.execute(
                """SELECT * FROM memories
                   WHERE content LIKE ? OR tags LIKE ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (search_pattern, search_pattern, limit),
            )
        )
        return [self._row_to_memory(dict(row)) for row in rows]

    def get_expired_memories(self) -> list[MemoryItem]:
        """Get all expired memories for cleanup."""
        rows = list(
            self._db.execute(
                """SELECT * FROM memories
                   WHERE ttl_hours IS NOT NULL
                     AND last_accessed IS NOT NULL
                     AND (strftime('%s','now') - last_accessed) > (ttl_hours * 3600)
                   ORDER BY created_at ASC"""
            )
        )
        return [self._row_to_memory(dict(row)) for row in rows]

    def cleanup_expired(self) -> int:
        """Remove expired memories. Returns count removed."""
        expired = self.get_expired_memories()
        count = 0
        for mem in expired:
            if self.delete_memory(mem.id):
                count += 1
        return count

    def get_stats(self) -> dict:
        """Get database statistics."""
        total = self._db["memories"].count
        by_type: dict[str, int] = {}
        by_importance: dict[str, int] = {}
        for row in self._db.execute(
            "SELECT memory_type, COUNT(*) as cnt FROM memories GROUP BY memory_type"
        ):
            by_type[row["memory_type"]] = row["cnt"]
        for row in self._db.execute(
            "SELECT importance, COUNT(*) as cnt FROM memories GROUP BY importance"
        ):
            by_importance[row["importance"]] = row["cnt"]

        return {
            "total_memories": total,
            "by_type": by_type,
            "by_importance": by_importance,
            "db_path": self.db_path,
        }

    # ---- Conversation storage ----

    def add_conversation_turn(
        self, session_id: str, turn: ConversationTurn
    ) -> str:
        """Add a conversation turn to a session."""
        turn_id = uuid.uuid4().hex[:12]
        self._db["conversations"].insert(
            {
                "id": turn_id,
                "session_id": session_id,
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp,
                "metadata": json.dumps(turn.metadata),
            },
            alter=True,
        )
        return turn_id

    def get_conversation(
        self, session_id: str, limit: int = 50
    ) -> list[ConversationTurn]:
        """Get conversation turns for a session."""
        rows = list(
            self._db.execute(
                """SELECT * FROM conversations
                   WHERE session_id = ?
                   ORDER BY timestamp ASC
                   LIMIT ?""",
                (session_id, limit),
            )
        )

        turns = []
        for row in rows:
            turns.append(
                ConversationTurn(
                    role=row["role"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    metadata=json.loads(row["metadata"]),
                )
            )
        return turns

    def delete_conversation(self, session_id: str) -> int:
        """Delete all turns in a session. Returns count deleted."""
        # Count before deleting
        count_row = self._db.execute(
            "SELECT COUNT(*) as cnt FROM conversations WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        count = count_row["cnt"] if count_row else 0
        # Delete
        self._db.execute(
            "DELETE FROM conversations WHERE session_id = ?", (session_id,)
        )
        return count

    # ---- Context storage ----

    def add_context_entry(
        self, session_id: str, entry: ContextEntry
    ) -> str:
        """Add a context entry."""
        entry_id = f"{session_id}_{entry.category}_{time.time():.0f}"
        self._db["contexts"].insert(
            {
                "id": entry_id,
                "session_id": session_id,
                "category": entry.category,
                "content": entry.content,
                "relevance_score": entry.relevance_score,
                "created_at": time.time(),
            },
            alter=True,
        )
        return entry_id

    def get_context(self, session_id: str) -> list[ContextEntry]:
        """Get context entries for a session."""
        rows = list(
            self._db.execute(
                """SELECT * FROM contexts
                   WHERE session_id = ?
                   ORDER BY relevance_score DESC""",
                (session_id,),
            )
        )

        entries = []
        for row in rows:
            entries.append(
                ContextEntry(
                    category=row["category"],
                    content=row["content"],
                    relevance_score=row["relevance_score"],
                )
            )
        return entries

    # ---- Helpers ----

    def _row_to_memory(self, row: dict) -> MemoryItem:
        """Convert a database row dict to a MemoryItem."""
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            score=MemoryScore(
                importance=ImportanceLevel(row["importance"]),
                confidence=row["confidence"],
                last_accessed=row["last_accessed"],
                access_count=row["access_count"],
                ttl_hours=row["ttl_hours"],
            ),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()
