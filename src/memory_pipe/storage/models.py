"""Data models for MemoryPipe."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memories that can be stored."""

    FACT = "fact"
    CONVERSATION = "conversation"
    PREFERENCE = "preference"
    CONTEXT = "context"
    TASK = "task"
    NOTE = "note"


class ImportanceLevel(str, Enum):
    """Importance levels for memory scoring."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryScore(BaseModel):
    """Score for a memory item indicating its importance and persistence."""

    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    last_accessed: Optional[float] = None
    access_count: int = Field(ge=0, default=0)
    ttl_hours: Optional[float] = None  # None = permanent

    @property
    def effective_score(self) -> float:
        """Calculate effective score based on importance, confidence, and access."""
        importance_map = {
            ImportanceLevel.LOW: 0.25,
            ImportanceLevel.MEDIUM: 0.5,
            ImportanceLevel.HIGH: 0.75,
            ImportanceLevel.CRITICAL: 1.0,
        }
        base = importance_map.get(self.importance, 0.5)
        access_bonus = min(self.access_count * 0.02, 0.2)
        return min(base + access_bonus, 1.0)

    @property
    def is_expired(self) -> bool:
        """Check if this memory has expired based on TTL."""
        if self.ttl_hours is None:
            return False
        if self.last_accessed is None:
            return False
        return (time.time() - self.last_accessed) > (self.ttl_hours * 3600)


class MemoryItem(BaseModel):
    """A single memory item stored in the database."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str
    memory_type: MemoryType = MemoryType.FACT
    score: MemoryScore = Field(default_factory=MemoryScore)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: float = Field(default_factory=lambda: time.time())
    updated_at: float = Field(default_factory=lambda: time.time())

    def update_access(self) -> None:
        """Update access timestamp and count."""
        self.score.access_count += 1
        self.score.last_accessed = time.time()
        self.updated_at = time.time()

    def to_context_string(self, max_length: int = 500) -> str:
        """Convert memory item to a context string for LLM prompts."""
        parts = [f"[{self.memory_type.value}] {self.content}"]
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        if self.score.importance != ImportanceLevel.MEDIUM:
            parts.append(f"Importance: {self.score.importance.value}")
        text = " | ".join(parts)
        if len(text) > max_length:
            return text[: max_length - 3] + "..."
        return text


class ConversationTurn(BaseModel):
    """A single turn in a conversation memory."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = Field(default_factory=lambda: time.time())
    metadata: dict = Field(default_factory=dict)


class ContextEntry(BaseModel):
    """An entry in the context builder's output."""

    category: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0, default=1.0)

    def to_string(self) -> str:
        return f"[{self.category}] {self.content}"
