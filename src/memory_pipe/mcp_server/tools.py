"""MCP server tools for MemoryPipe."""

from __future__ import annotations

import json
import logging

from memory_pipe.engine.extractor import FactExtractor
from memory_pipe.storage.database import MemoryDatabase
from memory_pipe.storage.models import ImportanceLevel, MemoryType

logger = logging.getLogger(__name__)


def create_memory_tools(db_path: str = ":memory:") -> list[dict]:
    """Create MCP tool definitions for memory operations."""

    tools = [
        {
            "name": "add_memory",
            "description": "Add a memory item (fact, preference, context, note, or task).",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store.",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": [m.value for m in MemoryType],
                        "description": "Type of memory to store.",
                        "default": "fact",
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Importance level.",
                        "default": "medium",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for the memory.",
                        "default": [],
                    },
                },
                "required": ["content"],
            },
        },
        {
            "name": "search_memories",
            "description": "Search stored memories by text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results.",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "list_memories",
            "description": "List stored memories with optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": [m.value for m in MemoryType],
                        "description": "Filter by memory type.",
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Filter by importance.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results.",
                        "default": 20,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "extract_facts",
            "description": "Extract facts, preferences, and context from text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to extract from.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to add to extracted facts.",
                        "default": [],
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "get_memory",
            "description": "Get a specific memory by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Memory ID to retrieve.",
                    },
                },
                "required": ["memory_id"],
            },
        },
        {
            "name": "delete_memory",
            "description": "Delete a memory by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Memory ID to delete.",
                    },
                },
                "required": ["memory_id"],
            },
        },
        {
            "name": "get_stats",
            "description": "Get database statistics.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    ]

    return tools


def create_memory_handlers(db_path: str = ":memory:") -> dict:
    """Create handler functions for MCP tools."""
    db = MemoryDatabase(db_path)
    extractor = FactExtractor()

    handlers = {
        "add_memory": lambda params: _handle_add_memory(db, params),
        "search_memories": lambda params: _handle_search(db, params),
        "list_memories": lambda params: _handle_list(db, params),
        "extract_facts": lambda params: _handle_extract(extractor, db, params),
        "get_memory": lambda params: _handle_get(db, params),
        "delete_memory": lambda params: _handle_delete(db, params),
        "get_stats": lambda params: _handle_stats(db),
    }

    return handlers


def _handle_add_memory(db: MemoryDatabase, params: dict) -> str:
    from memory_pipe.storage.models import ImportanceLevel, MemoryItem, MemoryScore, MemoryType

    content = params.get("content", "")
    if not content:
        return json.dumps({"error": "Content is required"})

    memory = MemoryItem(
        content=content,
        memory_type=MemoryType(params.get("memory_type", "fact")),
        score=MemoryScore(
            importance=ImportanceLevel(params.get("importance", "medium")),
            confidence=params.get("confidence", 0.5),
        ),
        tags=params.get("tags", []),
    )

    db.add_memory(memory)
    return json.dumps({
        "success": True,
        "id": memory.id,
        "type": memory.memory_type.value,
    })


def _handle_search(db: MemoryDatabase, params: dict) -> str:
    query = params.get("query", "")
    limit = params.get("limit", 10)
    memories = db.search_memories(query, limit=limit)
    return json.dumps([m.model_dump() for m in memories])


def _handle_list(db: MemoryDatabase, params: dict) -> str:
    memory_type = params.get("memory_type")
    importance = params.get("importance")
    limit = params.get("limit", 20)

    memories = db.list_memories(
        memory_type=MemoryType(memory_type) if memory_type else None,
        importance=ImportanceLevel(importance) if importance else None,
        limit=limit,
    )
    return json.dumps([m.model_dump() for m in memories])


def _handle_extract(extractor: FactExtractor, db: MemoryDatabase, params: dict) -> str:
    text = params.get("text", "")
    tags = params.get("tags", [])

    result = extractor.extract(text)
    if tags:
        for item in result.facts + result.preferences + result.context:
            item.tags.extend(tags)
            db.add_memory(item)

    return json.dumps({
        "confidence": result.confidence,
        "total_facts": result.total_facts,
        "facts": [f.model_dump() for f in result.facts],
        "preferences": [f.model_dump() for f in result.preferences],
        "context": [f.model_dump() for f in result.context],
    })


def _handle_get(db: MemoryDatabase, params: dict) -> str:
    memory_id = params.get("memory_id", "")
    memory = db.get_memory(memory_id)
    if memory:
        return json.dumps(memory.model_dump())
    return json.dumps({"error": f"Memory not found: {memory_id}"})


def _handle_delete(db: MemoryDatabase, params: dict) -> str:
    memory_id = params.get("memory_id", "")
    deleted = db.delete_memory(memory_id)
    return json.dumps({"success": deleted, "id": memory_id})


def _handle_stats(db: MemoryDatabase) -> str:
    return json.dumps(db.get_stats())
