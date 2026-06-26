# AGENTS.md — Notes for AI Agents

## Project Overview
MemoryPipe is a lightweight AI agent memory and context persistence library. It provides SQLite-backed storage for facts, preferences, conversations, and context — enabling AI agents to remember across sessions.

## Key Files
- `src/memory_pipe/storage/database.py` — SQLite storage engine with CRUD operations
- `src/memory_pipe/storage/models.py` — Pydantic data models
- `src/memory_pipe/engine/extractor.py` — Pattern-based fact extraction
- `src/memory_pipe/engine/context.py` — Context building for LLM prompts
- `src/memory_pipe/search/vector_store.py` — TF-IDF vector search (optional)
- `src/memory_pipe/cli/main.py` — CLI application using Click

## Architecture Decisions
- SQLite via `sqlite-utils` for zero-config local storage
- Pydantic v2 for data validation
- Pattern-based extraction (no ML dependencies required)
- Optional vector search via TF-IDF (numpy optional dependency)
- Click for CLI framework
- Rich for terminal formatting

## Testing
- Run: `python -m pytest tests/ -v`
- Tests cover models, database CRUD, extraction, vector search, and context building
- Use `:memory:` database in tests for isolation

## Conventions
- Use `:memory:` for test databases
- Always close databases after use
- Tag auto-extracted items with "auto-extracted"
- Importance levels: LOW, MEDIUM, HIGH, CRITICAL
