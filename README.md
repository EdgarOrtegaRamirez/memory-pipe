# MemoryPipe

A lightweight, local-first AI agent memory and context persistence library. Store, retrieve, and search facts, conversations, and context for AI agents — without cloud dependencies.

## Features

- **SQLite-backed storage** — Fast, portable, zero-config database
- **Automatic fact extraction** — Extract facts, preferences, and context from text using pattern matching
- **Memory scoring** — Importance levels, TTL support, access counting, and automatic expiration
- **Semantic search** — TF-IDF vector search (optional vector embeddings via numpy)
- **Context building** — Assemble relevant memories into LLM prompt context strings
- **Conversation tracking** — Store and retrieve multi-turn conversations
- **CLI tool** — Full command-line interface for all operations
- **Python library** — Clean API for integration into agent workflows
- **MCP server** — Tool definitions for AI agent integration
- **No cloud dependencies** — Everything runs locally

## Quick Start

```bash
# Install
pip install memory-pipe

# Add a memory
memory-pipe add "I prefer Python over JavaScript" --tags preferences --importance high

# Search memories
memory-pipe search "Python"

# View all memories
memory-pipe list

# Extract facts from text
memory-pipe extract "My name is Alice, I live in London, and I love coding" --tags auto

# Build context for LLM prompts
memory-pipe context --query "What does the user like?"

# Show stats
memory-pipe stats

# Clean up expired memories
memory-pipe cleanup
```

## Python API

```python
from memory_pipe.storage.database import MemoryDatabase
from memory_pipe.storage.models import MemoryItem, MemoryType, ImportanceLevel, MemoryScore
from memory_pipe.engine.context import ContextBuilder

# Create database
db = MemoryDatabase("my_memory.db")

# Add a memory
memory = MemoryItem(
    content="User prefers dark mode",
    memory_type=MemoryType.PREFERENCE,
    score=MemoryScore(importance=ImportanceLevel.HIGH, confidence=0.9),
    tags=["settings", "ui"],
)
db.add_memory(memory)

# Search
results = db.search_memories("dark mode", limit=10)

# Build context for LLM
builder = ContextBuilder(db)
context = builder.build_context(
    user_query="What are my preferences?",
    include_facts=True,
    include_preferences=True,
)

# Use in prompt
prompt = f"{context}\n\nUser: What are my preferences?"

db.close()
```

## Fact Extraction

MemoryPipe extracts facts from text using pattern matching:

| Category | Pattern | Example |
|----------|---------|---------|
| Identity | "My name is...", "I am..." | "My name is Alice" |
| Location | "I live in..." | "I live in London" |
| Profession | "I work as..." | "I work as a developer" |
| Preference | "I like/love/enjoy..." | "I love Python" |
| Anti-preference | "I don't like/hate..." | "I hate spicy food" |
| Context | "Currently...", "I plan to..." | "Currently working on project" |

```python
from memory_pipe.engine.extractor import FactExtractor

extractor = FactExtractor()
result = extractor.extract("My name is Bob. I work as a doctor. I love hiking.")

for fact in result.facts:
    print(f"{fact.content} [{fact.memory_type.value}]")

for pref in result.preferences:
    print(f"Preference: {pref.content}")
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `memory-pipe add` | Add a memory item |
| `memory-pipe list` | List/search stored memories |
| `memory-pipe search` | Search memories by text |
| `memory-pipe extract` | Extract facts from text |
| `memory-pipe history` | View conversation history |
| `memory-pipe context` | Build context string |
| `memory-pipe stats` | Show database statistics |
| `memory-pipe cleanup` | Remove expired memories |
| `memory-pipe sample-config` | Print sample configuration |

## Architecture

```
memory_pipe/
├── storage/
│   ├── models.py      # Data models (MemoryItem, MemoryScore, etc.)
│   └── database.py    # SQLite-backed storage engine
├── engine/
│   ├── extractor.py   # Fact extraction from text
│   └── context.py     # Context building for LLM prompts
├── search/
│   └── vector_store.py # TF-IDF vector search
├── cli/
│   └── main.py        # CLI application
└── mcp_server/
    └── tools.py       # MCP tool definitions
```

## Installation

```bash
# Core installation
pip install memory-pipe

# With vector search support
pip install memory-pipe[vector]

# Development installation
pip install memory-pipe[dev]
```

## Configuration

```yaml
# memory-pipe.yaml
database:
  path: memory_pipe.db
  vacuum_interval_hours: 24

extraction:
  auto_extract: true
  default_importance: medium
  default_tags: [auto]

cleanup:
  enabled: true
  expired_only: true
  max_items: 10000

context:
  max_length: 4000
  include_facts: true
  include_preferences: true
  include_recent: true
  recent_count: 10
```

## License

MIT License — see [LICENSE](LICENSE) for details.
