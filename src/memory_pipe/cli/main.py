"""CLI module for MemoryPipe - AI Agent Memory & Context Persistence."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.text import Text

from memory_pipe.engine.context import ContextBuilder
from memory_pipe.engine.extractor import FactExtractor
from memory_pipe.search.vector_store import VectorStore
from memory_pipe.storage.database import MemoryDatabase
from memory_pipe.storage.models import (
    ConversationTurn,
    ContextEntry,
    ImportanceLevel,
    MemoryItem,
    MemoryScore,
    MemoryType,
)

logger = logging.getLogger("memory_pipe.cli")
console = Console()


def _load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from a YAML file."""
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found: %s", config_path)
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _format_memory_table(memories: list[MemoryItem]) -> Table:
    """Format memories as a rich table."""
    table = Table(title=f"Found {len(memories)} memory item(s)")
    table.add_column("ID", style="cyan", width=12)
    table.add_column("Type", style="magenta")
    table.add_column("Content", style="white")
    table.add_column("Importance", style="yellow")
    table.add_column("Score", style="green")
    table.add_column("Created", style="dim")

    for mem in memories:
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(mem.created_at))
        table.add_row(
            mem.id,
            mem.memory_type.value,
            mem.content[:80],
            mem.score.importance.value,
            f"{mem.score.effective_score:.2f}",
            created,
        )

    return table


@click.group()
@click.version_option(package_name="memory-pipe")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def main(ctx: click.Context, verbose: bool, config_path: Optional[str]) -> None:
    """MemoryPipe - AI Agent Memory & Context Persistence.

    Store, retrieve, and search facts, conversations, and context for AI agents.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = _load_config(config_path)

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)


@main.command()
@click.argument("content")
@click.option("-t", "--type", "memory_type", type=click.Choice([m.value for m in MemoryType]), default="fact")
@click.option("-i", "--importance", type=click.Choice([i.value for i in ImportanceLevel]), default="medium")
@click.option("-c", "--confidence", type=float, default=0.5, help="Extraction confidence (0.0-1.0)")
@click.option("-T", "--tags", multiple=True, help="Tags for the memory (can be specified multiple times)")
@click.option("-d", "--db", "db_path", default=":memory:", help="Database file path")
@click.pass_context
def add(
    ctx: click.Context,
    content: str,
    memory_type: str,
    importance: str,
    confidence: float,
    tags: tuple[str, ...],
    db_path: str,
) -> None:
    """Add a memory item to the database."""
    db = MemoryDatabase(db_path)

    memory = MemoryItem(
        content=content,
        memory_type=MemoryType(memory_type),
        score=MemoryScore(
            importance=ImportanceLevel(importance),
            confidence=confidence,
        ),
        tags=list(tags),
    )

    db.add_memory(memory)
    console.print(f"[green]✓[/] Added memory [bold]{memory.id}[/] ({memory_type})")
    console.print(f"  Content: {content[:100]}")
    console.print(f"  Tags: {', '.join(tags) or '(none)'}")

    db.close()


@main.command()
@click.argument("memory_id", default=None, required=False)
@click.option("-t", "--type", "memory_type", type=click.Choice([m.value for m in MemoryType]))
@click.option("-i", "--importance", type=click.Choice([i.value for i in ImportanceLevel]))
@click.option("-T", "--tags", multiple=True)
@click.option("-l", "--limit", default=20, help="Maximum results")
@click.option("-d", "--db", "db_path", default=":memory:", help="Database file path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def list_memories(
    ctx: click.Context,
    memory_id: Optional[str],
    memory_type: Optional[str],
    importance: Optional[str],
    tags: tuple[str, ...],
    limit: int,
    db_path: str,
    as_json: bool,
) -> None:
    """List or search stored memories."""
    db = MemoryDatabase(db_path)

    if memory_id:
        mem = db.get_memory(memory_id)
        if mem:
            if as_json:
                console.print(json.dumps(mem.model_dump(), indent=2))
            else:
                console.print(f"[bold]Memory:[/bold] {mem.id}")
                console.print(f"  Type: {mem.memory_type.value}")
                console.print(f"  Content: {mem.content}")
                console.print(f"  Importance: {mem.score.importance.value}")
                console.print(f"  Confidence: {mem.score.confidence:.2f}")
                console.print(f"  Tags: {', '.join(mem.tags) or '(none)'}")
                console.print(f"  Accesses: {mem.score.access_count}")
        else:
            console.print(f"[red]Memory not found:[/red] {memory_id}")
    else:
        memories = db.list_memories(
            memory_type=MemoryType(memory_type) if memory_type else None,
            importance=ImportanceLevel(importance) if importance else None,
            tags=list(tags),
            limit=limit,
        )

        if not memories:
            console.print("[yellow]No memories found.[/yellow]")
        elif as_json:
            console.print(json.dumps([m.model_dump() for m in memories], indent=2))
        else:
            console.print(_format_memory_table(memories))

    db.close()


@main.command()
@click.argument("query")
@click.option("-l", "--limit", default=10, help="Maximum results")
@click.option("-d", "--db", "db_path", default=":memory:", help="Database file path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    limit: int,
    db_path: str,
    as_json: bool,
) -> None:
    """Search memories by text content."""
    db = MemoryDatabase(db_path)
    memories = db.search_memories(query, limit=limit)

    if not memories:
        console.print("[yellow]No matching memories found.[/yellow]")
    elif as_json:
        console.print(json.dumps([m.model_dump() for m in memories], indent=2))
    else:
        console.print(_format_memory_table(memories))

    db.close()


@main.command()
@click.argument("text")
@click.option("-t", "--tags", multiple=True, help="Tags for extracted facts")
@click.option("-d", "--db", "db_path", default=":memory:", help="Database file path")
@click.option("--json", "as_json", is_flag=True, help="Output extraction results as JSON")
@click.pass_context
def extract(
    ctx: click.Context,
    text: str,
    tags: tuple[str, ...],
    db_path: str,
    as_json: bool,
) -> None:
    """Extract facts, preferences, and context from text."""
    extractor = FactExtractor()
    result = extractor.extract(text)

    if as_json:
        output = {
            "confidence": result.confidence,
            "total_facts": result.total_facts,
            "facts": [f.model_dump() for f in result.facts],
            "preferences": [f.model_dump() for f in result.preferences],
            "context": [f.model_dump() for f in result.context],
        }
        console.print(json.dumps(output, indent=2))
    else:
        console.print(f"[bold]Extraction Results[/bold] (confidence: {result.confidence:.2f})")

        if result.facts:
            console.print(f"\n[yellow]Facts ({len(result.facts)}):[/yellow]")
            for f in result.facts:
                console.print(f"  • {f.content}")

        if result.preferences:
            console.print(f"\n[yellow]Preferences ({len(result.preferences)}):[/yellow]")
            for f in result.preferences:
                console.print(f"  • {f.content}")

        if result.context:
            console.print(f"\n[yellow]Context ({len(result.context)}):[/yellow]")
            for f in result.context:
                console.print(f"  • {f.content}")

    db = MemoryDatabase(db_path)
    if tags:
        for fact in result.facts:
            fact.tags.extend(tags)
        for pref in result.preferences:
            pref.tags.extend(tags)
        for ctx_item in result.context:
            ctx_item.tags.extend(tags)
        for item in result.facts + result.preferences + result.context:
            db.add_memory(item)
        console.print(f"\n[green]✓[/] Saved {len(result.facts + result.preferences + result.context)} items to database")
    db.close()


@main.command()
@click.argument("session_id")
@click.option("-d", "--db", "db_path", default=":memory:", help="Database file path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def history(
    ctx: click.Context,
    session_id: str,
    db_path: str,
    as_json: bool,
) -> None:
    """View conversation history for a session."""
    db = MemoryDatabase(db_path)
    turns = db.get_conversation(session_id)

    if not turns:
        console.print(f"[yellow]No conversation history for session: {session_id}[/yellow]")
    elif as_json:
        console.print(json.dumps([t.model_dump() for t in turns], indent=2))
    else:
        console.print(f"[bold]Conversation History: {session_id}[/bold] ({len(turns)} turns)")
        for turn in turns:
            role_icon = {"user": "[blue]Q[/]", "assistant": "[green]A[/]", "system": "[dim]S[/]"}.get(
                turn.role, "[dim]?[/]"
            )
            content_preview = turn.content[:120]
            console.print(f"  {role_icon} {content_preview}")

    db.close()


@main.command()
@click.option("-d", "--db", "db_path", default=":memory:", help="Database file path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def stats(ctx: click.Context, db_path: str, as_json: bool) -> None:
    """Show database statistics."""
    db = MemoryDatabase(db_path)
    stats_data = db.get_stats()

    if as_json:
        console.print(json.dumps(stats_data, indent=2))
    else:
        console.print("[bold]MemoryPipe Statistics[/bold]")
        console.print(f"  Database: {stats_data['db_path']}")
        console.print(f"  Total memories: {stats_data['total_memories']}")

        if stats_data["by_type"]:
            console.print("\n  By type:")
            for mem_type, count in stats_data["by_type"].items():
                console.print(f"    {mem_type}: {count}")

        if stats_data["by_importance"]:
            console.print("\n  By importance:")
            for importance, count in stats_data["by_importance"].items():
                console.print(f"    {importance}: {count}")

    db.close()


@main.command()
@click.option("-d", "--db", "db_path", default=":memory:", help="Database file path")
@click.pass_context
def cleanup(ctx: click.Context, db_path: str) -> None:
    """Remove expired memories from the database."""
    db = MemoryDatabase(db_path)
    count = db.cleanup_expired()
    console.print(f"[green]✓[/] Removed {count} expired memory(s)")
    db.close()


@main.command(name="sample-config")
@click.pass_context
def sample_config(ctx: click.Context) -> None:
    """Print a sample configuration file."""
    sample = {
        "database": {
            "path": "memory_pipe.db",
            "vacuum_interval_hours": 24,
        },
        "extraction": {
            "auto_extract": True,
            "default_importance": "medium",
            "default_tags": ["auto"],
        },
        "cleanup": {
            "enabled": True,
            "expired_only": True,
            "max_items": 10000,
        },
        "context": {
            "max_length": 4000,
            "include_facts": True,
            "include_preferences": True,
            "include_recent": True,
            "recent_count": 10,
        },
    }
    console.print(yaml.dump(sample, default_flow_style=False))


@main.command(name="context")
@click.option("-q", "--query", help="User query for context prioritization")
@click.option("-s", "--session-id", help="Session ID for conversation context")
@click.option("--db", "db_path", default=":memory:", help="Database file path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def build_context(
    ctx: click.Context,
    query: Optional[str],
    session_id: Optional[str],
    db_path: str,
    as_json: bool,
) -> None:
    """Build context string from stored memories."""
    db = MemoryDatabase(db_path)
    builder = ContextBuilder(db)

    context = builder.build_context(
        user_query=query,
        session_id=session_id,
    )

    if as_json:
        console.print(json.dumps({"context": context, "length": len(context)}, indent=2))
    else:
        console.print("[bold]Built Context[/bold]")
        console.print(f"  Length: {len(context)} chars")
        console.print(f"  Query: {query or '(none)'}")
        console.print(f"  Session: {session_id or '(none)'}")
        console.print(f"\n{context}")

    db.close()


if __name__ == "__main__":
    main()
