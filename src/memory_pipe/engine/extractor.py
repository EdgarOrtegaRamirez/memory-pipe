"""Fact extraction engine for MemoryPipe."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from memory_pipe.storage.models import (
    ImportanceLevel,
    MemoryItem,
    MemoryScore,
    MemoryType,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of fact extraction from text."""

    facts: list[MemoryItem] = field(default_factory=list)
    preferences: list[MemoryItem] = field(default_factory=list)
    context: list[MemoryItem] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def total_facts(self) -> int:
        return len(self.facts) + len(self.preferences) + len(self.context)


class FactExtractor:
    """Extract structured facts from conversational text."""

    # Patterns for detecting facts
    FACT_PATTERNS = [
        # "I am / I'm ..."
        (r"\b(?:i(?:'m| am))\s+([a-z][a-z\s,;.!?\-]+(?:\s+is\s+\w+)?(?:\s+at\s+\w+[^.]*)?)\b", "identity"),
        # "I live in / I work at / I study at"
        (r"\b(?:i(?:'m| am))\s+(?:living\s+)?(?:in|at)\s+([a-zA-Z\s,.\-]+?)(?:\.|$)", "location"),
        # "I work as / I work as a"
        (r"\bi\s+work(?:\s+as)?\s+(?:a\s+)?([a-zA-Z\s]+?)(?:\.|$)", "profession"),
        # "I like / I love / I enjoy / I prefer"
        (r"\bi\s+(?:like|love|enjoy|prefer)\s+([a-zA-Z\s,]+?)(?:\.|$)", "preference"),
        # "I don't like / I hate"
        (r"\bi\s+(?:don't\s+like|dislike|hate)\s+([a-zA-Z\s,]+?)(?:\.|$)", "anti_preference"),
        # "I have a / I have"
        (r"\bi\s+have\s+(?:a\s+)?([a-zA-Z\s]+?)(?:\.|$)", "possession"),
        # "My name is / My name's"
        (r"\bmy\s+name\s+is\s+([a-zA-Z\s]+?)(?:\.|$)", "identity"),
        # "I was born in"
        (r"\bi\s+was\s+born\s+(?:in|on)\s+([a-zA-Z\s,]+?)(?:\.|$)", "identity"),
        # "I have (a|an) ..."
        (r"\bi\s+have\s+(?:a|an)\s+([a-zA-Z\s]+?)(?:\.|$)", "possession"),
    ]

    # Patterns for detecting preferences
    PREFERENCE_PATTERNS = [
        (r"\b(?:i(?:'m| am)|i)\s+(?:would\s+)?(?:prefer|like|love|enjoy|favor)\s+([a-zA-Z\s,]+?)(?:\.|$)", "preference"),
        (r"\b(?:i(?:'m| am)|i)\s+(?:don't\s+(?:like|want|need)|dislike)\s+([a-zA-Z\s,]+?)(?:\.|$)", "anti_preference"),
        (r"\bi\s+(?:always|never|usually|often|sometimes)\s+(?:like|prefer|use)\s+([a-zA-Z\s,]+?)(?:\.|$)", "habit"),
    ]

    # Patterns for detecting context
    CONTEXT_PATTERNS = [
        (r"\b(?:currently|right\s+now|these\s+days)\s+([a-zA-Z\s,]+?)(?:\.|$)", "current_state"),
        (r"\b(?:in\s+the\s+future|going\s+to|plan\s+to)\s+([a-zA-Z\s,]+?)(?:\.|$)", "plan"),
        (r"\b(?:used\s+to|previously|before)\s+([a-zA-Z\s,]+?)(?:\.|$)", "past"),
    ]

    def __init__(self) -> None:
        self._compiled_patterns: dict = {}

    def extract(self, text: str) -> ExtractionResult:
        """Extract facts, preferences, and context from text.

        Args:
            text: Input text to extract from.

        Returns:
            ExtractionResult containing categorized facts.
        """
        if not text or not text.strip():
            return ExtractionResult()

        result = ExtractionResult()
        text_lower = text.lower()

        # Extract facts
        for pattern, category in self.FACT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                fact_text = match.group(1).strip().rstrip(".,;!?")
                if fact_text and len(fact_text) > 2:
                    importance = self._assess_importance(fact_text, category)
                    fact = MemoryItem(
                        content=fact_text,
                        memory_type=MemoryType.FACT,
                        score=MemoryScore(
                            importance=importance,
                            confidence=0.7,
                            metadata={"source": "extraction", "category": category},
                        ),
                        tags=[category, "auto-extracted"],
                    )
                    result.facts.append(fact)

        # Extract preferences
        for pattern, category in self.PREFERENCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                pref_text = match.group(1).strip().rstrip(".,;!?")
                if pref_text and len(pref_text) > 2:
                    pref_type = MemoryType.PREFERENCE
                    if category == "anti_preference":
                        pref_text = f"NOT {pref_text}"
                    fact = MemoryItem(
                        content=pref_text,
                        memory_type=pref_type,
                        score=MemoryScore(
                            importance=ImportanceLevel.HIGH,
                            confidence=0.8,
                            metadata={"source": "extraction", "category": category},
                        ),
                        tags=[category, "auto-extracted"],
                    )
                    result.preferences.append(fact)

        # Extract context
        for pattern, category in self.CONTEXT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ctx_text = match.group(1).strip().rstrip(".,;!?")
                if ctx_text and len(ctx_text) > 2:
                    fact = MemoryItem(
                        content=ctx_text,
                        memory_type=MemoryType.CONTEXT,
                        score=MemoryScore(
                            importance=ImportanceLevel.MEDIUM,
                            confidence=0.6,
                            metadata={"source": "extraction", "category": category},
                        ),
                        tags=[category, "auto-extracted"],
                    )
                    result.context.append(fact)

        # Calculate overall confidence
        if result.total_facts > 0:
            confidences = [f.score.confidence for f in result.facts + result.preferences + result.context]
            result.confidence = sum(confidences) / len(confidences)

        # Deduplicate
        result.facts = self._deduplicate(result.facts)
        result.preferences = self._deduplicate(result.preferences)
        result.context = self._deduplicate(result.context)

        logger.info(
            "Extracted %d facts, %d preferences, %d context items from text",
            len(result.facts),
            len(result.preferences),
            len(result.context),
        )

        return result

    def extract_from_conversation(
        self, turns: list[dict[str, str]]
    ) -> ExtractionResult:
        """Extract facts from a list of conversation turns.

        Args:
            turns: List of dicts with 'role' and 'content' keys.

        Returns:
            ExtractionResult with all extracted facts.
        """
        combined_facts: list[MemoryItem] = []
        combined_prefs: list[MemoryItem] = []
        combined_ctx: list[MemoryItem] = []

        for turn in turns:
            if turn.get("role") == "user":
                result = self.extract(turn.get("content", ""))
                combined_facts.extend(result.facts)
                combined_prefs.extend(result.preferences)
                combined_ctx.extend(result.context)

        return ExtractionResult(
            facts=combined_facts,
            preferences=combined_prefs,
            context=combined_ctx,
        )

    def _assess_importance(self, text: str, category: str) -> ImportanceLevel:
        """Assess the importance of a fact based on its content and category."""
        text_lower = text.lower()

        # Critical facts
        critical_keywords = [
            "name", "born", "age", "death", "marriage", "divorce",
            "diagnosis", "condition", "allergy", "medication",
        ]
        for keyword in critical_keywords:
            if keyword in text_lower:
                return ImportanceLevel.CRITICAL

        # High importance for identity and profession
        if category in ("identity", "profession"):
            return ImportanceLevel.HIGH

        # Medium importance for location
        if category == "location":
            return ImportanceLevel.HIGH

        # Low importance for general possessions
        if category == "possession":
            return ImportanceLevel.LOW

        return ImportanceLevel.MEDIUM

    def _deduplicate(self, items: list[MemoryItem]) -> list[MemoryItem]:
        """Remove duplicate items based on content."""
        seen: set[str] = set()
        unique: list[MemoryItem] = []

        for item in items:
            content_lower = item.content.lower().strip()
            if content_lower not in seen:
                seen.add(content_lower)
                unique.append(item)

        return unique
