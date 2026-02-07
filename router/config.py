"""Router configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RouterConfig:
    """All tunables for the SmartRouter."""

    # --- Models ---
    execution_model: str = "anthropic/claude-haiku-4-5"

    # --- Semantic search ---
    similarity_threshold: float = 0.25
    relative_score_cutoff: float = 0.6

    # --- Tool cache ---
    cache_max_size: int = 10
    preload_count: int = 5

    # --- Conversation ---
    max_history_turns: int = 20
    max_steps: int = 10

    # --- Health ---
    health_cooldown_seconds: int = 300  # 5 minutes

    # --- Metrics ---
    metrics_file: Path = field(default_factory=lambda: Path("data/usage_metrics.jsonl"))

    # --- MCP Registry ---
    registry: List[Dict] = field(default_factory=list)
