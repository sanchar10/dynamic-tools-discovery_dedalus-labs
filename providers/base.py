"""Abstract interfaces for agent and embedding providers.

Implement these to plug in any agent framework (Dedalus, Azure AI Agent Service,
Semantic Kernel, etc.) without changing the router logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes — provider-agnostic result types
# ---------------------------------------------------------------------------

@dataclass
class MCPToolResult:
    """A single MCP tool invocation result."""
    server_url: str
    tool_name: str
    is_error: bool = False
    duration_ms: Optional[int] = None


@dataclass
class RunResult:
    """Standardized result from an agent execution run."""
    final_output: str
    messages: List[Dict] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)
    mcp_results: List[MCPToolResult] = field(default_factory=list)
    steps_used: int = 0


# ---------------------------------------------------------------------------
# Abstract providers
# ---------------------------------------------------------------------------

class EmbeddingProvider(ABC):
    """Synchronous embedding provider.

    Must be sync because agent runners call tool functions synchronously —
    ``discover_tools`` needs to embed queries inline during an agent run.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts and return their vectors."""
        ...


class AgentProvider(ABC):
    """Async agent execution provider.

    Wraps an agentic runner that can call tools (local callables) and
    connect to MCP servers.
    """

    @abstractmethod
    async def run(
        self,
        messages: List[Dict],
        model: str,
        *,
        tools: Optional[List[Callable]] = None,
        mcp_servers: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        max_steps: int = 10,
    ) -> RunResult:
        """Execute an agent run and return a RunResult."""
        ...

    @abstractmethod
    async def run_stream(
        self,
        messages: List[Dict],
        model: str,
        *,
        tools: Optional[List[Callable]] = None,
        mcp_servers: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        max_steps: int = 10,
    ) -> AsyncIterator[str]:
        """Execute an agent run and stream text chunks."""
        ...
