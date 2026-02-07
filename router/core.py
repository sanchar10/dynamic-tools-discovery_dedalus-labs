"""SmartRouter — the main orchestrator.

Replaces the two-phase (Discovery → Execution) architecture with a single
execution pass that uses an LRU tool cache. The agent discovers new tools
on-demand via ``discover_tools`` and only triggers a re-run when new servers
are found.

Flow per turn:
  1. Build messages from conversation history + new user input.
  2. Execute with cached MCP servers + ``discover_tools`` always available.
  3. If ``discover_tools`` was called → add new servers to LRU → re-run
     with expanded server set.
  4. Post-run: touch used servers in LRU, mark failures, record metrics,
     update conversation history.
"""

from __future__ import annotations

import sys
from typing import AsyncIterator, Dict, List

from providers.base import AgentProvider, EmbeddingProvider

from .config import RouterConfig
from .health import HealthTracker
from .history import ConversationHistory
from .metrics import UsageMetrics
from .registry import ToolRegistry
from .tool_cache import ToolCache


# ---------------------------------------------------------------------------
# Instructions template
# ---------------------------------------------------------------------------

_AGENT_INSTRUCTIONS = """\
You are a helpful assistant with access to external tool servers.

{cache_status}

If you need a capability that is NOT available in the currently connected tools, \
call the discover_tools function with short search queries describing what you need. \
When formulating search queries, consider the full conversation context — not just \
the latest message. For example, if the user discussed Microsoft and then asks \
"what is its stock price?", search for "stock market data" rather than guessing.

Example: discover_tools(["stock market data", "text translation"])

If no external tools are needed, answer directly from your knowledge. \
Do NOT call discover_tools if you already have the right tools connected.\
"""


class SmartRouter:
    """Provider-agnostic agent router with LRU tool caching."""

    def __init__(
        self,
        agent: AgentProvider,
        embeddings: EmbeddingProvider,
        config: RouterConfig,
    ):
        self._agent = agent
        self._config = config

        # Sub-systems
        self._registry = ToolRegistry(
            embeddings,
            config.registry,
            similarity_threshold=config.similarity_threshold,
            relative_score_cutoff=config.relative_score_cutoff,
        )
        self._cache = ToolCache(max_size=config.cache_max_size)
        self._health = HealthTracker(cooldown_seconds=config.health_cooldown_seconds)
        self._history = ConversationHistory(max_turns=config.max_history_turns)
        self._metrics = UsageMetrics(metrics_file=config.metrics_file)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Startup: cache embeddings + preload popular tools."""
        print("Caching registry embeddings...")
        count = self._registry.cache_embeddings()
        print(f"Cached embeddings for {count} tool(s).")

        # Preload from historical usage
        top = self._metrics.get_top_tools(self._config.preload_count)
        if top:
            # Only preload tools still in the registry
            registry_urls = {t["url"] for t in self._config.registry}
            to_preload = [u for u in top if u in registry_urls]
            self._cache.preload(to_preload)
            if to_preload:
                print(f"Preloaded {len(to_preload)} tool(s) from usage history: {', '.join(to_preload)}")

    def shutdown(self) -> None:
        """Flush metrics on exit."""
        self._metrics.flush_session()

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------

    async def handle_turn(self, user_input: str) -> str:
        """Process a single user turn. Returns the assistant response."""

        # Mutable list shared with the discover_tools closure
        newly_discovered: List[str] = []

        # --- discover_tools closure ---
        def discover_tools(queries: list[str]) -> str:
            """Search for tool servers that provide the capabilities described in the queries.
            Each query should be a short natural-language description of a capability you need.
            Call this with multiple queries if the task requires different capabilities.
            Example: discover_tools(["stock market data", "text translation"])"""
            results = self._registry.search(queries)
            for r in results:
                url = r["url"]
                if self._health.is_healthy(url) and url not in newly_discovered:
                    newly_discovered.append(url)
                    evicted = self._cache.add(url)
                    if evicted:
                        _log(f"  [Cache evicted: {evicted}]")
            if results:
                descs = [f"- {r['description']} (score: {r['score']})" for r in results]
                return "Found these capabilities:\n" + "\n".join(descs)
            return "No matching tools found for those queries."

        # Build messages
        messages = self._history.append_user(user_input)

        # Active servers (healthy subset of cache)
        active_urls = self._health.filter_healthy(self._cache.get_urls())
        instructions = self._build_instructions(active_urls)

        # --- Execution run ---
        result = await self._agent.run(
            messages=messages,
            model=self._config.execution_model,
            tools=[discover_tools],
            mcp_servers=active_urls if active_urls else None,
            instructions=instructions,
            max_steps=self._config.max_steps,
        )

        # --- Check if discovery happened → re-run with new servers ---
        if newly_discovered:
            _log(f"  [Discovered {len(newly_discovered)} new tool(s): {', '.join(newly_discovered)}]")
            # Rebuild with updated cache — fresh execution, not continuation
            active_urls = self._health.filter_healthy(self._cache.get_urls())
            instructions = self._build_instructions(active_urls)
            result = await self._agent.run(
                messages=messages,  # original messages, not first run's output
                model=self._config.execution_model,
                tools=[discover_tools],
                mcp_servers=active_urls if active_urls else None,
                instructions=instructions,
                max_steps=self._config.max_steps,
            )

        # --- Post-run processing ---
        self._post_run(result, active_urls)

        return result.final_output

    async def handle_turn_stream(self, user_input: str) -> AsyncIterator[str]:
        """Process a turn with streaming output.

        If discovery is triggered, the discovery run is non-streaming.
        Only the final execution run is streamed.
        """
        newly_discovered: List[str] = []

        def discover_tools(queries: list[str]) -> str:
            """Search for tool servers that provide the capabilities described in the queries.
            Each query should be a short natural-language description of a capability you need.
            Call this with multiple queries if the task requires different capabilities.
            Example: discover_tools(["stock market data", "text translation"])"""
            results = self._registry.search(queries)
            for r in results:
                url = r["url"]
                if self._health.is_healthy(url) and url not in newly_discovered:
                    newly_discovered.append(url)
                    self._cache.add(url)
            if results:
                descs = [f"- {r['description']} (score: {r['score']})" for r in results]
                return "Found these capabilities:\n" + "\n".join(descs)
            return "No matching tools found for those queries."

        messages = self._history.append_user(user_input)
        active_urls = self._health.filter_healthy(self._cache.get_urls())
        instructions = self._build_instructions(active_urls)

        # Probe run (non-streaming) — may trigger discovery
        result = await self._agent.run(
            messages=messages,
            model=self._config.execution_model,
            tools=[discover_tools],
            mcp_servers=active_urls if active_urls else None,
            instructions=instructions,
            max_steps=self._config.max_steps,
        )

        if newly_discovered:
            _log(f"  [Discovered {len(newly_discovered)} new tool(s): {', '.join(newly_discovered)}]")
            active_urls = self._health.filter_healthy(self._cache.get_urls())
            instructions = self._build_instructions(active_urls)
            # Stream the re-run with original messages
            collected: List[str] = []
            async for chunk in self._agent.run_stream(
                messages=messages,  # original messages, not first run's output
                model=self._config.execution_model,
                tools=[discover_tools],
                mcp_servers=active_urls if active_urls else None,
                instructions=instructions,
                max_steps=self._config.max_steps,
            ):
                collected.append(chunk)
                yield chunk

            # Update history from collected output
            # (streaming doesn't return a RunResult, so we reconstruct minimally)
            full_output = "".join(collected)
            self._history.update(
                result.messages + [{"role": "assistant", "content": full_output}]
            )
        else:
            # No discovery — just yield the full result
            yield result.final_output
            self._post_run(result, active_urls)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _post_run(self, result, active_urls: List[str]) -> None:
        """LRU touch, health tracking, metrics, history update."""
        # Touch servers that were actually used
        for mr in result.mcp_results:
            if mr.is_error:
                _log(f"  [Server error: {mr.server_url}/{mr.tool_name} — marking unhealthy]")
                self._health.mark_unhealthy(mr.server_url)
                self._cache.evict(mr.server_url)
            else:
                self._cache.touch(mr.server_url)
                self._metrics.record_tool_use(mr.server_url)

        # Update conversation history
        self._history.update(result.messages)

    def _build_instructions(self, active_urls: List[str]) -> str:
        """Generate agent instructions reflecting current cache state."""
        if active_urls:
            names = ", ".join(active_urls)
            status = f"You currently have these tool servers connected: {names}."
        else:
            status = "You have no tool servers connected yet."
        return _AGENT_INSTRUCTIONS.format(cache_status=status)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def cache_contents(self) -> List[str]:
        return self._cache.get_urls()

    @property
    def history_turns(self) -> int:
        return self._history.turn_count


def _log(msg: str) -> None:
    """Print to stderr so it doesn't mix with streamed output."""
    print(msg, file=sys.stderr, flush=True)
