"""Dedalus Labs SDK implementation of the provider interfaces."""

from __future__ import annotations

import os
from typing import AsyncIterator, Callable, Dict, List, Optional

from dedalus_labs import AsyncDedalus, Dedalus, DedalusRunner

from .base import AgentProvider, EmbeddingProvider, MCPToolResult, RunResult


class DedalusEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using the Dedalus Labs sync client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ):
        self._client = Dedalus(api_key=api_key or os.getenv("DEDALUS_API_KEY"))
        self._model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [d.embedding for d in response.data]


class DedalusAgentProvider(AgentProvider):
    """Agent provider using the Dedalus Labs async runner."""

    def __init__(self, api_key: Optional[str] = None):
        self._client = AsyncDedalus(api_key=api_key or os.getenv("DEDALUS_API_KEY"))
        self._runner = DedalusRunner(self._client)

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
        result = await self._runner.run(
            messages=messages,
            model=model,
            tools=tools,
            mcp_servers=mcp_servers or None,
            instructions=instructions,
            max_steps=max_steps,
        )
        return self._convert_result(result)

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
        stream = await self._runner.run(
            messages=messages,
            model=model,
            tools=tools,
            mcp_servers=mcp_servers or None,
            instructions=instructions,
            max_steps=max_steps,
            stream=True,
        )
        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content

    # ------------------------------------------------------------------

    @staticmethod
    def _convert_result(result) -> RunResult:
        """Map Dedalus _RunResult → provider-agnostic RunResult."""
        mcp_results = []
        for mr in getattr(result, "mcp_results", []) or []:
            mcp_results.append(
                MCPToolResult(
                    server_url=getattr(mr, "server_name", ""),
                    tool_name=getattr(mr, "tool_name", ""),
                    is_error=getattr(mr, "is_error", False),
                    duration_ms=getattr(mr, "duration_ms", None),
                )
            )

        return RunResult(
            final_output=result.final_output or "",
            messages=result.to_input_list(),
            tools_called=list(getattr(result, "tools_called", []) or []),
            mcp_results=mcp_results,
            steps_used=getattr(result, "steps_used", 0),
        )
