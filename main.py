import os
import asyncio
from typing import List, Dict

from dotenv import load_dotenv

# Import Dedalus SDK types
from dedalus_labs import Dedalus, AsyncDedalus, DedalusRunner

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DISCOVERY_MODEL = "openai/gpt-5-nano"            # Cheap model for tool discovery
EXECUTION_MODEL = "anthropic/claude-haiku-4-5"   # Smarter model for execution
SIMILARITY_THRESHOLD = 0.25                      # Min cosine score to match a tool
RELATIVE_SCORE_CUTOFF = 0.6                      # Tool must score >= 60% of the best match
HISTORY_WINDOW_FOR_DISCOVERY = 4                 # Last N messages sent to discovery phase

# ---------------------------------------------------------------------------
# MCP Registry — add your MCP servers here
# ---------------------------------------------------------------------------
MCP_REGISTRY: List[Dict] = [
    {
        "url": "tsion/yahoo-finance-mcp",
        "description": "Stock market data, financial stats, quotes, and ticker information for stocks and equities"
    },
    {
        "url": "issac/fetch-mcp",
        "description": "Fetch and read webpages, check robots.txt, ping URLs, and extract content from web pages"
    }
]

# ---------------------------------------------------------------------------
# Semantic similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 * norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


def cache_registry_embeddings(client: Dedalus) -> List[Dict]:
    """Embed all MCP_REGISTRY descriptions once at startup."""
    print("Caching registry embeddings...")
    cached = []
    descriptions = [tool["description"] for tool in MCP_REGISTRY]
    # Batch embed all descriptions in one call
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=descriptions
    )
    for i, tool in enumerate(MCP_REGISTRY):
        cached.append({
            "url": tool["url"],
            "description": tool["description"],
            "embedding": response.data[i].embedding,
        })
    print(f"Cached embeddings for {len(cached)} tools.")
    return cached


def search_tools(queries: List[str], client: Dedalus, registry_cache: List[Dict]) -> List[Dict]:
    """Semantic search: find matching tools for a list of queries.
    
    Uses two-tier filtering:
    1. Absolute threshold — tool must score above SIMILARITY_THRESHOLD
    2. Relative cutoff — tool must score >= RELATIVE_SCORE_CUTOFF * best_score_for_that_query
    This prevents weak false matches (e.g. city-info matching 'stock price' at 0.34
    when yahoo-finance matches at 0.74).
    """
    matched = {}  # url -> tool dict, deduped

    if not queries:
        return []

    # Batch embed all queries at once
    q_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=queries
    )

    for i, query in enumerate(queries):
        q_embed = q_response.data[i].embedding

        # Score all tools for this query
        scored = []
        for tool in registry_cache:
            score = cosine_similarity(q_embed, tool["embedding"])
            if score >= SIMILARITY_THRESHOLD:
                scored.append((tool, score))

        if not scored:
            continue

        # Apply relative cutoff: only keep tools within range of the best score
        best_score = max(s for _, s in scored)
        cutoff = best_score * RELATIVE_SCORE_CUTOFF

        for tool, score in scored:
            if score >= cutoff and tool["url"] not in matched:
                matched[tool["url"]] = {
                    "url": tool["url"],
                    "description": tool["description"],
                    "score": round(score, 4),
                }

    return list(matched.values())


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

async def run_conversation():
    # Initialize clients
    dedalus_sync = Dedalus(api_key=os.getenv("DEDALUS_API_KEY"))
    dedalus_async = AsyncDedalus(api_key=os.getenv("DEDALUS_API_KEY"))
    runner = DedalusRunner(dedalus_async)

    # Cache embeddings once at startup
    registry_cache = cache_registry_embeddings(dedalus_sync)

    # Conversation history (persists across turns)
    history: List[Dict] = []

    # Collector for discovered tool URLs (reset each turn)
    discovered_urls: List[str] = []

    # --- Local tool the agent can call to discover MCP servers ---
    def discover_tools(queries: list[str]) -> str:
        """Search for tool servers that provide the capabilities described in the queries.
        Each query should be a short natural language description of a capability you need.
        Call this with multiple queries if the task requires different capabilities.
        Example: discover_tools(["stock market data", "text translation"])"""
        results = search_tools(queries, dedalus_sync, registry_cache)
        for r in results:
            if r["url"] not in discovered_urls:
                discovered_urls.append(r["url"])
        if results:
            descriptions = [f"- {r['description']} (score: {r['score']})" for r in results]
            return "Found these capabilities:\n" + "\n".join(descriptions)
        return "No matching tools found for those queries."

    print("Dedalus Semantic MCP Router (multi-turn)")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        user_msg = {"role": "user", "content": user_input}

        # Reset discovery collector for this turn
        discovered_urls.clear()

        # ---------------------------------------------------------------
        # Phase 1: DISCOVERY (ephemeral — agent generates search queries)
        # ---------------------------------------------------------------
        # Send recent history window + new message (not full history)
        discovery_messages = history[-HISTORY_WINDOW_FOR_DISCOVERY:] + [user_msg]

        print("  [Discovering tools...]")
        try:
            discovery_result = await runner.run(
                messages=discovery_messages,
                tools=[discover_tools],
                instructions=(
                    "You are a tool discovery assistant. Your ONLY job is to determine "
                    "what external tool capabilities are needed to answer the user's question. "
                    "Call discover_tools with search queries describing the capabilities needed. "
                    "Use multiple queries if the question needs different capabilities. "
                    "If the question can be answered from your own knowledge without any "
                    "external tools, respond with just: NO_TOOLS_NEEDED"
                ),
                model=DISCOVERY_MODEL,
                max_steps=3,
            )
        except Exception as e:
            print(f"  [Discovery error: {e}]")
            discovered_urls.clear()

        # Check if the agent decided no tools are needed
        no_tools_needed = (
            not discovered_urls
            and discovery_result
            and "NO_TOOLS_NEEDED" in (discovery_result.final_output or "")
        )

        if discovered_urls:
            print(f"  [Discovered {len(discovered_urls)} tool(s): {', '.join(discovered_urls)}]")
        elif no_tools_needed:
            print("  [No external tools needed]")
        else:
            print("  [No matching tools found — answering from knowledge]")

        # ---------------------------------------------------------------
        # Phase 2: EXECUTION (real run with discovered MCP servers)
        # ---------------------------------------------------------------
        execution_messages = history + [user_msg]

        try:
            if no_tools_needed:
                # Simple case — no tools, just answer
                result = await runner.run(
                    messages=execution_messages,
                    model=EXECUTION_MODEL,
                )
            else:
                # Run with discovered MCP servers + escape hatch
                result = await runner.run(
                    messages=execution_messages,
                    model=EXECUTION_MODEL,
                    mcp_servers=discovered_urls if discovered_urls else None,
                    tools=[discover_tools],  # Escape hatch for mid-run discovery
                    max_steps=10,
                )

                # Phase 2b: If discover_tools was called during execution,
                # re-run once with expanded tool set
                if hasattr(result, 'tools_called') and 'discover_tools' in (result.tools_called or []):
                    print(f"  [Agent requested more tools — re-running with {len(discovered_urls)} tool(s)]")
                    result = await runner.run(
                        messages=result.to_input_list(),
                        model=EXECUTION_MODEL,
                        mcp_servers=discovered_urls if discovered_urls else None,
                        max_steps=10,
                    )

        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        # Update conversation history from the execution result
        history = result.to_input_list()

        print(f"\nAssistant: {result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(run_conversation())
