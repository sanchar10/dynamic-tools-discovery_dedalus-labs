"""Dedalus Semantic MCP Router — entry point.

Thin CLI that wires up Dedalus providers and starts the interactive
SmartRouter loop.
"""

import os
import asyncio

from dotenv import load_dotenv

from providers.dedalus_provider import DedalusAgentProvider, DedalusEmbeddingProvider
from router import SmartRouter, RouterConfig

load_dotenv()

# ---------------------------------------------------------------------------
# MCP Registry — add your MCP servers here
# ---------------------------------------------------------------------------
MCP_REGISTRY = [
    {
        "url": "tsion/yahoo-finance-mcp",
        "name": "Yahoo Finance",
        "category": "finance",
        "description": "Stock market data, financial stats, quotes, and ticker information for stocks and equities",
        "keywords": ["stocks", "equities", "MSFT", "AAPL", "ticker", "price", "market cap", "dividends"],
    },
    {
        "url": "issac/fetch-mcp",
        "name": "Web Fetch",
        "category": "web",
        "description": "Fetch and read webpages, check robots.txt, ping URLs, and extract content from web pages",
        "keywords": ["http", "html", "scrape", "headlines", "URL", "website", "crawl"],
    },
]


async def main() -> None:
    api_key = os.getenv("DEDALUS_API_KEY")
    if not api_key:
        print("Error: DEDALUS_API_KEY not set in environment.")
        return

    # Providers
    embeddings = DedalusEmbeddingProvider(api_key=api_key)
    agent = DedalusAgentProvider(api_key=api_key)

    # Router config
    config = RouterConfig(registry=MCP_REGISTRY)

    # Build router
    router = SmartRouter(agent=agent, embeddings=embeddings, config=config)
    await router.initialize()

    print("\nSemantic MCP Router (multi-turn)")
    print("Type 'quit' or 'exit' to stop.\n")

    try:
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

            try:
                response = await router.handle_turn(user_input)
                print(f"\nAssistant: {response}\n")
            except Exception as e:
                print(f"\nError: {e}\n")
    finally:
        router.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
