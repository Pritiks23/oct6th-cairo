from __future__ import annotations
from typing import Optional
from deepagents import create_deep_agent
from langchain_core.runnables import Runnable
from .llm import get_mc1_model
from .memory import CairoMemoryTools
from .tools.search import internet_search
from .tools.recommendation import (
    set_weights_tool,
    boost_creator_tool,
    demote_creator_tool,
    block_tag_tool,
    unblock_tag_tool,
    search_content_tool,
    trending_content_tool,
    personalized_feed_tool,
)
from .policy import guard_tools
from langchain.agents import DeepAgent  # <-- new import for explicit DeepAgent usage if needed

CAIRO_SYSTEM_INSTRUCTIONS = """
You are CAIRO - ColomboAI In-App Reactive Operator - an in-app agent that is context-aware, privacy-respectful, and action-oriented.
You operate within the ColomboAI ecosystem (GenAI, Feed, CAIRO, News, Generative Shop).

Mandatory restrictions (NON-NEGOTIABLE):
- You MUST NOT like posts, auto-like, comment on posts, or auto-comment. Never take such actions or suggest them.
- When acting on social surfaces, only read, summarize, plan, draft, schedule (with user confirmation), or recommend.
- Do not publish externally without explicit confirmation.

Recommendation Engine Control:
- You CAN control the recommendation engine via tools to set weights, boost/demote creators, and block/unblock tags.
- You CAN help users discover content using search, trending content, and personalized feeds.
- Prefer small, reversible changes and explain the expected impact when you act.
- Log what you changed and why (in your reply) so humans can review.

General guidance:
- Use long-term memory (Mem0) when helpful.
- Prefer concise, structured answers and include sources for web findings.
- For complex tasks, first write a short plan, then execute step-by-step.
"""

def build_cairo_agent(builtin_tools: Optional[list[str]] = None) -> Runnable:
    mem_tools = CairoMemoryTools()
    
    tools = [
        internet_search,        # SearxNG search
        mem_tools.add_tool,     # Mem0 write
        mem_tools.search_tool,  # Mem0 search
        mem_tools.get_all_tool, # Mem0 list
        # Recommendation engine controls
        set_weights_tool,
        boost_creator_tool,
        demote_creator_tool,
        block_tag_tool,
        unblock_tag_tool,
        # Social media discovery tools
        search_content_tool,
        trending_content_tool,
        personalized_feed_tool,
    ]
    
    tools = guard_tools(tools)  # enforce policy: block like/comment actions

    model = get_mc1_model(temperature=0.2, max_tokens=2048)

    # Create the DeepAgent via deepagents helper
    agent = create_deep_agent(
        tools=tools,
        instructions=CAIRO_SYSTEM_INSTRUCTIONS,
        model=model,
        builtin_tools=builtin_tools,
    )

    # Optional: expose a direct DeepAgent instance if you want programmatic access
    deep_agent_instance = DeepAgent(
        tools=tools,
        name="CAIRO_DeepAgent",
        description="Context-aware in-app agent controlling recommendations and content discovery"
    )

    return agent
