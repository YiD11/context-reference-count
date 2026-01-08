"""
Example: Using context-ref with LangGraph tools.

This example demonstrates how to use the ToolInterceptor to cache
tool call results and reduce redundant API calls.
"""



from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END

from context_ref import ToolInterceptor, CacheConfig


def search_web(query: str) -> str:
    """Simulate a web search tool."""
    print(f"[Actually calling search API with: {query}]")
    return f"Search results for: {query}"


def get_weather(location: str) -> str:
    """Simulate a weather API call."""
    print(f"[Actually calling weather API for: {location}]")
    return f"Weather in {location}: 72°F, sunny"


def main():
    from langchain_core.messages import AIMessage

    config = CacheConfig(
        similarity_threshold=0.75,
        reuse_threshold=0.95,
        max_cache_size=100,
        collection_name="langgraph_tool_cache",  # Use a unique collection name
    )

    interceptor = ToolInterceptor(
        config=config,
        on_cache_hit=lambda name, args, out: print(f"[CACHE HIT] {name}: {args}"),
        on_cache_miss=lambda name, args: print(f"[CACHE MISS] {name}: {args}"),
    )

    tools = [search_web, get_weather]

    tool_node = ToolNode(
        tools,
        wrap_tool_call=interceptor.create_wrapper(),
    )

    graph = StateGraph(MessagesState)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    app = graph.compile()

    print("=== First calls (cache misses) ===")

    search_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_web", "args": {"query": "what is Python"}, "id": "1"}
        ],
    )
    result1 = app.invoke({"messages": [search_call]})
    print(f"Result: {result1['messages'][-1].content}\n")

    weather_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_weather", "args": {"location": "San Francisco"}, "id": "2"}
        ],
    )
    result2 = app.invoke({"messages": [weather_call]})
    print(f"Result: {result2['messages'][-1].content}\n")

    print("=== Second calls (cache hits) ===")

    search_call_2 = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_web", "args": {"query": "what is Python"}, "id": "3"}
        ],
    )
    result3 = app.invoke({"messages": [search_call_2]})
    print(f"Result: {result3['messages'][-1].content}\n")

    print("=== Statistics ===")
    stats = interceptor.stats
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Total cached entries: {stats['total_entries']}")


if __name__ == "__main__":
    main()
