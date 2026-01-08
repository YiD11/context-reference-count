"""Tool interceptor wrapper for LangGraph."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from context_ref.core.cache import ToolCache
from context_ref.core.config import CacheConfig
from context_ref.core.models import CacheHit


class CacheDecision(Enum):
    """Decision on how to handle a tool call."""

    REUSE = auto()
    PROVIDE_CONTEXT = auto()
    EXECUTE = auto()


@dataclass
class DecisionResult:
    """Result of cache decision making."""

    decision: CacheDecision
    cache_hit: CacheHit | None = None
    context_hints: list[CacheHit] | None = None


class ToolInterceptor:
    """
    Interceptor for LangGraph tool calls with caching support.

    Wraps tool execution to check cache before calling and save results after.
    Integrates with LangGraph's ToolNode via tool_call_wrapper.

    Tracks two types of cache references:
    - reuse_count: When an entry is directly reused (high similarity >= reuse_threshold)
    - provide_context_count: When an entry is provided as context hint

    This split enables more nuanced scoring that weights direct reuse
    higher than context provision based on config.reuse_context_factor.
    """

    def __init__(
        self,
        cache: ToolCache | None = None,
        config: CacheConfig | None = None,
        on_cache_hit: Callable[[str, dict, Any], None] | None = None,
        on_cache_miss: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.config = config or CacheConfig()
        self.cache = cache or ToolCache(config=self.config)
        self._on_cache_hit = on_cache_hit
        self._on_cache_miss = on_cache_miss
        self._stats = {"hits": 0, "misses": 0, "context_provided": 0}

    def decide(
        self,
        tool_name: str,
        input_args: dict[str, Any],
    ) -> DecisionResult:
        """
        Decide how to handle a tool call based on cache state.

        Returns:
            DecisionResult with one of:
            - REUSE: Direct reuse of cached result (similarity >= reuse_threshold)
                     Increments reuse_count for the matched entry.
            - PROVIDE_CONTEXT: Provide historical context to the model.
                     Increments provide_context_count for matched entries.
            - EXECUTE: Execute the tool call normally (no cache hit).
        """
        hits = self.cache.search(tool_name, input_args)

        if not hits:
            return DecisionResult(decision=CacheDecision.EXECUTE)

        best_hit = hits[0]

        if best_hit.similarity >= self.config.reuse_threshold:
            self.cache.increment_reuse(best_hit.entry.id)
            return DecisionResult(
                decision=CacheDecision.REUSE,
                cache_hit=best_hit,
            )

        if best_hit.similarity >= self.config.similarity_threshold:
            successful_hits = [h for h in hits if h.entry.success]
            if successful_hits:
                context_hints = successful_hits[:3]
                for hit in context_hints:
                    self.cache.increment_context(hit.entry.id)
                return DecisionResult(
                    decision=CacheDecision.PROVIDE_CONTEXT,
                    context_hints=context_hints,
                )

        return DecisionResult(decision=CacheDecision.EXECUTE)

    def format_context_hints(self, hints: list[CacheHit]) -> str:
        """Format cache hints as context for the LLM."""
        if not hints:
            return ""

        lines = ["Historical tool usage suggestions:"]
        for i, hit in enumerate(hints, 1):
            total_refs = hit.entry.reuse_count + hit.entry.provide_context_count
            lines.append(
                f"\n{i}. Tool: {hit.entry.tool_name} "
                f"(similarity: {hit.similarity:.2f}, "
                f"reuse: {hit.entry.reuse_count}, "
                f"context: {hit.entry.provide_context_count}, "
                f"total_refs: {total_refs})"
            )
            lines.append(f"   Input: {hit.entry.input_text}")
            output_preview = str(hit.entry.output)[:200]
            if len(str(hit.entry.output)) > 200:
                output_preview += "..."
            lines.append(f"   Output: {output_preview}")

        return "\n".join(lines)

    def create_wrapper(self) -> Callable:
        """
        Create a wrapper function for LangGraph's ToolNode.

        Usage:
            interceptor = ToolInterceptor()
            tool_node = ToolNode(tools, tool_call_wrapper=interceptor.create_wrapper())
        """

        def wrapper(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], ToolMessage],
        ) -> ToolMessage:
            tool_name = request.tool_call["name"]
            tool_args = request.tool_call.get("args", {})

            decision_result = self.decide(tool_name, tool_args)

            if decision_result.decision == CacheDecision.REUSE:
                self._stats["hits"] += 1
                if self._on_cache_hit:
                    self._on_cache_hit(
                        tool_name,
                        tool_args,
                        decision_result.cache_hit.entry.output,
                    )
                return ToolMessage(
                    content=str(decision_result.cache_hit.entry.output),
                    tool_call_id=request.tool_call["id"],
                    name=tool_name,
                )

            if decision_result.decision == CacheDecision.PROVIDE_CONTEXT:
                self._stats["context_provided"] += 1

            self._stats["misses"] += 1
            if self._on_cache_miss:
                self._on_cache_miss(tool_name, tool_args)

            result = execute(request)

            success = not (hasattr(result, "status") and result.status == "error")
            self.cache.save(tool_name, tool_args, result.content, success=success)

            return result

        return wrapper

    def wrap_tool(self, tool: Callable) -> Callable:
        """
        Wrap a single tool function with caching.

        Usage:
            @interceptor.wrap_tool
            def my_tool(query: str) -> str:
                ...
        """

        def wrapped(*args, **kwargs) -> Any:
            tool_name = getattr(tool, "__name__", str(tool))
            tool_args = kwargs if kwargs else {"args": args}

            decision_result = self.decide(tool_name, tool_args)

            if decision_result.decision == CacheDecision.REUSE:
                self._stats["hits"] += 1
                if self._on_cache_hit:
                    self._on_cache_hit(
                        tool_name,
                        tool_args,
                        decision_result.cache_hit.entry.output,
                    )
                return decision_result.cache_hit.entry.output

            self._stats["misses"] += 1
            if self._on_cache_miss:
                self._on_cache_miss(tool_name, tool_args)

            result = tool(*args, **kwargs)
            self.cache.save(tool_name, tool_args, result, success=True)
            return result

        wrapped.__name__ = getattr(tool, "__name__", "wrapped_tool")
        wrapped.__doc__ = getattr(tool, "__doc__", None)
        return wrapped

    @property
    def stats(self) -> dict[str, int]:
        """Get interceptor statistics."""
        return {
            **self._stats,
            "hit_rate": (
                self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
                if (self._stats["hits"] + self._stats["misses"]) > 0
                else 0.0
            ),
            **self.cache.stats(),
        }

    def reset_stats(self) -> None:
        """Reset interceptor statistics."""
        self._stats = {"hits": 0, "misses": 0, "context_provided": 0}
