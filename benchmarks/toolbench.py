"""ToolBench 数据集加载和处理。

18.7万训练样本，50+类别，16000+ RapidAPI 工具，G1/G2/G3 难度级别。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolBenchQuery:
    """查询数据。"""

    query_id: str
    query: str
    tool_name: str
    api_name: str | None = None
    category: str = ""
    required_params: dict[str, Any] = field(default_factory=dict)
    optional_params: dict[str, Any] = field(default_factory=dict)
    api_description: str = ""


@dataclass
class ToolCall:
    """对话中的工具调用。"""

    tool_name: str
    api_name: str
    arguments: dict[str, Any]
    output: Any | None = None
    query_id: str = ""
    step: int = 0


def load_toolbench_queries(
    data_dir: str | Path,
    subset: str = "G1",
    limit: int | None = None,
    categories: list[str] | None = None,
) -> list[ToolBenchQuery]:
    """加载查询数据（subset: G1/G2/G3）。

    Args:
        data_dir: 数据目录
        subset: G1/G2/G3 子集
        limit: 最多加载多少条
        categories: 类别筛选列表，None表示全选
    """
    data_dir = Path(data_dir)
    query_file = data_dir / "instruction" / f"{subset}_query.json"

    if not query_file.exists():
        raise FileNotFoundError(f"查询文件不存在: {query_file}")

    with open(query_file, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)

    queries = []
    for item in raw_queries[:limit] if limit else raw_queries:
        query_text = item.get("query", "")
        relevant_apis = item.get("relevant APIs", [])
        api_list = item.get("api_list", [])
        query_id = str(item.get("query_id", ""))

        # 跳过缺少关键信息的查询
        if not relevant_apis or not api_list:
            continue

        if categories:
            item_categories = {api.get("category_name", "") for api in api_list}
            if not any(cat in categories for cat in item_categories):
                continue

        # 提取主要API信息
        primary_api = relevant_apis[0]
        tool_name = primary_api[0] if isinstance(primary_api, list) else primary_api
        api_name = (
            primary_api[1]
            if isinstance(primary_api, list) and len(primary_api) > 1
            else None
        )

        # 查找对应的API详细信息
        api_info = None
        for api in api_list:
            if api.get("tool_name") == tool_name:
                if api_name is None or api.get("api_name") == api_name:
                    api_info = api
                    break

        # 提取参数信息
        required_params = {}
        optional_params = {}
        if api_info:
            for param in api_info.get("required_parameters", []):
                param_name = param.get("name", "")
                if param_name:
                    required_params[param_name] = param.get("default", "")
            for param in api_info.get("optional_parameters", []):
                param_name = param.get("name", "")
                if param_name:
                    optional_params[param_name] = param.get("default", "")

        category = api_info.get("category_name", "") if api_info else ""

        queries.append(
            ToolBenchQuery(
                query_id=query_id,
                query=query_text,
                tool_name=tool_name,
                api_name=api_name,
                category=category,
                required_params=required_params,
                optional_params=optional_params,
                api_description=api_info.get("api_description", "") if api_info else "",
            )
        )

    return queries


def load_toolbench_tool_calls(
    data_dir: str | Path,
    file_name: str = "toolllama_G123_dfs_train.json",
    limit: int | None = None,
) -> list[ToolCall]:
    """从对话跟踪中提取工具调用。"""
    data_dir = Path(data_dir)
    file_path = data_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"对话文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    tool_calls = []

    for conv_idx, conv in enumerate(conversations[:limit] if limit else conversations):
        query_id = conv.get("id", f"conv_{conv_idx}")
        conversations_list = conv.get("conversations", [])

        step = 0
        for entry in conversations_list:
            # 只处理assistant的响应，其中包含工具调用
            if entry.get("from") != "assistant":
                continue

            value = entry.get("value", "")

            # 解析Action和Action Input
            if "Action:" not in value or "Action Input:" not in value:
                continue

            try:
                # 提取Action（工具名称）
                action_start = value.find("Action:") + len("Action:")
                action_end = value.find("Action Input:")
                action = value[action_start:action_end].strip()

                # 提取Action Input（参数）
                action_input_start = value.find("Action Input:") + len("Action Input:")
                # 查找下一个换行或结束位置
                rest_value = value[action_input_start:]
                action_input_end = (
                    rest_value.find("\n\n") if "\n\n" in rest_value else len(rest_value)
                )
                action_input = rest_value[:action_input_end].strip()

                if not action or not action_input:
                    continue

                # 尝试解析JSON格式的参数
                try:
                    args = json.loads(action_input)
                    if not isinstance(args, dict):
                        args = {"value": args}
                except json.JSONDecodeError:
                    # 如果不是JSON，作为原始输入保存
                    args = {"raw_input": action_input}

                tool_calls.append(
                    ToolCall(
                        tool_name=action,
                        api_name=action,  # ToolBench中action即为API名称
                        arguments=args,
                        query_id=query_id,
                        step=step,
                    )
                )
                step += 1

            except (ValueError, IndexError) as e:
                # 跳过解析失败的条目
                continue

    return tool_calls


def extract_query_tool_pairs(
    data_dir: str | Path,
    subset: str = "G1",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """提取查询-工具对。"""
    data_dir = Path(data_dir)
    queries = load_toolbench_queries(data_dir, subset=subset, limit=limit)

    pairs = []
    for query in queries:
        # 合并必需和可选参数
        all_params = {**query.required_params, **query.optional_params}

        # 如果没有参数，使用查询文本作为输入
        if not all_params:
            all_params = {"query": query.query}

        pairs.append(
            {
                "id": query.query_id,
                "query": query.query,
                "tool_name": query.tool_name,
                "api_name": query.api_name or query.tool_name,
                "input_args": all_params,
                "category": query.category,
            }
        )

    return pairs


def format_for_benchmark(
    queries: list[ToolBenchQuery],
) -> list[dict[str, Any]]:
    """格式化为标准查询格式。"""
    formatted = []
    for query in queries:
        # 构建参数（仅包含实际的API参数）
        arguments = {**query.required_params, **query.optional_params}
        if not arguments:
            arguments = {"query": query.query}

        # input_args包含执行API所需的所有信息
        formatted.append(
            {
                "id": query.query_id,
                "query": query.query,
                "tool_name": query.tool_name,
                "input_args": {
                    "api_name": query.api_name or query.tool_name,
                    "category": query.category,
                    "arguments": arguments,
                    **arguments,
                },
            }
        )

    return formatted


def count_tool_usage(
    data_dir: str | Path,
    limit: int | None = None,
) -> dict[str, int]:
    """统计工具使用频率（降序）。"""
    tool_calls = load_toolbench_tool_calls(data_dir, limit=limit)
    usage = {}

    for call in tool_calls:
        tool = call.tool_name
        usage[tool] = usage.get(tool, 0) + 1

    return dict(sorted(usage.items(), key=lambda x: x[1], reverse=True))


def get_unique_tools(
    data_dir: str | Path,
    subsets: list[str] | None = None,
) -> list[dict[str, Any]]:
    """提取唯一工具定义（去重）。"""
    data_dir = Path(data_dir)
    if subsets is None:
        subsets = ["G1", "G2", "G3"]

    tools = []
    seen_keys = set()

    for subset in subsets:
        query_file = data_dir / "instruction" / f"{subset}_query.json"
        if not query_file.exists():
            continue

        with open(query_file, "r", encoding="utf-8") as f:
            raw_queries = json.load(f)

        for item in raw_queries:
            for api in item.get("api_list", []):
                tool_name = api.get("tool_name", "")
                api_name = api.get("api_name", "")

                # 使用(tool_name, api_name)作为唯一键
                tool_key = (tool_name, api_name)
                if tool_key in seen_keys:
                    continue

                seen_keys.add(tool_key)
                tools.append(
                    {
                        "tool_name": tool_name,
                        "api_name": api_name,
                        "category": api.get("category_name", ""),
                        "description": api.get("api_description", ""),
                        "method": api.get("method", "GET"),
                        "required_params": api.get("required_parameters", []),
                        "optional_params": api.get("optional_parameters", []),
                    }
                )

    return tools
