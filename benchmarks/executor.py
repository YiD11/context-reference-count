"""
ToolBench API 执行器。
"""

import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOOLBENCH_SERVICE_URL = "http://8.130.32.149:8080/rapidapi"


@dataclass
class APIExecutionResult:
    success: bool
    output: Any
    error: str | None = None
    latency: float = 0.0
    status_code: int = 0


def standardize_name(name: str) -> str:
    result = name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    return re.sub(r'[^a-z0-9_]', '', result)


def substitute_url_params(url: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """替换 URL 中的路径参数占位符，返回 (处理后URL, 剩余查询参数)。"""
    remaining = dict(arguments)
    for placeholder in re.findall(r'\{(\w+)\}', url):
        for key, val in list(remaining.items()):
            if key.lower() == placeholder.lower():
                url = url.replace(f"{{{placeholder}}}", str(val))
                del remaining[key]
                break
        else:
            raise ValueError(f"URL 参数 '{placeholder}' 未提供，当前参数: {list(arguments.keys())}")
    return url, remaining


class ToolBenchAPIExecutor:
    """直接调用 RapidAPI 或通过 ToolBench 服务器代理。"""

    def __init__(
        self,
        tool_dir: str | Path,
        rapid_api_key: str | None = None,
        toolbench_key: str | None = None,
        use_toolbench_server: bool = False,
    ):
        self.tool_dir = Path(tool_dir)
        self.api_key = rapid_api_key or os.environ["RAPIDAPI_KEY"] if not use_toolbench_server else ""
        self.toolbench_key = toolbench_key or os.environ.get("TOOLBENCH_KEY", "")
        self.use_toolbench_server = use_toolbench_server
        self.tool_cache: dict[str, dict[str, Any]] = {}

        if use_toolbench_server and not self.toolbench_key:
            raise ValueError("use_toolbench_server=True 但未设置 TOOLBENCH_KEY")

    def load_tool_definition(self, tool_name: str, category: str) -> dict[str, Any]:
        cache_key = f"{category}:{tool_name}"
        if cache_key in self.tool_cache:
            return self.tool_cache[cache_key]

        search_name = standardize_name(tool_name)
        cat_dir = self.tool_dir / category

        for tool_file in cat_dir.glob("*.json"):
            with open(tool_file, encoding="utf-8") as f:
                tool_def = json.load(f)
            def_name = standardize_name(tool_def.get("tool_name", ""))
            if def_name == search_name or tool_file.stem == search_name:
                tool_def["_category"] = category
                tool_def["_standardized_name"] = tool_file.stem
                self.tool_cache[cache_key] = tool_def
                return tool_def

        raise FileNotFoundError(f"工具定义未找到: {tool_name} in {category}")

    def execute_api(
        self,
        tool_name: str,
        api_name: str,
        arguments: dict[str, Any],
        category: str,
    ) -> APIExecutionResult:
        import requests

        start_time = time.time()
        tool_def = self.load_tool_definition(tool_name, category)

        search_api = standardize_name(api_name)
        api_def = None
        for api in tool_def["api_list"]:
            if standardize_name(api.get("name", "")) == search_api:
                api_def = api
                break

        if api_def is None:
            raise KeyError(f"API '{api_name}' 不存在于工具 '{tool_name}'")

        if self.use_toolbench_server:
            return self._execute_via_server(tool_def, api_def, arguments, start_time)
        return self._execute_direct(tool_def, api_def, arguments, start_time)

    def _execute_via_server(
        self,
        tool_def: dict,
        api_def: dict,
        arguments: dict,
        start_time: float,
    ) -> APIExecutionResult:
        import requests

        payload = {
            "category": tool_def["_category"],
            "tool_name": tool_def["_standardized_name"],
            "api_name": api_def["name"],
            "tool_input": json.dumps(arguments),
            "strip": "truncate",
            "toolbench_key": self.toolbench_key,
        }

        time.sleep(2)  # Rate limit
        response = requests.post(
            TOOLBENCH_SERVICE_URL,
            json=payload,
            headers={"toolbench_key": self.toolbench_key},
            timeout=15,
        )
        response.raise_for_status()

        result = response.json()
        latency = time.time() - start_time
        error_msg = result.get("error", "")

        return APIExecutionResult(
            success=not bool(error_msg),
            output=result.get("response"),
            error=error_msg or None,
            latency=latency,
            status_code=response.status_code,
        )

    def _execute_direct(
        self,
        tool_def: dict,
        api_def: dict,
        arguments: dict,
        start_time: float,
    ) -> APIExecutionResult:
        import requests

        url = api_def["url"]
        method = api_def.get("method", "GET").upper()
        host = tool_def["host"]

        # 合并默认参数
        merged = {}
        for param in api_def.get("required_parameters", []) + api_def.get("optional_parameters", []):
            if param.get("default") is not None:
                merged[param["name"]] = param["default"]
        merged.update(arguments)

        url, query_params = substitute_url_params(url, merged)

        headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": host,
        }

        if method == "GET":
            response = requests.get(url, headers=headers, params=query_params, timeout=10)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=query_params, timeout=10)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=query_params, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, params=query_params, timeout=10)
        else:
            raise ValueError(f"不支持的 HTTP 方法: {method}")

        latency = time.time() - start_time
        output = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        success = 200 <= response.status_code < 300

        return APIExecutionResult(
            success=success,
            output=output,
            error=None if success else f"HTTP {response.status_code}",
            latency=latency,
            status_code=response.status_code,
        )


class DynamicAPIExecutor:
    """动态导入并执行 api.py 中的函数（ToolBench 官方方式）。"""

    def __init__(self, tool_dir: str | Path, rapid_api_key: str | None = None):
        self.tool_dir = Path(tool_dir)
        self.api_key = rapid_api_key or os.environ["RAPIDAPI_KEY"]

    def execute_api(
        self,
        tool_name: str,
        api_name: str,
        arguments: dict[str, Any],
        category: str,
    ) -> APIExecutionResult:
        start_time = time.time()

        # 定位 api.py
        search_name = standardize_name(tool_name)
        cat_dir = self.tool_dir / category

        tool_path = None
        for p in cat_dir.iterdir():
            if p.is_dir() and standardize_name(p.name) == search_name:
                tool_path = p
                break

        if tool_path is None:
            raise FileNotFoundError(f"工具目录未找到: {tool_name} in {category}")

        api_file = tool_path / "api.py"
        if not api_file.exists():
            raise FileNotFoundError(f"api.py 不存在: {api_file}")

        # 动态导入
        module_name = f"_toolbench_{category}_{tool_path.name}"
        spec = importlib.util.spec_from_file_location(module_name, api_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # 查找函数
        func_name = standardize_name(api_name)
        if not hasattr(module, func_name):
            available = [n for n in dir(module) if not n.startswith("_") and callable(getattr(module, n))]
            raise AttributeError(f"函数 '{func_name}' 不存在，可用: {available}")

        api_func = getattr(module, func_name)

        # 执行
        call_args = dict(arguments)
        call_args["toolbench_rapidapi_key"] = self.api_key
        result = api_func(**call_args)

        latency = time.time() - start_time
        return APIExecutionResult(success=True, output=result, latency=latency)
