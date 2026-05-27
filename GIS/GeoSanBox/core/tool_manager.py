import importlib
import inspect
from typing import Any, Callable, Dict, List, Optional, get_type_hints


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Dict[str, str],
        category: str = "general",
        returns: Optional[Dict[str, str]] = None,
    ) -> None:
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

        param_schema: Dict[str, Dict[str, Any]] = {}
        required_params: List[str] = []
        for param_name, param_desc in parameters.items():
            parts = param_desc.split(":", 1)
            ptype = parts[0].strip() if len(parts) > 0 else "any"
            pdesc = parts[1].strip() if len(parts) > 1 else ""
            param_schema[param_name] = {
                "type": ptype,
                "description": pdesc,
            }
            param = sig.parameters.get(param_name)
            if param is not None and param.default is inspect.Parameter.empty:
                required_params.append(param_name)

        self._tools[name] = {
            "name": name,
            "function": func,
            "description": description,
            "parameters": param_schema,
            "required": required_params,
            "category": category,
            "returns": returns or {"type": "dict", "description": "执行结果"},
        }

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._tools.get(name)

    def list_all(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
                "required": t["required"],
                "category": t["category"],
                "returns": t["returns"],
            }
            for t in self._tools.values()
        ]

    def list_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
                "required": t["required"],
                "category": t["category"],
                "returns": t["returns"],
            }
            for t in self._tools.values()
            if t["category"] == category
        ]

    def get_categories(self) -> List[str]:
        return sorted(set(t["category"] for t in self._tools.values()))

    def to_openai_schema(self) -> List[Dict[str, Any]]:
        schema = []
        for t in self._tools.values():
            properties = {}
            for pname, pinfo in t["parameters"].items():
                type_map = {
                    "str": "string",
                    "int": "integer",
                    "float": "number",
                    "bool": "boolean",
                    "list": "array",
                    "dict": "object",
                }
                properties[pname] = {
                    "type": type_map.get(pinfo["type"], "string"),
                    "description": pinfo["description"],
                }
            schema.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": t["required"],
                    },
                },
            })
        return schema

    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"status": "error", "error": f"Tool '{name}' not found"}

        for required_param in tool["required"]:
            if required_param not in kwargs:
                return {
                    "status": "error",
                    "error": f"Missing required parameter: '{required_param}'",
                }

        try:
            result = tool["function"](**kwargs)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def auto_discover(self, module_names: List[str]) -> None:
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "TOOLS_METADATA"):
                    for meta in module.TOOLS_METADATA:
                        func = getattr(module, meta["name"], None)
                        if func is not None:
                            self.register(
                                name=meta["name"],
                                func=func,
                                description=meta.get("description", ""),
                                parameters=meta.get("parameters", {}),
                                category=meta.get("category", "general"),
                                returns=meta.get("returns"),
                            )
            except ImportError as e:
                print(f"[ToolRegistry] Failed to import '{module_name}': {e}")

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# 全局注册表实例
registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return registry
