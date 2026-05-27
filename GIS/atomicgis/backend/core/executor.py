from core.registry import get_tool
from core.state import SandboxState

state = SandboxState()

def execute_workflow(workflow):

    results = []

    for step in workflow["steps"]:
        tool_name = step["tool"]
        params = step.get("params", {})

        tool = get_tool(tool_name)

        result = tool.execute(state, **params)

        results.append({
            "tool": tool_name,
            "result": result
        })

    return {
        "status": "success",
        "results": results
    }