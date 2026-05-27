TOOLS = {}

def register(tool):
    TOOLS[tool.name] = tool

def get_tool(name):
    return TOOLS[name]