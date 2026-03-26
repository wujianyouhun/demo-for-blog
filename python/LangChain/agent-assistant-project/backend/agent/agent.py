
from tools.math_tool import calc
from tools.search_tool import search

def run_agent(query: str):
    if "计算" in query:
        return calc(query)
    if "搜索" in query:
        return search(query)
    return "这是一个基础智能助手"
