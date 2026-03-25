
from tools.spatial import buffer_analysis

def run_agent(query: str):
    if "缓冲区" in query:
        return buffer_analysis()
    return "这是一个基于本地模型的GIS智能体回答"
