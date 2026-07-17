from __future__ import annotations
import os
from .tools import find_wind_sites,requested_count

def run_agent(query: str,boundary_path: str,mode: str='local'):
    count=requested_count(query)
    explanation=f'本地确定性 Agent 已解析目标数量 {count}，在投影坐标系内生成候选格网并按地形代理评分排序。'
    if mode=='openai':
        if not os.getenv('OPENAI_API_KEY'):raise ValueError('OpenAI 模式需要 OPENAI_API_KEY；本地模式无需密钥')
        from langchain_openai import ChatOpenAI
        response=ChatOpenAI(temperature=0).invoke(f'用中文简短说明以下 GIS 任务如何执行，不要编造数据：{query}')
        explanation=str(response.content)
    elif mode!='local':
        raise ValueError(f'未知 Agent 模式: {mode}')
    return find_wind_sites(boundary_path,count),explanation
