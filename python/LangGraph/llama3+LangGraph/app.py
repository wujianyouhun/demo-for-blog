from typing import TypedDict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# ===================== 全局状态 =====================
class AgentState(TypedDict):
    user_query: str
    query_type: str
    context: str
    response: str
    history: List[str]

# ===================== LLM 模型（Llama3）=====================
llm = ChatOllama(
    model="llama3",
    base_url="http://ollama:11434",  # Docker 内部地址
    temperature=0.3
)

# ===================== 智能体节点 =====================
def classify_agent(state: AgentState):
    prompt = f"""判断问题类型，只能返回 normal 或 tech：
用户问题：{state['user_query']}"""
    t = llm.invoke(prompt).content.strip()
    return {"query_type": t, "history": state["history"] + ["分类完成"]}

def normal_agent(state: AgentState):
    res = llm.invoke(state["user_query"]).content
    return {"response": res, "history": state["history"] + ["普通回答完成"]}

def retrieval_agent(state: AgentState):
    context = "LangGraph 是用于构建多智能体、循环、状态持久化的工作流框架。"
    return {"context": context, "history": state["history"] + ["检索完成"]}

def tech_agent(state: AgentState):
    prompt = f"根据资料回答：{state['context']}\n问题：{state['user_query']}"
    res = llm.invoke(prompt).content
    return {"response": res, "history": state["history"] + ["技术回答完成"]}

# ===================== 路由 =====================
def route_question(state: AgentState):
    return "retrieval" if state["query_type"] == "tech" else "normal"

# ===================== 构建工作流 =====================
workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_agent)
workflow.add_node("normal", normal_agent)
workflow.add_node("retrieval", retrieval_agent)
workflow.add_node("tech", tech_agent)

workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_question, {
    "normal": "normal",
    "retrieval": "retrieval"
})
workflow.add_edge("normal", END)
workflow.add_edge("retrieval", "tech")
workflow.add_edge("tech", END)

app = workflow.compile()

# ===================== FastAPI 接口 =====================
api = FastAPI(title="LangGraph + Llama3")

@api.get("/")
def index():
    return HTMLResponse(open("index.html", "r", encoding="utf-8").read())

@api.post("/run")
def run_workflow(query: str):
    try:
        result = app.invoke({
            "user_query": query,
            "history": ["开始执行"]
        })
        return {
            "query": result["user_query"],
            "type": result["query_type"],
            "response": result["response"],
            "history": result["history"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))