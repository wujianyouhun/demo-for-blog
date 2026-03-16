from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from tools.sql_tool import run_postgis_query

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

tools = [run_postgis_query]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
