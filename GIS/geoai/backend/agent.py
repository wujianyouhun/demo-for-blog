from langchain_core.tools import tool
from langchain_core.agents import AgentExecutor, create_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from tools import find_wind_sites

def create_agent():
    @tool
    def wind_tool(query: str) -> str:
        """Select wind farm candidate sites based on query."""
        return find_wind_sites(query)

    llm = ChatOpenAI(temperature=0)

    prompt = hub.pull("hwchase17/react-agent")
    agent = create_agent(llm, [wind_tool], prompt=prompt)

    agent_executor = AgentExecutor(agent=agent, tools=[wind_tool], verbose=True)
    return agent_executor
