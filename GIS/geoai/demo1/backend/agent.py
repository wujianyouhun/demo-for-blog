from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tools import find_wind_sites

def create_agent():
    @tool
    def wind_tool(query: str) -> str:
        """Select wind farm candidate sites based on query."""
        return find_wind_sites(query)

    # 注意：这里需要设置 OpenAI API 密钥
    # 可以通过环境变量 OPENAI_API_KEY 设置，或者直接在这里设置
    # 例如：llm = ChatOpenAI(temperature=0, api_key="your_api_key")
    llm = ChatOpenAI(temperature=0)

    from langchain.agents import AgentExecutor
    from langchain.agents import create_react_agent
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template("""
    Answer the following questions as best you can. You have access to the following tools:

    wind_tool: Select wind farm candidate sites based on query.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [wind_tool]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Question: {input}
    Thought: Let me use the wind_tool to find wind farm sites.
    Action: wind_tool
    Action Input: {input}
    """)

    agent = create_react_agent(llm, [wind_tool], prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[wind_tool], verbose=True)
    return agent_executor
