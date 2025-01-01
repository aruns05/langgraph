from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


import os
load_dotenv()

#Agent scratchpad is the history which needs to be plugged into the agent
react_prompt: PromptTemplate = hub.pull("hwchase17/react")

@tool
def triple(num: float) -> float:
    """_summary_

    Args:
        num (float): a number to triple

    Returns:
        float: the number tripled -> multiplied by 3
    """
    
    return float(num)*3

tools = [TavilySearchResults(max_results=1),triple]
llm = ChatOpenAI(
    model="gpt-4o"
)
#this is runnable
react_agent_runnable = create_react_agent(llm=llm,tools=tools,prompt=react_prompt)



