from collections import defaultdict
import json
from typing import List
from langchain_core.messages import BaseMessage,ToolMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
import os

from schemas import AnswerQuestion, Reflection
from chains import parser
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

load_dotenv()

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search,max_results=5)

#Tool executor runs everything in parallel. It has a batch processing 
tool_executor= ToolExecutor([tavily_tool])


def execute_tools(state : List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation:AIMessage=state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)
    
    ids=[]
    tool_invocations=[]

    
    for parsed_calls in parsed_tool_calls:
        for query in parsed_calls["args"]["search_queries"]:
            tool_invocations.append(ToolInvocation(
                tool="tavily_search_results_json",
                tool_input=query
                ))
            ids.append(parsed_calls["id"])
    outputs = tool_executor.batch(tool_invocations)
    
    outputs_map =defaultdict(dict)
    for id_, output, invocation in zip(ids,outputs,tool_invocations):
        outputs_map[id_][invocation.tool_input]=output
        
    tool_messages = []
    for id_,  mapped_output in outputs_map.items():
        tool_messages.append(ToolMessage(content=json.dumps(mapped_output), tool_call_id=id_))
    
    return tool_messages
    #json.load(outputs[0][1]["content"])
    

if __name__=="__main__":
        
    human_message = HumanMessage(
        content="Write about AI-Powered SOC/ autonomous SOC problem, "
        "List startup that do that and raised capital"
    )
    
    answer = AnswerQuestion(
        answer="",
        reflection = Reflection(missing="", superflous=""),
        search_queries=[
            "AI-powered SOC startup funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-Powered SOC startups"
        ],
        #id="call_4ZKskq3p8SyrqXmf9toBmvUP"
    )
    
    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                   {
                       "name":AnswerQuestion.__qualname__,
                       "args":answer.dict(),
                       "id":""
                       #"id":"call_4ZKskq3p8SyrqXmf9toBmvUP",
                   } 
                ]
            )   
        ]
    )