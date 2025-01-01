import datetime

from dotenv import load_dotenv
import os
import sys

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import wrappers, traceable

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)

from schemas import InitialAnswer
parser_pydantic = PydanticToolsParser(tools=[InitialAnswer])

llm = ChatOpenAI(
    model="gpt-4o"
)

player_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an expert cricket researcher.
            1. {first_instruction}
            2. Reviewx your answer . Be severe to maximize improvement.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format. "    
        ),
        (
            "human",
            "Who is a better cricketer between {player1} and {player2} "    
        ),
    ]
)

first_answer_prompt_template = player_prompt_template.partial(
    first_instruction="Provide a detailed 200 word answer with statistics. Donot exceed 200 words"
)

first_answer = first_answer_prompt_template | llm.bind_tools(
    tools=[InitialAnswer], tool_choice="InitialAnswer"
)

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    player1 = "Virat Kohli"
    player2 = "Rohit Sharma"
    
    #human_message = HumanMessage(content="""Who is a better cricketer between {player1} and {player2}?""")
    human_message = HumanMessage(content="")
    chain =(
        first_answer_prompt_template 
        | llm.bind_tools(tools=[InitialAnswer], tool_choice="InitialAnswer")
        | parser_pydantic
    )
    
    response = chain.invoke(input={"messages":[human_message],"player1":player1,"player2":player2})
    print(response)