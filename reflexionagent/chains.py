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

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(
    model="gpt-4o"
)
parser = JsonOutputToolsParser(return_id= True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an expert researcher.
                Current time {curr_time}
            1. {first_instruction}
            2. Review and critique your answer . Be severe to maximise improvement.
            3. Recommend search queries to research information and improve your answer . 
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format. "
        ),
    ]
).partial(
    curr_time=datetime.datetime.now().isoformat()
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed 300 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)


revise_instructions ="""
    Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer
    - You must include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not count to word limit)
        - [1] https://example.com
        - [2] https://example.com
    - You should remove the previous critique to remove superflous information to make sure it is not more than 300 words.     
"""

revisor = actor_prompt_template.partial(first_instruction= revise_instructions) | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)


if __name__=='__main__':
    sys.path.append(os.getcwd())
    human_message = HumanMessage(content="Write about Ai-Powered SOC / autonomous soc problem domain,"
                                 "list startups that do that and raised capital")
    chain =(
        first_responder_prompt_template 
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )
    
    response = chain.invoke(input={"messages":[human_message]})
    print(response)