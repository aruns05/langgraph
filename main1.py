from typing import List
from typing_extensions import TypedDict
import re
from pydantic import BaseModel, Field

# langchain related libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# langgraph related libraries
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    question: str
    llm_output: str
    documents: list[str]
    cnt_retries: int

#Node — Question Scope Classifier
class QuestionScopeClass(BaseModel):
    """Scope of the question"""
    score: str = Field(
        description="Boolean value to check if question is about what, where or comparison. If yes -> 'Yes', else 'No'"
    )

def question_intent_classifier(state: AgentState):
    question = state["question"]
    state['cnt_retries']=0
    parser = JsonOutputParser(pydantic_object=QuestionScopeClass)
    output_format=parser.get_format_instructions()
    print(output_format)
    system = """You are a question classifier. Check if the question is about one of the following topics: 
        1. definition
        2. availability
        3. comparison
        If the question IS about these topics, respond with "Yes", otherwise respond with "No".
        
        Format output as: `{output_format}`
        """

    intent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}"),
        ]
    )

    llm = openAI()
    grader_llm = intent_prompt | llm | parser
    result = grader_llm.invoke({"question": question, 'output_format': output_format})
    print(f"to_proceed: {result['score']}")
    state["on_topic"] = result['score']
    return state

# router to enable conditional edges
def on_topic_router(state: AgentState):
    print('ontopic router ... ')
    on_topic = state["on_topic"]
    if on_topic.lower() == "yes":
        return "on_topic"
    return "off_topic"


def grade_answer(state: AgentState):
    answer= state['llm_output']
    print('grading....')
    pattern =r'do not know|sorry|apolog'
    is_answer = 'Yes' if not re.search(pattern, answer.lower()) else 'No'
    state['is_answer_ok'] = is_answer
    print(f"answer grade: {is_answer}")
    return state

def is_answer_router(state: AgentState):
    print('grading router ... ')
    is_answer = state["is_answer_ok"]
    if state['cnt_retries'] >2:  # max of 3 retries allowed (0 to 2)
        return "hit_max_retries"
    if is_answer.lower() == "yes":
        return "is_answer"
    return "is_not_answer"

def question_rephraser(state: AgentState):
    print('rephrasing ...')
    question = state['question']
    print(f"retrying: {state['cnt_retries']+1}")
    llm = openAI()

    template = """
        You are an expert in rephrasing English questions. \
        You hav been tasked to rephrase question from the Retail and Supply Chain domain. \
        While rephrasing, you may do the following:
        1. Extract keywords from the original question
        2. Expand or create abbreviations of the question as needed
        3. Understand the intent of the question
        4. Include the above information to generate a rephrased version of the original question.\
        Do not output anything else apart from the rephrased question.\
        
        Question: {question}
        """

    prompt = ChatPromptTemplate.from_template(
        template=template,
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    # print(result)
    state['question'] = result
    state['cnt_retries'] +=1
    return state


# retriever = get_retriever(config_vector_store) 

# def retrieve_docs(state: AgentState):
#     question = state["question"]
#     documents = retriever.invoke(input=question)
#     state["documents"] = documents
#     print(f"cnt of retrieved docs: {len(documents)}")
#     return state


def generate_answer(state: AgentState):
    question = state['question']
    context = [doc.page_content for doc in state['documents']]
    print('generating answer ... ')
    llm = openAI()

    template = """
        You are a Customer Support Chatbot aimed to answer the user's queries coming from Retail and Ecommerce industry.\
        Keep the tone conversational and professional.\
        Remember that abbreviations mentioned are related to these domains.\
        Answer the question strictly based on the context provided.\
        Avoid mentioning in the response that a context was referred.\
        Avoid using words like 'certainly" and "it looks like" in the generated response.\
        Do not output anything else apart from the answer.\
        
        {context}

        Question: {question}
        """

    prompt = ChatPromptTemplate.from_template(
        template=template,
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})
    print(f"result from generation: {result}")
    state['llm_output'] = result
    return state

def get_default_reply(state:AgentState):
    print('get the default answer ...')
    state['llm_output'] = 'I do not have an answer.'
    return state


workflow = StateGraph(AgentState)

# Add the Nodes
workflow.add_node('intent_classifier', question_intent_classifier)
#workflow.add_node('retrieve_docs', retrieve_docs)
workflow.add_node('generate_answer', generate_answer)
workflow.add_node('grade_answer', grade_answer)
workflow.add_node('question_rephraser', question_rephraser)
workflow.add_node('default_reply', get_default_reply)

# Add the Edges including the Conditional Edges
workflow.add_edge('intent_classifier', START)
workflow.add_conditional_edges(
    'intent_classifier', on_topic_router, 
    {
        'on_topic': 'retrieve_docs', 
        'off_topic': 'default_reply'
    }
)
workflow.add_edge('retrieve_docs', 'generate_answer')
workflow.add_edge('generate_answer', 'grade_answer')
workflow.add_conditional_edges(
    'grade_answer', is_answer_router,
    {
        'is_answer':END, 
        'is_not_answer':'question_rephraser', 
        'hit_max_retries':'default_reply'
    }
)
workflow.add_edge('question_rephraser', 'retrieve_docs')
workflow.add_edge('default_reply', END)

# compile the workflow
app = workflow.compile()


query = "Capital of India"
response = app.invoke(input={"question": query})

print(result['llm_output'])