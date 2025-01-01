from dotenv import load_dotenv
from typing import List
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage, ToolMessage


import os
load_dotenv()

if __name__=="__main__":
    print(os.environ['LANGCHAIN_API_KEY'])