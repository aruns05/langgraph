from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith import traceable

class Reflection(BaseModel):
    missing : str= Field(description="Critique of what is missing")
    superflous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question."""
    
    search_queries: List[str]=Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")
    answer: str = Field(description="300 word detailed answer to the question asked.")
    reflection: Reflection = Field(description="Your reflection of the initial answer.")
      
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""
    
    references: List[str] = Field(
        description="Citations motivating your updated answer"
    )