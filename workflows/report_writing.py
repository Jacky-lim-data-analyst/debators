"""LangGraph research agent

A comprehensive research agent using LangGraph for workflow orchestration
This agent queries a local vector store, fetch debate chunks, analyze results 
and generates detailed reports"""

from typing import TypedDict, Annotated, Sequence, Literal
from datetime import datetime
import operator
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from ..utils.reports import save_llm_response

# =========== State Definition ======
class ResearchState(TypedDict):
    """State for the research workflow"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    search_results: list[dict]
    research_findings: str
    iterations: int
    max_iterations: int
    final_report: str

# ==== Node function ====
def initialize_research(state: ResearchState) -> dict:
    """Initialize research process and extract query"""
    messages = state['messages']

    # extract user query from last human message
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    return {
        "query": user_query,
        "iterations": 0,
        "max_iterations": state.get("max_iterations", 3),
        "should_continue": True,
        "search_results": [],
        "research findings": ""
    }

def search_vector_store(state: ResearchState) -> dict:
    """Query the local vector store for relevant info"""
    query = state['query']
    current_findings = state.get('research_findings', '')

    # perform vector search
    