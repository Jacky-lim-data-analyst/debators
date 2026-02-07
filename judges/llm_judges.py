"""LLM as judge on which side win
The structured output: 
{
    "outcome": Enum
    "reason": str
}"""
import os
import sys

from typing import Type, TypeVar, Optional, Any, Dict, List, Literal
from pydantic import BaseModel, Field
from enum import Enum
from ollama import chat, ChatResponse
import json

# Ensure the project root is on `sys.path` so imports like `from config import ...`
# work when this script is run as a module or from different working directories.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#  ===== Outcome data ====
class OutcomeType(str, Enum):
    """Enum representing possible election outcomes"""
    GOVERNMENT_WINS = "government_wins"
    TIE = "tie"
    OPPOSITION_WINS = "opposition_wins"

class DebateOutcome(BaseModel):
    """Model representing an election outcome with justification"""
    outcome: OutcomeType = Field(
        description="Which debate side wins based on their arguments? Government or opposition?"
    )

    reason: str = Field(
        description="Justification for the outcome decision"
    )

    class ConfigDict:
        use_enum_values = False

def get_llm_evaluation(model_name: str, )

if __name__ == "__main__":
    outcome1 = DebateOutcome(
        outcome=OutcomeType.GOVERNMENT_WINS,
        reason="Government secured 52% of votes with strong performance in urban areas"
    )

    print(outcome1.model_dump())
