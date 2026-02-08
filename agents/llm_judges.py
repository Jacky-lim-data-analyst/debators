"""LLM as judge on which side win
The structured output: 
{
    "outcome": Enum
    "reason": str
}"""
import os
import sys

from typing import Type, TypeVar, Any
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from ollama import ChatResponse, Client
import json

# Ensure the project root is on `sys.path` so imports like `from config import ...`
# work when this script is run as a module or from different working directories.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SERVER_IP_ADDRESS, OLLAMA_PORT

T = TypeVar('T', bound=BaseModel)

#  ===== Outcome data ====
class OutcomeType(str, Enum):
    """Enum representing possible election outcomes"""
    PROPOSITION_WINS = "government_wins"
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


# === Ollama evaluator with structured output ===
class OllamaEvaluator:
    """LLM as evaluator of government and opposition arguments"""
    def __init__(self,
                 host_ip: str | None = None,
                 port: int | None = None,
                 model_name: str = "qwen3:4b",
                 temperature: float = 0.0,
                 think: bool = True,
                 system_prompt: str | None = None,
                 **default_options: Any):
        """Initialize model as evaluator
        Args:
            model: The Ollama model to use
            temperature: Default temperature for generation (0-1)
            think: Thinking level ('low', 'medium', 'high')
            **default_options: Additional default options for Ollama"""
        if not host_ip:
            self.host_ip = SERVER_IP_ADDRESS
        if not port:
            self.port = OLLAMA_PORT
        self.model_name = model_name
        self.default_temperature = temperature
        self.default_think = think
        self.default_options = default_options
        if not system_prompt:
            # load from json file
            # for now, empty string
            self.system_prompt = ""

        connection_string = "http://" + self.host_ip + ":" + str(self.port)
        self.client = Client(host=connection_string)

    def generate(
        self, 
        prompt: str,
        response_model: Type[T],
        max_retries: int = 3,
        **options: Any
    ) -> T:
        """Generate the structured response: DebateOutcome
        Args:
            prompt: The user prompt
            response_model: Pydantic model class to validate against
            max_retries: Number of retries on validation failure
            system_prompt: Optional system prompt
            **options: Additional options for this specific call
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            ValidationError: If response cannot be validated after max_retries
            Exception: If Ollama API fails"""
        # Build messages
        messages = [{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': prompt}]

        # merge options
        call_options = {
            'temperature': self.default_temperature,
            **self.default_options,
            **options
        }

        # prepare call parameters
        call_params = {
            'model': self.model_name,
            'messages': messages,
            'format': response_model.model_json_schema(),
            'options': call_options
        }

        # add think parameter
        if self.default_think:
            call_params['think'] = self.default_think

        # attempt generation with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                response: ChatResponse = self.client.chat(**call_params)

                # validate and parse response
                validated_response = response_model.model_validate_json(response.message.content)
                return validated_response
            except ValidationError as e:
                last_error = e
                if attempt < max_retries - 1:
                    continue
            except Exception as e:
                raise Exception(f"Ollama API error: {str(e)}")
            
        # if all retries failed
        raise ValidationError(
            f"Failed to generate valid response after {max_retries} attempts"
            f"Last Error: {last_error}"
        )

if __name__ == "__main__":
    # outcome1 = DebateOutcome(
    #     outcome=OutcomeType.GOVERNMENT_WINS,
    #     reason="Government secured 52% of votes with strong performance in urban areas"
    # )

    # print(outcome1.model_dump())
    # model = OllamaEvaluator()

    # election_result = model.generate(
    #     prompt="The recent election results show that the incumbent party won 52% of votes "
    #            "with particularly strong support in urban areas. Analyze this outcome.",
    #     response_model=DebateOutcome,
    #     temperature=0.2
    # )

    # print(f"Outcome: {election_result.outcome.value}")
    # print(f"Reason: {election_result.reason}")

    from pathlib import Path

    print(Path(__file__).parent.parent.resolve() / "prompts" / "system_instructions.json")
