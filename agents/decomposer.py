"""LLM agent responsible to break down debate topic into sub-questions pair
These subquestion pairs will be sent to proposition and opposition teams
One question for the proposition; another one for the opposition"""

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import os
import json
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parent.parent.resolve()

# class QuestionPair(BaseModel):
#     """A pair of related subquestions for proposition and opposition"""
#     proposition_subquestion: str = Field(
#         description="A focused question for the proposition team to research and argue"
#     )
#     opposition_subquestion: str = Field(
#         description="A corresponding question for the opposition team, addressing the same aspect from their perspective"
#     )
#     aspect: str = Field(
#         description="The specific aspect or dimension of the debate this pair addresses (e.g. Economic impacts)"
#     )

# class DebateDecomposition(BaseModel):
#     """Complete decomposition of a debate topic into multiple question pairs"""
#     topic: str = Field(description="The original debate topic")
#     question_pairs: List[QuestionPair] = Field(
#         description="List of question pairs covering different aspects of the debate topic"
#     )

class QuestionDecomposer:
    """Decompose debate topics into structured sub-question pairs using DeepSeek Reasoner"""
    def __init__(self, api_key: str | None = None, model_name: str = "deepseek-reasoner"):
        # Load environment variables
        load_dotenv()
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise EnvironmentError("Deepseek API key must be provided")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model_name

    def decompose_topic(self, debate_topic: str, num_pairs: int = 5):
        """
        Decompose a debate topic into sub-question pairs for both teams
        
        Args:
            debate_topic: The main debate topic/motion to decompose
            num_pairs: Number of question pairs to generate (default: 5)
        
        Returns:
            DebateDecomposition object containing the question pairs"""
        system_prompt = self._create_system_prompt()

        user_prompt = f"""Debate topic: {debate_topic}
        number of subquestion pairs: {num_pairs}"""

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_object",
            }
        )
        
        # Extract the response
        content = response.choices[0].message.content
        reasoning_content = response.choices[0].message.reasoning_content
        
        # Store reasoning for debugging if needed
        self.last_reasoning = reasoning_content
        
        # Parse the structured output
        # decomposition = DebateDecomposition.model_validate_json(content)
        
        return content

    def _create_system_prompt(self) -> str:
        """Create the prompt for the LLM to decompose the debate topic"""
        prompt_filepath = PROJECT_ROOT_DIR / "prompts" / "system_instructions.json"
        with open(prompt_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print("system prompt json must be rendered as dict")
            raise
        return data.get("decomposer_agent", "")
    
    # DEBUGGING ONLY
    # def print_decomposition(self, decomposition: DebateDecomposition):
    #     """Pretty print the decomposition for easy viewing"""
    #     print(f"\n{'='*80}")
    #     print(f"DEBATE TOPIC: {decomposition.topic}")
    #     print(f"{'='*80}\n")

    #     for i, pair in enumerate(decomposition.question_pairs, 1):
    #         print(f"PAIR {i}: {pair.aspect}")
    #         print(f"{'-'*80}")
    #         print(f"📘 PROPOSITION: {pair.proposition_subquestion}")
    #         print(f"📕 OPPOSITION:  {pair.opposition_subquestion}")
    #         print()

if __name__ == "__main__":
    # Initialize the decomposer
    decomposer = QuestionDecomposer()
    
    # # Example debate topic
    topic = "This house believes that artificial intelligence will do more harm than good to humanity"
    
    # # Decompose the topic
    result = decomposer.decompose_topic(topic, num_pairs=5)

    if result is not None:
        print(result)
        print(json.loads(result))
    
    # # Display results
    # decomposer.print_decomposition(result)

    # print(decomposer._create_decomposition_prompt())
