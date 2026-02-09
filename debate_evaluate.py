"""
Debate evaluation system 
Queries debates from MongoDB, evaluates with LLM, and saves results in Markdown and JSON"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from pymongo import AsyncMongoClient, ASCENDING
import sys
import os

# ensure the project root is on `sys.path`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MONGODB_PORT, SERVER_IP_ADDRESS, OLLAMA_PORT
from agents.llm_judges import OllamaEvaluator, DebateOutcome, OutcomeType

class DebateEvaluator:
    """Evaluate debate pairs using LLM and stores results"""

    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        database_name: str = "debate_db",
        collection_name: str = "first_debate",
        evaluator_model: str = "qwen3:4b",
        output_dir: str = "./evaluation_results"
    ):
        self.mongodb_uri = mongodb_uri or f"mongodb://127.0.0.1:{MONGODB_PORT}"
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM evaluator
        self.evaluator = OllamaEvaluator(
            model_name=evaluator_model,
            temperature=0.2,
            think=True
        )

    async def connect(self):
        """Establish MongoDB connection"""
        self.client = AsyncMongoClient(self.mongodb_uri)
        return self.client[self.database_name][self.collection_name]
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            await self.client.close()

    async def get_debates_by_topic(self, topic: str) -> List[Dict]:
        """Retrieve all debate pairs for a specific topic"""
        collection = await self.connect()
        try:
            cursor = collection.find({"topic": topic}).sort("pair_index", ASCENDING)
            debates = await cursor.to_list(length=None)  # all debate pairs
            return debates
        finally:
            await self.disconnect()

    def create_evaluation_prompt(self, debate_pair: Dict) -> str:
        """Create a prompt for the LLM evaluator"""
        topic = debate_pair.get("topic", "Unknown topic")
        prop_question = debate_pair.get("questions", {}).get("positive", "")
        if not prop_question:
            print("Warning: proposition question is empty string!!!")
        opp_question = debate_pair.get("questions", {}).get("negative", "")
        if not opp_question:
            print("Warning: opposition question is empty string!!!")

        prop_response = debate_pair.get("responses", {}).get("positive", "")
        if not prop_response:
            print("Warning: proposition response is empty string!!!")
        opp_response = debate_pair.get("responses", {}).get("negative", "")
        if not opp_response:
            print("Warning: opposition response is empty string!!!")

        prompt = f"""DEBATE TOPIC: {topic}

SUBQUESTION FOR PROPOSITION: {prop_question}

PROPOSITION (Government) RESPONSE:
{prop_response}

SUBQUESTION FOR OPPOSITION: {opp_question}

OPPOSITION RESPONSE: 
{opp_response}

Decide which side wins this particular exchange, or it is a tie. Provide your justification"""
        
        return prompt
    
    def evaluate_debate_pair(self, debate_pair: Dict) -> Dict:
        """Evaluate a single debate pair using LLM"""
        prompt = self.create_evaluation_prompt(debate_pair)

        try:
            result: DebateOutcome = self.evaluator.generate(
                prompt=prompt,
                response_model=DebateOutcome,
                max_retries=3
            )

            return {
                "outcome": result.outcome.value,
                "reason": result.reason,
                "evaluated_at": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "outcome": "Error",
                "reason": f"Evaluation failed: {str(e)}",
                "evaluated_at": datetime.now().isoformat()
            }
        
    async def evaluate_topic(self, topic: str, update_db: bool = True) -> Dict:
        """Evaluate all debate pairs for a specific topic
        
        Args:
            topic: The debate topic to evaluate
            update_db: Whether to update MongoDB with results
            
        Returns:
            Dictionary containing evaluation results and statistics"""
        print(f"\n{'='*80}")
        print(f"Evaluating debates for topic: {topic}")
        print(f"{'='*80}\n")

        # fetch debates
        debates = await self.get_debates_by_topic(topic)

        if not debates:
            print(f"No debates found for topic: {topic}")
            return {"error": "No debates found"}
        
        print(f"Found {len(debates)} debate pairs to evaluate\n")

        # evaluate each pair
        results = []
        stats = {
            "proposition_wins": 0,
            "opposition_wins": 0,
            "ties": 0,
            "errors": 0
        }

        for idx, debate_pair in enumerate(debates, 1):
            print(f"Evaluating pair {idx}/{len(debates)}...", end=" ")

            evaluation = self.evaluate_debate_pair(debate_pair)

            # update stats
            outcome = evaluation.get("outcome")
            if outcome == OutcomeType.PROPOSITION_WINS.value:
                stats["proposition_wins"] += 1
                print("Proposition wins")
            elif outcome == OutcomeType.OPPOSITION_WINS.value:
                stats["opposition_wins"] += 1
                print("Opposition wins")
            elif outcome == OutcomeType.TIE:
                stats["ties"] += 1
                print("Tie")
            else:
                stats["errors"] += 1
                print("Error")

            # Extract responses for inclusion in report
            responses = debate_pair.get("responses", {})

            # prepare result entry
            result_entry = {
                "pair_index": debate_pair.get("pair_index"),
                "questions": debate_pair.get("questions"),
                "proposition_response": responses.get("positive", "No response"),
                "opposition_response": responses.get("negative", "No response"),
                "evaluation": evaluation,
                "debate_id": str(debate_pair.get("_id"))
            }

            results.append(result_entry)

            # update database if requested
            if update_db and debate_pair.get("_id"):
                await self.update_debate_evaluation(
                    debate_id=debate_pair["_id"],
                    evaluation=evaluation
                )

        # compile final results
        evaluation_results = {
            "topic": topic,
            "evaluated_at": datetime.now().isoformat(),
            "total_pairs": len(debates),
            "statistics": stats,
            "results": results
        }

        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Pairs: {len(debates)}")
        print(f"Proposition Wins: {stats['proposition_wins']}")
        print(f"Opposition Wins: {stats['opposition_wins']}")
        print(f"Ties: {stats['ties']}")
        print(f"Errors: {stats['errors']}")
        print(f"{'='*80}\n")
        
        return evaluation_results

    async def update_debate_evaluation(self, debate_id, evaluation: Dict):
        """Update a debate doc with evaluation result"""
        collection = await self.connect()
        try:
            await collection.update_one(
                {"_id": debate_id},
                {"$set": {"evaluation": evaluation}}
            )
        finally:
            await self.disconnect()

    def save_markdown_report(self, results: Dict, filename: Optional[str] = None):
        """save evaluation results as markdown report"""
        