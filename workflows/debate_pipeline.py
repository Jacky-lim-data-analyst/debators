"""Main pipeline to orchestrate the debate workflow
1. Decomepose topic into question pair
2. Generate responses from both teams
3. Store results in MongoDB"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

# ensure the project root is on `sys.path`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
from agents.decomposer import QuestionDecomposer
from agents.debate_agents import RotatingFallbackDebator, SoloDebator
from database.document_indexer import make_pair_doc
from pymongo import AsyncMongoClient
from config import MONGODB_PORT

# configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- Main debate pipeline orchestration ---
class DebatePipeline:
    """Orchestrates the complete debate workflow"""

    def __init__(
        self,
        deepseek_api_key: Optional[str] = None,
        mongodb_uri: Optional[str] = None,
        database_name: str = "debate_db",
        collection_name: str = "debate_pairs"
    ):
        """Initialize the debate pipeline
        Args:
            deepseek_api_key: API key for DeepSeek (optional, can use env var)
            mongodb_uri: MongoDB connection URI (defaults to local instance)
            database_name: Name of the MongoDB database
            collection_name: Name of the MongoDB collection"""
        # initializes components
        self.decomposer = QuestionDecomposer(api_key=deepseek_api_key)
        self.proposition_agent = RotatingFallbackDebator(temperature=0.1, max_tokens=4096)
        self.opposition_agent = SoloDebator(api_key=deepseek_api_key, temperature=0.1, max_tokens=4096)

        # mongodb configuration
        self.mongodb_uri = mongodb_uri or f"mongodb://127.0.0.1:{MONGODB_PORT}"
        self.database_name = database_name
        self.collection_name = collection_name

        logger.info("Debate pipeline initialized!")

    def _create_agent_prompt(
        self,
        topic: str,
        question: str,
        role: str,
        system_instruction: Optional[str] = None 
    ) -> List[Dict[str, str]]:
        """Create a prompt for the debate agents
        Args:
            topic: The debate topic
            question: The specific question to answer
            role: Either 'proposition' or 'opposition'
            system_instructions: Optional custom system instructions
            
        Returns:
            List of message dictionaries for the agent"""
        if system_instruction is None:
            system_instruction_path = PROJECT_ROOT / "prompts" / "system_instructions.json"
            with open(system_instruction_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            system_instruction = data.get("debate_agent", "")

        if not system_instruction: 
            logger.warning("The system instruction can't be read from external file")

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"""Debate Topic: {topic}

Question: {question}

Please provide a comprehensive answer to this question from the {role} perspective"""} 
        ]
    
    @staticmethod
    async def _save_json(data, path):
        await asyncio.to_thread(
            lambda: open(path, 'w', encoding='utf-8').write(json.dumps(data, ensure_ascii=False))
        )
    
    async def process_question_pair(
        self,
        topic: str,
        pair: Dict,
        pair_index: int
    ) -> Optional[Dict]:
        """Process a single question pair through both debate agents
        
        Args:
            topic: The debate topic
            pair: Question pair dictionary with aspect, proposition_subquestion, opposition_subquestion
            pair_index: Index of the pair for logging
            
        Returns:
            Complete document ready for MongoDB insertion, or None if failed
        """
        aspect = pair.get('aspect', f'Aspect {pair_index}')
        q_pos = pair.get('proposition_subquestion', '')
        q_neg = pair.get('opposition_subquestion', '')

        logger.info(f"Processing pair {pair_index + 1}: {aspect}")

        # generate proposition response
        logger.info(f" Generating proposition response...")
        prop_prompt = self._create_agent_prompt(topic, q_pos, 'proposition')
        try:
            r_pos = self.proposition_agent.invoke({"messages": prop_prompt})
            if not r_pos:
                logger.warning(f"Proposition agent returned empty response for pair {pair_index + 1}")
                r_pos = "[No response generated]"
        except Exception as e:
            logger.error(f"Error generating proposition response: {e}")
            r_pos = f"[Error: {str(e)}]"

        # generate opposition response
        logger.info(f"Generating opposition response...")
        opp_prompt = self._create_agent_prompt(topic, q_neg, 'opposition')
        try:
            r_neg = self.opposition_agent.invoke(opp_prompt)
            if not r_neg:
                logger.warning(f"  Opposition agent returned empty response for pair {pair_index + 1}")
                r_neg = "[No response generated]"
        except Exception as e:
            logger.error(f"Error generating opposition response: {e}")
            r_neg = f"[Error: {str(e)}]"

        # create document
        doc = make_pair_doc(
            topic=topic,
            q_pos=q_pos,
            r_pos=r_pos,
            q_neg=q_neg,
            r_neg=r_neg,
            model_pos="rotating_fallback_chain",
            model_neg="deepseek-chat"
        )

        # add aspect information
        doc['aspect'] = aspect
        doc['pair_index'] = pair_index

        logger.info(f"Pair {pair_index + 1} processed successfully")
        return doc
    
    async def store_results(self, documents: List[Dict]) -> bool:
        """Store debate results in MongoDB
        Args:
            documents: List of debate pair documents to store
            
        Returns:
            True if successful, False otherwise"""
        client = None
        try:
            client = AsyncMongoClient(self.mongodb_uri)
            database = client[self.database_name]
            collection = database[self.collection_name]

            # insert all docs
            if documents:
                result = await collection.insert_many(documents)
                logger.info(f"Successfully stored {len(result.inserted_ids)} documents in MongoDB")
                return True
            else:
                logger.warning("No documents to store")
                return False
        except Exception as e:
            logger.error(f"Error storing results in MongoDB: {e}")
            return False
        finally:
            if client:
                await client.close()

    async def run(
        self, 
        topic: str,
        num_pairs: int = 5,
        save_to_json: bool = True,
        json_output_path: Optional[str] = None
    ) -> Dict:
        """Run the complete debate pipeline
        
        Args:
            topic: The debate topic to analyze
            num_pairs: Number of question pairs to generate
            save_to_json: Whether to save results to JSON file
            json_output_path: Path for JSON output (optional)
            
        Returns:
            Dictionary containing all results
        """
        logger.info("="*80)
        logger.info(f"Starting debate pipeline for topic: {topic}")
        logger.info("="*80)

        # step 1: Decompose topic
        logger.info(f"\nDecomposing topic into {num_pairs} question pairs...")
        try:
            decomposition_json = self.decomposer.decompose_topic(topic, num_pairs)
            decomposition = json.loads(decomposition_json)
            logger.info(f"Successfully generated {len(decomposition.get('question_pairs', []))} question pairs")
        except Exception as e:
            logger.error(f"Error decomposing topic: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "decomposition"
            }
        
        # step 2: Process each question pair
        logger.info(f"\nStep 2: Processing question pairs with debate agents...")
        question_pairs = decomposition.get('question_pairs', [])
        documents = []

        for i, pair in enumerate(question_pairs):
            try:
                doc = await self.process_question_pair(topic, pair, i)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing pair {i + 1}: {e}")
                continue

        if not documents:
            logger.error("No documents were generated")
            return {
                "success": False,
                "error": "No documents generated",
                "stage": "debate_generation"
            }
        
        # step 3: store in mongodb
        logger.info(f"\nStep 3: Storing {len(documents)} results in MongoDB...")
        storage_success = await self.store_results(documents)

        # save to json (optional)
        if save_to_json:
            if json_output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_output_path = f"debate_results_{timestamp}.json"

            try:
                output_data = {
                    "topic": topic,
                    "generated_at": datetime.now().isoformat(),
                    "num_pairs": len(documents),
                    "debate_results": documents
                }

                await self._save_json(output_data, json_output_path)

                logger.info(f"Results saved to: {json_output_path}")

            except Exception as e:
                logger.error(f"Error saving JSON output: {e}")

        # return summary
        logger.info("Pipeline completed successfully")

        return {
            "success": True,
            "topic": topic,
            "num_pairs_generated": len(documents),
            "stored_in_mongodb": storage_success,
            "json_output": json_output_path if save_to_json else None,
            "documents": documents
        }
        
async def main():
    """Usage of the debate pipeline"""
    # initialize pipeline
    pipeline = DebatePipeline(
        database_name="debate_db",
        collection_name="first_debate"
    )

    # example debate topic
    topic = "The house believes that social media algorithms should be regulated as public utilities."

    # run the pipeline
    results = await pipeline.run(
        topic=topic,
        num_pairs=5,
        save_to_json=True
    )

    # print summary
    if results.get('success'):
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"Topic: {results['topic']}")
        print(f"Question pairs processed: {results['num_pairs_generated']}")
        print(f"Stored in MongoDB: {results['stored_in_mongodb']}")
        if results.get('json_output'):
            print(f"JSON output saved to: {results['json_output']}")
        print("="*80)
    else:
        print(f"\nPipeline failed at stage: {results.get('stage')}")
        print(f"Error: {results.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())
