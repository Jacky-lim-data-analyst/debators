"""Vector database utilities for indexing and retrieving debate arguments using ChromaDB"""
import asyncio 
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import sys
import os

# ensure the project root is on `sys.path` 
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MONGODB_PORT

class DebatePosition(str, Enum):
    """Enum for debate positions"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOTH = "both"

class DebateVectorDB:
    """Vector database manager for debate arguments using ChromaDB"""

    def __init__(
        self,
        collection_name: str,
        db_host: str,
        db_port: int = 9000,
        embedding_model: str = "embeddinggemma:latest"):
        """"""
        self.client = chromadb.HttpClient(host=db_host, port=db_port, ssl=False)
        embeddings = OllamaEmbeddingFunction(
            url="http://192.168.0.162:11434",
            model_name=embedding_model
        )
        # get collection 
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embeddings,
            metadata={"description": "Debate arguments from proposition and opposition"}
        )

    def _generate_document_id(self, pair_hash: str, position: str, arg_type: str) -> str:
        """
        Generate unique document ID
        
        Args:
            pair_hash: Hash of the debate pair
            position: 'positive' or 'negative'
            arg_type: 'question' or 'response'
        
        Returns:
            Unique document ID
        """
        return f"{pair_hash}_{position}_{arg_type}"
    
    def _create_document_text(self, topic: str, question: str, response: str, position: str) -> str:
        """
        Create formatted document text combining topic, question, and response
        
        Args:
            topic: Debate topic
            question: Question asked
            response: Response/argument given
            position: Position taken (positive/negative)
        
        Returns:
            Formatted document text
        """
        position_label = "Proposition" if position == "positive" else "Opposition"
        return f"""Topic: {topic}
Position: {position_label}
Question: {question}
Argument: {response}"""
    
    def index_debate_pair(self, debate_data: Dict) -> Dict[str, int]:
        """
        Index a single debate pair (both positive and negative arguments)
        
        Args:
            debate_data: Dictionary containing debate pair data
        
        Returns:
            Dictionary with count of indexed documents
        """
        topic = debate_data.get("topic", "Unknown topic")
        pair_hash = debate_data.get("pair_hash", "")
        aspect = debate_data.get("aspect", "")

        questions = debate_data.get("questions", {})
        responses = debate_data.get("responses", {})
        provenance = debate_data.get("provenance", {})

        documents = []
        metadatas = []
        ids = []

        # index both positive and negative arguments
        for position in ("positive", "negative"):
            question = questions.get(position, "")
            response = responses.get(position, "")

            if not question or not response:
                continue

            # create document text
            doc_text = self._create_document_text(topic, question, response, position)

            # generate document id
            doc_id = self._generate_document_id(pair_hash, position, "full_argument")

            # prepare metadata
            prov_data = provenance.get(position, {})
            metadata = {
                "topic": topic, 
                "position": position,
                "aspect": aspect, 
                "pair_hash": pair_hash,
                "model": prov_data.get("model", "unknown")
            }

            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(doc_id)

        # add documents to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        return {"indexed": len(documents)}
    
    def index_debate_pairs_batch(self, debate_pairs: List[Dict]) -> Dict[str, int]:
        """
        Index multiple debate pairs in batch
        
        Args:
            debate_pairs: List of debate pair dictionaries
        
        Returns:
            Dictionary with statistics
        """
        total_indexed = 0
        failed = 0
        
        for debate_data in debate_pairs:
            try:
                result = self.index_debate_pair(debate_data)
                total_indexed += result["indexed"]
            except Exception as e:
                print(f"Failed to index debate pair {debate_data.get('pair_hash', 'unknown')}")
                failed += 1

        return {
            "total_indexed": total_indexed,
            "total_pairs_processed": len(debate_pairs),
            "failed": failed
        }
    
    # def query_arguments(
    #     self, 
    #     query_text: str,
    #     n_results: int = 5,
    #     position_filter: Optional[DebatePosition] = None,
    #     topic_filter: Optional[str] = None) -> Dict[str, Any]:
    #     """Query debate arguments using semantic search
        
    #     Args:
    #         query_text: Search query text
    #         n_results: Number of results to return
    #         position_filter: Filter by position (positive/negative/both)
    #         topic_filter: Filter by specific topic"""