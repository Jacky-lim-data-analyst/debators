"""Utility script to query and view debate results from MongoDB"""
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional
from pymongo import AsyncMongoClient, ASCENDING
import sys
import os

# Ensure the project root is on `sys.path`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MONGODB_PORT

class DebateQueryTool:
    """Tool for querying debate_results from MongoDB"""

    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        database_name: str = "debate_db",
        collection_name: str = "first_debate"
    ):
        self.mongodb_uri = mongodb_uri or f"mongodb://127.0.0.1:{MONGODB_PORT}"
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None

    async def connect(self):
        """Establish MongoDB connection"""
        self.client = AsyncMongoClient(self.mongodb_uri)
        return self.client[self.database_name][self.collection_name]
    
    async def disconnect(self):
        """Close mongoDB connection"""
        if self.client:
            await self.client.close()

    async def list_all_topics(self) -> List[str]:
        """Get all unique debate topics in the database"""
        collection = await self.connect()
        try:
            topics = await collection.distinct("topic")
            return sorted(topics)
        finally:
            await self.disconnect()

    async def count_debates(self, topic: Optional[str] = None) -> int:
        """Count total number of debate pairs per topic
        Args:
            topic: Optional topic filter"""
        collection = await self.connect()
        try:
            query = {"topic": topic} if topic else {}
            count = await collection.count_documents(query)
            return count
        finally:
            await self.disconnect()

    async def get_debates_by_topic(self, topic: str) -> List[Dict]:
        """Retrieve all debate pairs for a specific topic
        
        Args:
            topic: The debate topic to search for
        """
        collection = await self.connect()
        try:
            cursor = collection.find({"topic": topic}).sort("pair_index", ASCENDING)
            debates = await cursor.to_list(length=None)
            return debates
        finally:
            await self.disconnect()

    async def get_statistics(self) -> Dict:
        """Get overall statistics about stored debates"""
        collection = await self.connect()
        try:
            total_pairs = await collection.count_documents({})
            topics = await collection.distinct("topic")

            # get model usage statistics
            pipeline = [
                {
                    "$group": {
                        "_id": "$provenance.positive.model",
                        "count": {"$sum": 1}
                    }
                }
            ]

            prop_models = {}
            async with await collection.aggregate(pipeline) as cursor:
                async for doc in cursor:
                    prop_models[doc["_id"]] = doc["count"]

            pipeline[0]["$group"]["_id"] = "$provenance.negative.model"
            opp_models = {}
            async with await collection.aggregate(pipeline) as cursor:
                async for doc in cursor:
                    opp_models[doc["_id"]] = doc["count"]

            return {
                "total_pairs": total_pairs,
                "unique_topics": len(topics),
                "topics": topics,
                "proposition_models": prop_models,
                "opposition_models": opp_models
            }

        finally:
            await self.disconnect()

    # def format_debate_display(self, debate: Dict, show_full: bool = False) -> str:
    #     """Format a de"""

async def main():
    query_tool = DebateQueryTool()

    topics = await query_tool.list_all_topics()
    print(topics)
    # stats = await query_tool.get_statistics()
    # print(stats)
    topic = 'The house believes that social media algorithms should be regulated as public utilities.'
    debate_pairs = await query_tool.get_debates_by_topic(topic=topic)
    print(debate_pairs)

if __name__ == "__main__":
    asyncio.run(main())
