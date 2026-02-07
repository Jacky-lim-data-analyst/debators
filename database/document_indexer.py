"""Connection to local MongoDB instance"""
import hashlib, json
from datetime import datetime
import asyncio
from pymongo import AsyncMongoClient

import os
import sys

# Ensure the project root is on `sys.path` so imports like `from config import ...`
# work when this script is run as a module or from different working directories.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MONGODB_PORT

# database schema as shown in `design_choice.md`

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def canonicalize(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)

def make_pair_doc(topic, q_pos, r_pos, q_neg, r_neg, model_pos, model_neg):
    prov = {
        "positive": {
            "model": model_pos,
            "created_at": datetime.now().isoformat()
        },
        "negative": {
            "model": model_neg,
            "created_at": datetime.now().isoformat()
        }
    }
    pair_source = canonicalize([topic, q_pos, q_neg, model_pos, model_neg])
    pair_hash = sha256_hex(pair_source)
    doc = {
        "topic": topic,
        "questions": {"positive": q_pos, "negative": q_neg},
        "responses": {"positive": r_pos, "negative": r_neg},
        "provenance": prov,
        "pair_hash": pair_hash
    }
    return doc

# ===== main function ====
async def main():
    uri = "mongodb://127.0.0.1:" + str(MONGODB_PORT)
    client = AsyncMongoClient(uri)

    try:
        # get or create a collection
        # database = client["<new_database>"]
        database = client["test"]
        collection = database["examples"]
        doc = make_pair_doc(
            topic="Should we implement a universal basic income (UBI)?",
            q_pos="Good UBI",
            q_neg="Bad UBI",
            r_pos="Yes",
            r_neg="No",
            model_pos="claude-X",
            model_neg="llama-X"
        )
        # await database.create_collection("examples")
        await database.examples.insert_one(doc)

        # find multiple
        # results = collection.find({"question": "Who are you?"})

        # async for document in results:
        #     print(document)

        await client.close()

    except Exception as e:
        raise Exception(f"Unable to insert document: {e}")
    
if __name__ == "__main__":
    asyncio.run(main())
