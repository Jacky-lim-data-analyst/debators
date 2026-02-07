# design choices

## MongoDB database schema
Store each opposing pair with a stable pair-id, keep the exact topic, questions (positive & negative) + models.

Pairs of single documents
```json
{
    "_id": ObjectId(...),
    "topic": "ai_impacts",
    "questions": {
        "positive": "Positive impacts of AI",
        "negative": "Negative impacts of AI"
    },
    "responses": {
        "positive": "<LLM response>",
        "negative": "<LLM response>"
    },
    "provenance": {
        "positive": {
            "model": "gpt-X",
            "created_at": ISODate(...)   # iso string format
        },
        "negative": {
            "model": "gemini-X",
            "created_at": ISODate(...)  # iso string format
        }
    },
    "pair_hash": "sha256hex(...)"
}
```

How to guarantee exact retrieval later
1. Use a deterministic hash (pair_hash).

    Hash canonicalized: `hash(topic + q_pos + q_neg + model_pos + model_neg)` (use canonical whitespace, lowercase if appropriate). Store SHA-256 hex. Query the `pair_hash` to get the exact pair that produced the stored text.
