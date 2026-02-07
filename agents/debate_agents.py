"""Model providers:
1. Cerebras
2. Groq
3. Google GenAI
4. OpenRouter
5. ZAI
6. Ollama
7. DeepSeek"""

from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

import os
import sys
import random
from typing import List
import logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SERVER_IP_ADDRESS, OLLAMA_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RotatingFallbackDebator:
    def __init__(self, 
                 temperature: float = 0.1, 
                 max_tokens: int = 4096,
                 timeout: float = 10.0):
        load_dotenv()
        connection_string = "http://" + SERVER_IP_ADDRESS + ":" + str(OLLAMA_PORT)
        # define all models
        self.all_models = [
            ChatCerebras(model="llama-3.1-8b", temperature=temperature, max_tokens=max_tokens, timeout=timeout),
            ChatGroq(model="openai/gpt-oss-20b", temperature=temperature, max_tokens=max_tokens, timeout=timeout),
            ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=temperature, max_output_tokens=max_tokens, timeout=timeout),
            ChatOllama(
                base_url=connection_string,
                model="granite4:3b",
                temperature=temperature
            )
        ]

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key is not None:
            self.all_models.append(
                ChatOpenAI(
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    model="nvidia/nemotron-3-nano-30b-a3b:free",
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    timeout=timeout
                )
            )

        zai_api_key = os.getenv("ZAI_API_KEY")
        if zai_api_key is not None:
            self.all_models.append(
                ChatOpenAI(
                    model="glm-4.7-flash",
                    api_key=zai_api_key,
                    base_url="https://api.z.ai/api/paas/v4/",
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    timeout=timeout
                )
            )

        self.model_names = [
            "llama-3.1", "gpt-oss-20b", "gemini-flash-preview", "granite4:3b", 
            "nemotron-3-nano", "glm-4.7-flash"
        ]

    def get_chain_with_rotation(self):
        """Randomly shuffle models, then create fallback chain in that order.
        Each call gets a different random ordering"""
        # create a shuffled copy of indices
        indices = list(range(len(self.all_models)))
        random.shuffle(indices)

        # get shuffled models
        shuffled_models = [self.all_models[i] for i in indices]
        shuffled_names = [self.model_names[i] for i in indices]

        logger.info(f"Model order for this request: {' -> '.join(shuffled_names)}")

        # primary is first, rest are fallbacks
        primary = shuffled_models[0]
        fallbacks = shuffled_models[1:]

        # create chain with fallbacks
        return primary.with_fallbacks(fallbacks), shuffled_names
    
    def invoke(self, prompt: str) -> str:
        chain, model_order = self.get_chain_with_rotation()

        try:
            response = chain.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"All models failed in order {model_order}: {str(e)}")
            raise

if __name__ == "__main__":
    llm = RotatingFallbackDebator()

    response = llm.invoke("What is the distance between Earth and Moon?")
    print(f"Response: {response}")
