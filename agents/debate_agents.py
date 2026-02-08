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
from langchain.agents import create_agent

# from langchain_deepseek import ChatDeepSeek
from openai import OpenAI

import os
import sys
import random
import logging
from typing import List, Dict
import json
from tools.web_search import perform_web_search_tool, perform_web_search

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
                 timeout: float = 10.0
        ):
        load_dotenv()
        connection_string = "http://" + SERVER_IP_ADDRESS + ":" + str(OLLAMA_PORT)
        # define all models
        self.all_models = [
            ChatCerebras(model="llama-3.1-8b", temperature=temperature, max_tokens=max_tokens, timeout=timeout),
            ChatGroq(model="openai/gpt-oss-20b", temperature=temperature, max_tokens=max_tokens, timeout=timeout),
            ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature, max_output_tokens=max_tokens, timeout=timeout),
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
    
    def invoke(self, prompt: Dict):
        """Prompt is of the following format:
        [
            ("system", "..."),
            ("human", "{topic}, {question}")
        ]"""
        chain, model_order = self.get_chain_with_rotation()

        agent = create_agent(chain, tools=[perform_web_search_tool], system_prompt="You are a helpful assistant")

        try:
            response = agent.invoke(prompt)
            final_response = response['messages'][-1]
            return final_response.content
        except Exception as e:
            logger.error(f"All models failed in order {model_order}: {str(e)}")
            return ""

class SoloDebator:
    """Debators from one model with graceful error handling and content discovery"""
    def __init__(self, 
                 api_key: str | None = None,
                 temperature: float = 0.1,
                 max_tokens: int = 4096,
        ):
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("DEEPSEEK_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "perform_web_search",
                    "description": "Search the web for latest information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search keyword to get relevant information"
                            },
                        },
                        "required": ["query"],
                    },
                }
            },
        ]
        
    def invoke(self, prompt: List[Dict[str, str]]):
        """
        Invokes the model and handles tool calls.
        Returns content and a disclaimer if the process was interrupted"""
        last_content = ""
        try:
            messages = prompt
            # Use a loop to handle multiple tool turns
            for _ in range(5):  # Limit to 5 turns to prevent infinite loops
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=False,
                    tools=self.tools,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                assistant_message = response.choices[0].message
                messages.append(assistant_message)

                # update last known content is available
                if assistant_message.content:
                    last_content = assistant_message.content

                # If there are no tool calls, this is the final answer
                if not assistant_message.tool_calls:
                    return last_content

                # Process ALL tool calls generated in this turn
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    if function_name == "perform_web_search":
                        query = arguments.get("query")
                        logger.info(f"Executing search for: {query}")
                        result = perform_web_search(query)
                        
                        # Append the tool result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result)
                        })

            # Turn limit reached
            warning = "\n\n[Disclaimer: The maximum tool calling depth was reached. This response might be incomplete]"
            return last_content + warning

        except Exception as e:
            logger.error(f"Model API requests error: {e}")
            # return whatever we gathered so far
            warning = f"\n\n[Disclaimer: An error occurred during invoke ({type(e).__name__})."
            return (last_content if last_content else "No response") + warning

if __name__ == "__main__":
    llm = RotatingFallbackDebator()

    response = llm.invoke({"messages": [{"role": "user", "content": "Who is the current American President?"}]})
    print(f"Response: {response}")
    # load_dotenv()
    # client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    # print(client.models.list())
    # model = SoloDebator()
    # print(model.invoke([
    #     {"role": "system", "content": "You are a helpful assistant"},
    #     {"role": "user", "content": "What is the current status of Artemis II project?"},
    # ]))
    # print(perform_web_search("Artemis II"))
