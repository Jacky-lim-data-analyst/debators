from langchain.tools import tool
from typing import List, Protocol, Dict
import os
import sys
import requests

# Ensure the project root is on `sys.path` so imports like `from config import ...`
# work when this script is run as a module or from different working directories.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SEARXNG_PORT, WHOOGLE_PORT, SERVER_IP_ADDRESS
from dotenv import load_dotenv

from ddgs import DDGS
from tavily import TavilyClient


# ------------
# Output standardization:
# [{
#    "title": ...,
#    "url": ...,
#    "content": ...,
#    "source": ...,
# }]
# ----------
class WebSearchProvider(Protocol):
    def search(self, query: str, **kwargs) -> List[Dict[str, str]]: ...


class DuckDuckGoProvider:
    def __init__(self, max_results: int = 10) -> None:
        self.max_results = max_results

    def search(self, query: str, **kwargs) -> List[Dict[str, str]]:
        """Web search by ddg"""
        results = DDGS().text(query, max_results=self.max_results)

        if not results:
            print("Duckduckgo returns empty results")
            return []

        # return results
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "content": r.get("body", ""),
                "source": "duckduckgo"
            } for r in results
        ]
    
class TavilyProvider:
    def __init__(self, api_key: str | None = None, max_results: int = 10):
        if not api_key:
            api_key = os.environ.get("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=api_key)
        self.max_results = max_results

    def search(self, query: str, **kwargs) -> List[Dict[str, str]]:
        """Web search by tavily"""
        search_results = self.client.search(query, max_results=self.max_results)

        results = search_results.get('results')
        if results is None:
            print("Tavily returns empty result")
            return []
        
        return [
            {
                'title': r.get("title", ""),
                'url': r.get('url', ""),
                'content': r.get('content', ''),
                'source': "Tavily"
            } for r in results
        ]
    
class SearXNGProvider:
    def __init__(self, port: int | None = None):
        if not port:
            self.port = SEARXNG_PORT

    def search(self, query: str, **kwargs) -> List[Dict[str, str]]:
        """Web search with locally hosted searXNG provider"""
        resp_post_redirect = requests.post(
            "http://localhost:" + str(self.port),
            data={"q": query, "format":"json"},
            allow_redirects=True,
            timeout=10
        )
        try:
            resp_post_redirect.raise_for_status()
            json_data = resp_post_redirect.json()

            results = json_data.get("results")
            if results is None:
                print("SearXNG returns empty results")
                return []
            return [
                {
                    'title': r.get("title", ""),
                    'url': r.get("url", ""),
                    'content': r.get("content", ""),
                    'source': 'SearXNG'
                } for r in results
            ]
        except Exception as e:
            print(f"SearXNG error: {e}")
            return []

class WhoogleProvider:
    def __init__(self, 
                 local_host: str | None = None, 
                 port: int | None = None,
                 max_results: int = 10):
        if not local_host:
            self.local_host = SERVER_IP_ADDRESS
        if not port:
            self.port = WHOOGLE_PORT
        self.max_results = max_results

    def search(self, query: str, **kwargs):
        """Web search by whoogle"""
        params = {
            "q": query,
            "format": "json",
            "num": self.max_results
        }
        headers = {"Accept": "application/json"}

        try:
            base_url = "http://" + self.local_host + ":" + str(self.port) + "/search" 
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            raw_results = data.get("results", [])

            if not raw_results:
                print("Whoogle returns empty results")
                return []
                
            return [
                {
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "content": r.get("content"),
                    "source": "whoogle"
                } for r in raw_results
            ]
            
        except Exception as ex:
            print(f"Whoogle error : {str(ex)}")
            return []

if __name__ == "__main__":
    # load_dotenv()

    provider = WhoogleProvider()

    print(provider.search("Moon landing"))

