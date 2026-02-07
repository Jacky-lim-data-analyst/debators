from langchain.tools import tool
from typing import List, Protocol, Dict, Literal, Type, Optional
import os
import sys
import requests
from random import shuffle

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

    def search(self, query: str, **kwargs) -> List[Dict[str, str]]:
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

class ChainedWebSearchProvider:
    """Chain of responsibility pattern for web search providers.
    Tries providers in order until one succeeds or all fail"""
    def __init__(self,
                 provider_chain: Optional[List[Literal['duckduckgo', 'tavily', 'searxng', 'whoogle']]] = None,
                 max_results = 10,
                 verbose: bool = True):
        """Initialize the chained web search provider
        
        Args:
            provider_chain: List of provider names in order of preference. 
                          Defaults to ['duckduckgo', 'tavily', 'searxng', 'whoogle']
            max_results: Maximum number of results to return
            verbose: Whether to print status messages"""
        if provider_chain is None:
            provider_chain = ['duckduckgo', 'tavily', 'searxng', 'whoogle']
            shuffle(provider_chain)

        self.provider_chain = provider_chain
        self.max_results = max_results
        self.verbose = verbose
        self._providers_cache: Dict[str, WebSearchProvider] = {}

    def _get_provider_instance(self, provider_name: Literal['duckduckgo', 'tavily', 'searxng', 'whoogle']) -> WebSearchProvider:
        """Get or create a provider instance with caching."""
        if provider_name in self._providers_cache:
            return self._providers_cache[provider_name]
        
        providers: Dict[str, Type[WebSearchProvider]] = {
            'duckduckgo': DuckDuckGoProvider,
            'tavily': TavilyProvider,
            'searxng': SearXNGProvider,
            'whoogle': WhoogleProvider
        }
        
        provider_class = providers.get(provider_name, DuckDuckGoProvider)
        provider_instance = provider_class(max_results=self.max_results)
        self._providers_cache[provider_name] = provider_instance
        
        return provider_instance
    
    def search(self, query: str, **kwargs) -> List[Dict[str, str]]:
        """
        Search using the chain of providers. Falls back to next provider if current one fails.
        
        Args:
            query: Search query string
            **kwargs: Additional arguments passed to provider search methods
            
        Returns:
            List of search results from the first successful provider
            
        Raises:
            Exception: If all providers in the chain fail
        """
        last_exception = None

        for provider_name in self.provider_chain:
            try:
                if self.verbose:
                    print(f"Trying provider: {provider_name}")

                provider = self._get_provider_instance(provider_name)
                results = provider.search(query, **kwargs)

                # checks if results are valid
                if results:
                    if self.verbose:
                        print(f"✓ Successfully retrieved {len(results)} results from {provider_name}")
                    return results
                else:
                    if self.verbose:
                        print(f"✗ {provider_name} returned empty results, trying next provider...")

            except Exception as e:
                last_exception = e
                if self.verbose:
                    print(f"✗ {provider_name} failed with error: {str(e)}")
                    print(f"  Falling back to next provider...")
                continue

        # all providers failed
        error_msg = f"All providers in chain {self.provider_chain} failed."
        if last_exception:
            error_msg += f" Last error: {str(last_exception)}. Returning empty list"
        if self.verbose:
            print(error_msg)
        return []
        
def get_ws_provider(provider_name: Literal['duckduckgo', 'tavily', 'searxng', 'whoogle']) ->  WebSearchProvider:
    """Factory method"""
    providers: Dict[str, Type[WebSearchProvider]] = {
        'duckduckgo': DuckDuckGoProvider,
        'tavily': TavilyProvider,
        'searxng': SearXNGProvider,
        'whoogle': WhoogleProvider
    }

    return providers.get(provider_name, DuckDuckGoProvider)()

def get_chained_search_provider(
    provider_chain: Optional[List[Literal['duckduckgo', 'tavily', 'searxng', 'whoogle']]] = None,
    max_results: int = 10,
    verbose: bool = True
) -> ChainedWebSearchProvider:
    """
    Factory method for creating a chained web search provider.
    
    Args:
        provider_chain: List of provider names in order of preference
        max_results: Maximum number of results to return
        verbose: Whether to print status messages
        
    Returns:
        ChainedWebSearchProvider instance
    """
    return ChainedWebSearchProvider(
        provider_chain=provider_chain,
        max_results=max_results,
        verbose=verbose
    ) 

if __name__ == "__main__":
    # load_dotenv()

    # provider = get_ws_provider('duckduckgo')
    chained_provider = get_chained_search_provider()

    print(chained_provider.search("Green energy"))
