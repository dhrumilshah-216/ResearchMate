from google import genai
from researchmate.config import get_settings
from tavily import TavilyClient

def create_gemini_client():
    settings = get_settings()

    client = genai.Client(
        api_key=settings.gemini_api_key
    )

    return client

def create_tavily_client():
    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. Add it to your .env file."
        )
    return TavilyClient(
        api_key=settings.tavily_api_key
    )