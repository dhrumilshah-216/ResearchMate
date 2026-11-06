from google import genai

from researchmate.config import get_settings


def create_gemini_client():
    settings = get_settings()

    client = genai.Client(
        api_key=settings.gemini_api_key
    )

    return client