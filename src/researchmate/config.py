import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    tavily_api_key: str | None

def get_settings() -> Settings:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    return Settings(
        gemini_api_key=api_key,
        gemini_model=model,
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
