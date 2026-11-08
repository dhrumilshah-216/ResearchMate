from typing import TypeVar
from google.genai import types, errors
from pydantic import BaseModel
from researchmate.client import create_gemini_client
from researchmate.config import get_settings

T = TypeVar("T", bound=BaseModel)

def generate_text(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.2
) -> str:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    settings = get_settings()
    client = create_gemini_client()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature
    )
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=config
        )
    except errors.APIError as exc:
        raise RuntimeError(
            f"Gemini API error ({exc.code}): {exc.message}"
        ) from exc
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response.text

def generate_structured(
    prompt: str,
    response_model: type[T],
    system_instruction: str | None = None,
    temperature: float = 0.2
) -> T:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    settings = get_settings()
    client = create_gemini_client()
    schema = response_model.model_json_schema()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type="application/json",
        response_json_schema=schema
    )
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=config
        )
    except errors.APIError as exc:
        raise RuntimeError(
            f"Gemini API error ({exc.code}): {exc.message}"
        ) from exc
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response_model.model_validate_json(response.text)