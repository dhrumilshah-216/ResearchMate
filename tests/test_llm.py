import pytest
from types import SimpleNamespace
from pydantic import ValidationError
from researchmate.schemas import TopicBrief
from researchmate.llm import (
    generate_text,
    generate_structured
)

class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.last_request = None
    def generate_content(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(
            text=self.response_text
        )

class FakeClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)

def test_generate_text(monkeypatch):
    fake_client = FakeClient("Graph neural networks are useful.")
    monkeypatch.setattr(
        "researchmate.llm.create_gemini_client",
        lambda: fake_client
    )
    monkeypatch.setattr(
        "researchmate.llm.get_settings",
        lambda: SimpleNamespace(
            gemini_model="test-model"
        )
    )
    result = generate_text("Explain graph neural networks.")
    assert result == "Graph neural networks are useful."

def test_generate_text_rejects_empty_prompt():
    with pytest.raises(ValueError):
        generate_text("   ")

def test_generate_structured(monkeypatch):
    response_json = """
    {
        "topic": "Graph Neural Networks",
        "summary": "A neural network approach for graph data.",
        "key_points": ["Nodes", "Edges", "Message passing"],
        "open_questions": ["How does scalability affect performance?"]
    }
    """
    fake_client = FakeClient(response_json)
    monkeypatch.setattr(
        "researchmate.llm.create_gemini_client",
        lambda: fake_client
    )
    monkeypatch.setattr(
        "researchmate.llm.get_settings",
        lambda: SimpleNamespace(
            gemini_model="test-model"
        )
    )
    result = generate_structured(
        prompt="Explain graph neural networks.",
        response_model=TopicBrief
    )
    assert isinstance(result, TopicBrief)
    assert result.topic == "Graph Neural Networks"
    assert len(result.key_points) == 3

def test_generate_structured_rejects_invalid_data(monkeypatch):
    invalid_json = """
    {
        "topic": 123,
        "summary": "Some summary",
        "key_points": "This should be a list",
        "open_questions": []
    }
    """
    fake_client = FakeClient(invalid_json)
    monkeypatch.setattr(
        "researchmate.llm.create_gemini_client",
        lambda: fake_client
    )
    monkeypatch.setattr(
        "researchmate.llm.get_settings",
        lambda: SimpleNamespace(
            gemini_model="test-model"
        )
    )
    with pytest.raises(ValidationError):
        generate_structured(
            prompt="Explain graph neural networks.",
            response_model=TopicBrief
        )