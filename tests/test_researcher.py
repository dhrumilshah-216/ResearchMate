import pytest
from types import SimpleNamespace
from unittest.mock import patch
from google.genai import types
from researchmate.researcher import run_research

class FakeModels:

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate_content(self, **kwargs):

        # Take a snapshot of the conversation history
        request_snapshot = {
            **kwargs,
            "contents": list(kwargs["contents"])
        }

        self.calls.append(request_snapshot)

        return self.responses.pop(0)

class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)

def create_response(parts, text=None):
    content = types.Content(
        role="model",
        parts=parts
    )
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=content)
        ],
        text=text
    )

def test_researcher_executes_tool_and_returns_result():
    function_call = types.FunctionCall(
        name="search_web",
        args={
            "query": "AI agents in software engineering",
            "max_results": 3
        }
    )

    tool_response = create_response(
        [
            types.Part(function_call=function_call)
        ]
    )

    final_response = create_response(
        [
            types.Part.from_text(
                text="AI agents are increasingly used in software engineering."
            )
        ],
        text="AI agents are increasingly used in software engineering."
    )

    fake_client = FakeClient([
        tool_response,
        final_response
    ])

    fake_settings = SimpleNamespace(
        gemini_model="test-model"
    )

    fake_search_result = {
        "query": "AI agents in software engineering",
        "sources": [
            {
                "title": "AI Agents Research",
                "url": "https://example.com/agents",
                "content": "Research about AI agents.",
                "snippet": "AI agents in software engineering.",
                "score": 0.95
            }
        ]
    }

    with patch(
        "researchmate.researcher.create_gemini_client",
        return_value=fake_client
    ), patch(
        "researchmate.researcher.get_settings",
        return_value=fake_settings
    ), patch(
        "researchmate.researcher.execute_tool",
        return_value=fake_search_result
    ) as mock_execute:
        result = run_research(
            "How are AI agents used in software engineering?"
        )

    assert result.total_searches == 1
    assert len(result.sources) == 1
    assert result.sources[0].title == "AI Agents Research"
    assert "AI agents" in result.summary
    mock_execute.assert_called_once()
    # First request contains user input.
    # Second request also contains the model's tool call
    # and the corresponding tool response.
    assert len(fake_client.models.calls) == 2
    second_request = fake_client.models.calls[1]
    assert len(second_request["contents"]) == 3
    assert second_request["contents"][-1].role == "user"
    assert (
    second_request["contents"][-1].parts[0].function_response
    is not None
)

def test_researcher_rejects_empty_query():
    with pytest.raises(ValueError):
        run_research("   ")