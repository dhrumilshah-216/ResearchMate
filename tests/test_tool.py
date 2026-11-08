import pytest
from unittest.mock import patch
from pydantic import ValidationError
from researchmate.tools import execute_tool
from researchmate.schemas import SearchResponse

def test_execute_search_web_validates_arguments():
    with pytest.raises(ValidationError):
        execute_tool(
            "search_web",
            {
                "query": "Graph RAG",
                "max_results": 100
            }
        )

def test_execute_search_web_returns_normalized_data():
    fake_result = SearchResponse(
        query="Graph RAG",
        sources=[]
    )
    with patch(
        "researchmate.tools.search_web",
        return_value=fake_result
    ) as mock_search:
        result = execute_tool(
            "search_web",
            {
                "query": "Graph RAG",
                "max_results": 3
            }
        )
    assert result["query"] == "Graph RAG"
    assert result["sources"] == []
    mock_search.assert_called_once_with(
        query="Graph RAG",
        max_results=3,
        include_domains=None
    )

def test_execute_tool_rejects_unknown_function():
    with pytest.raises(ValueError):
        execute_tool(
            "delete_everything",
            {}
        )