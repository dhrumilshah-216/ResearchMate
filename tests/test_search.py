import pytest

from unittest.mock import MagicMock, patch

from researchmate.search import search_web
from researchmate.schemas import ResearchSource, SearchResponse


def create_fake_client(response):
    fake_client = MagicMock()
    fake_client.search.return_value = response
    return fake_client


def test_search_web_returns_sources():

    fake_response = {
        "results": [
            {
                "title": "Graph RAG Research",
                "url": "https://example.com/graph-rag",
                "content": "A short description.",
                "raw_content": "Full extracted research content.",
                "score": 0.95
            }
        ]
    }

    fake_client = create_fake_client(fake_response)

    with patch(
        "researchmate.search.create_tavily_client",
        return_value=fake_client
    ):

        result = search_web("Graph RAG")

    assert isinstance(result, SearchResponse)
    assert len(result.sources) == 1

    source = result.sources[0]

    assert isinstance(source, ResearchSource)
    assert source.title == "Graph RAG Research"
    assert source.url == "https://example.com/graph-rag"
    assert source.content == "Full extracted research content."
    assert source.snippet == "A short description."
    assert source.score == 0.95


def test_search_web_uses_snippet_when_raw_content_missing():

    fake_response = {
        "results": [
            {
                "title": "Example Research",
                "url": "https://example.com",
                "content": "Only snippet available.",
                "score": 0.8
            }
        ]
    }

    fake_client = create_fake_client(fake_response)

    with patch(
        "researchmate.search.create_tavily_client",
        return_value=fake_client
    ):

        result = search_web("Example research")

    assert result.sources[0].content == "Only snippet available."


def test_search_web_rejects_empty_query():

    with pytest.raises(ValueError):
        search_web("   ")


def test_search_web_rejects_long_query():

    with pytest.raises(ValueError):
        search_web("a" * 401)


def test_search_web_rejects_invalid_max_results():

    with pytest.raises(ValueError):
        search_web("Graph RAG", max_results=20)


def test_search_web_passes_search_parameters():

    fake_client = create_fake_client({"results": []})

    with patch(
        "researchmate.search.create_tavily_client",
        return_value=fake_client
    ):

        search_web(
            "Graph RAG",
            max_results=3,
            include_domains=["arxiv.org"]
        )

    fake_client.search.assert_called_once_with(
        query="Graph RAG",
        search_depth="advanced",
        topic="general",
        max_results=3,
        include_domains=["arxiv.org"],
        include_raw_content="markdown",
        include_answer=False
    )


def test_search_web_handles_no_results():

    fake_client = create_fake_client({"results": []})

    with patch(
        "researchmate.search.create_tavily_client",
        return_value=fake_client
    ):

        result = search_web("A very specific research topic")

    assert isinstance(result, SearchResponse)
    assert result.sources == []