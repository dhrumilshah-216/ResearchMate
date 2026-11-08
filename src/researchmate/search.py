from researchmate.client import create_tavily_client
from researchmate.schemas import ResearchSource, SearchResponse

MAX_CONTENT_LENGTH = 12000

def search_web(
    query: str,
    max_results: int = 5,
    include_domains: list[str] | None = None
) -> SearchResponse:
    if not query.strip():
        raise ValueError("Search query cannot be empty.")
    if len(query) > 400:
        raise ValueError("Search query cannot exceed 400 characters.")
    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10.")
    client = create_tavily_client()
    response = client.search(
        query=query,
        search_depth="advanced",
        topic="general",
        max_results=max_results,
        include_domains=include_domains,
        include_raw_content="markdown",
        include_answer=False
    )
    sources = []
    for result in response.get("results", []):
        snippet = result.get("content") or ""
        raw_content = result.get("raw_content") or ""
        content = raw_content or snippet
        source = ResearchSource(
            title=result.get("title") or "Untitled source",
            url=result["url"],
            content=content[:MAX_CONTENT_LENGTH],
            snippet=snippet,
            score=result.get("score", 0.0)
        )
        sources.append(source)
    return SearchResponse(
        query=query,
        sources=sources
    )