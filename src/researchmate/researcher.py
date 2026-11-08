from google.genai import types
from researchmate.client import create_gemini_client
from researchmate.config import get_settings
from researchmate.tools import RESEARCH_TOOLS, execute_tool
from researchmate.schemas import ResearchResult, ResearchSource

MAX_TOOL_CALLS = 8
MAX_MODEL_TURNS = 12

RESEARCHER_SYSTEM_INSTRUCTION = """
You are ResearchMate's autonomous Researcher Agent.
Your job is to investigate a research question using the
available web search tool.

Instructions:
1. Understand the research question before searching.
2. Decide what information needs to be investigated.
3. Use search_web when external information is needed.
4. Construct specific and useful search queries.
5. Examine the returned source titles and extracted content.
6. Identify information gaps and decide whether additional
   searches are necessary.
7. Avoid repeating searches that provide no new value.
8. Prefer primary sources, academic papers, official
   documentation and credible publications.
9. Never treat search relevance scores as proof of accuracy.
10. Never invent sources, URLs, statistics or citations.
11. Distinguish established information from uncertainty.
12. Stop searching when sufficient useful evidence has
    been collected or when no further search is valuable.

When you have finished researching, provide a detailed,
evidence-grounded summary.
Cite sources using their actual URLs in Markdown links.
You must not claim to have verified information that you
have not actually inspected.
"""

def run_research(query: str) -> ResearchResult:
    if not query.strip():
        raise ValueError("Research query cannot be empty.")
    settings = get_settings()
    client = create_gemini_client()
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"""
                    Research the following question:
                    {query}
                    Investigate it using the available tools.
                    Decide dynamically when additional searches
                    are necessary.
                    """
                )
            ]
        )
    ]
    config = types.GenerateContentConfig(
        system_instruction=RESEARCHER_SYSTEM_INSTRUCTION,
        tools=RESEARCH_TOOLS,
        automatic_function_calling=(
            types.AutomaticFunctionCallingConfig(
                disable=True
            )
        ),
        temperature=0.2
    )
    collected_sources = {}
    total_searches = 0
    tool_calls = 0
    for _ in range(MAX_MODEL_TURNS):
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=contents,
            config=config
        )
        if not response.candidates:
            raise RuntimeError(
                "Gemini returned no response candidates."
            )
        model_content = response.candidates[0].content
        if model_content is None:
            raise RuntimeError(
                "Gemini returned an empty content object."
            )
        # Preserve the model's response in conversation history.
        contents.append(model_content)
        function_calls = [
            part.function_call
            for part in model_content.parts
            if part.function_call is not None
        ]
        # No function calls means the model has finished.
        if not function_calls:
            if not response.text:
                raise RuntimeError(
                    "Researcher returned no final text."
                )
            sources = [
                ResearchSource.model_validate(source)
                for source in collected_sources.values()
            ]
            return ResearchResult(
                research_question=query,
                summary=response.text,
                sources=sources,
                total_searches=total_searches
            )
        function_response_parts = []
        # Execute requested functions sequentially.
        for function_call in function_calls:
            if tool_calls >= MAX_TOOL_CALLS:
                tool_result = {
                    "error": (
                        "Tool execution limit reached. "
                        "Finish using the evidence already collected."
                    )
                }
            else:
                tool_calls += 1
                try:
                    arguments = dict(function_call.args or {})
                    tool_result = execute_tool(
                        function_name=function_call.name,
                        arguments=arguments
                    )
                    if function_call.name == "search_web":
                        total_searches += 1
                        for source in tool_result.get("sources", []):
                            collected_sources[source["url"]] = source
                except Exception as exc:
                    # Send the error back to Gemini so it can
                    # decide how to recover.
                    tool_result = {
                        "error": str(exc)
                    }
            function_response_parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response=tool_result
                )
            )
        # Return all tool results from this model turn.
        contents.append(
            types.Content(
                role="user",
                parts=function_response_parts
            )
        )
    raise RuntimeError(
        "Researcher exceeded the maximum model turn limit."
    )