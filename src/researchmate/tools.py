from google.genai import types
from researchmate.schemas import SearchWebArgs
from researchmate.search import search_web

SEARCH_WEB_DECLARATION = types.FunctionDeclaration(
    name="search_web",
    description=(
        "Search the internet for information related to a "
        "research question. Use this when you need external "
        "information, evidence, technical documentation, "
        "academic papers, or recent developments."
    ),
    parameters_json_schema=SearchWebArgs.model_json_schema()
)

RESEARCH_TOOLS = [
    types.Tool(
        function_declarations=[
            SEARCH_WEB_DECLARATION
        ]
    )
]

def execute_search_web(arguments: dict) -> dict:
    validated_args = SearchWebArgs.model_validate(arguments)
    result = search_web(
        query=validated_args.query,
        max_results=validated_args.max_results,
        include_domains=validated_args.include_domains
    )
    return result.model_dump(mode="json")

TOOL_REGISTRY = {
    "search_web": execute_search_web
}

def execute_tool(
    function_name: str,
    arguments: dict
) -> dict:
    if function_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool requested: {function_name}"
        )
    function = TOOL_REGISTRY[function_name]
    return function(arguments)