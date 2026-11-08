from researchmate.llm import generate_structured
from researchmate.schemas import ResearchPlan

PLANNER_SYSTEM_INSTRUCTION = """
You are ResearchMate's research planning component.
Your responsibility is to transform a user's research
request into a structured research plan.
Instructions:
1. Understand the user's actual research objective.
2. Break complex requests into 3 to 5 distinct questions.
3. Make every question independently researchable.
4. Avoid redundant or overlapping questions.
5. Cover different important dimensions of the topic.
6. Explain the purpose of each research question.
7. State important assumptions explicitly.
8. Identify topics outside the requested scope.
9. Do not invent facts or pretend research has already
   been conducted.
10. Keep the plan focused and suitable for web research.
"""

def plan_research(query: str) -> ResearchPlan:
    if not query.strip():
        raise ValueError("Research query cannot be empty.")
    prompt = f"""
    Create a research plan for the following request:
    Research request:
    {query}
    Decompose the request into distinct research questions.
    Ensure the questions collectively cover the user's
    main objective without unnecessary repetition.
    """
    return generate_structured(
        prompt=prompt,
        response_model=ResearchPlan,
        system_instruction=PLANNER_SYSTEM_INSTRUCTION,
        temperature=0.2
    )