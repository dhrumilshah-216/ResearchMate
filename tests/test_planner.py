import pytest

from unittest.mock import patch
from pydantic import ValidationError

from researchmate.planner import plan_research
from researchmate.schemas import ResearchPlan, ResearchQuestion


def test_research_plan_schema():

    plan = ResearchPlan(
        objective="Compare two retrieval architectures.",
        research_questions=[
            ResearchQuestion(
                question="How do their architectures differ?",
                purpose="Understand their technical foundations."
            ),
            ResearchQuestion(
                question="How do they compare in scalability?",
                purpose="Understand their enterprise performance."
            ),
            ResearchQuestion(
                question="Which enterprise use cases suit each?",
                purpose="Identify practical applications."
            )
        ],
        assumptions=["The comparison focuses on enterprise systems."],
        excluded_topics=["Unrelated machine learning architectures."]
    )

    assert len(plan.research_questions) == 3


def test_research_plan_rejects_too_few_questions():

    with pytest.raises(ValidationError):

        ResearchPlan(
            objective="Compare two retrieval architectures.",
            research_questions=[
                {
                    "question": "How do their architectures differ?",
                    "purpose": "Understand technical foundations."
                }
            ],
            assumptions=[],
            excluded_topics=[]
        )


def test_research_plan_rejects_duplicate_questions():

    with pytest.raises(ValidationError):

        ResearchPlan(
            objective="Compare two retrieval architectures.",
            research_questions=[
                {
                    "question": "How do their architectures differ?",
                    "purpose": "Understand technical foundations."
                },
                {
                    "question": "HOW DO THEIR ARCHITECTURES DIFFER?",
                    "purpose": "Understand technical foundations."
                },
                {
                    "question": "Which use cases suit each approach?",
                    "purpose": "Identify practical applications."
                }
            ],
            assumptions=[],
            excluded_topics=[]
        )


@patch("researchmate.planner.generate_structured")
def test_plan_research_calls_llm(mock_generate):

    expected_plan = ResearchPlan(
        objective="Compare two retrieval architectures.",
        research_questions=[
            {
                "question": "How do their architectures differ?",
                "purpose": "Understand technical foundations."
            },
            {
                "question": "How do they compare in scalability?",
                "purpose": "Understand enterprise performance."
            },
            {
                "question": "Which use cases suit each approach?",
                "purpose": "Identify practical applications."
            }
        ],
        assumptions=[],
        excluded_topics=[]
    )

    mock_generate.return_value = expected_plan

    result = plan_research(
        "Compare Graph RAG and traditional RAG."
    )

    assert result == expected_plan

    mock_generate.assert_called_once()

    call_arguments = mock_generate.call_args.kwargs

    assert call_arguments["response_model"] == ResearchPlan
    assert "Graph RAG" in call_arguments["prompt"]


def test_plan_research_rejects_empty_query():

    with pytest.raises(ValueError):
        plan_research("   ")