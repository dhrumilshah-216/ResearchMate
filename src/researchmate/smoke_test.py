from researchmate.llm import (
    generate_text,
    generate_structured
)

from researchmate.schemas import TopicBrief


def main():

    print("\nTesting ordinary text generation...\n")

    text_result = generate_text(
        prompt="Explain what an AI research assistant does in two sentences.",
        system_instruction=(
            "You are a clear and concise AI engineering tutor."
        )
    )

    print(text_result)

    print("\nTesting structured generation...\n")

    structured_result = generate_structured(
        prompt=(
            "Give me a brief overview of graph neural networks. "
            "Include important concepts and open research questions."
        ),
        response_model=TopicBrief,
        system_instruction=(
            "You are a technical research assistant. "
            "Return accurate and concise information."
        )
    )

    print(structured_result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()