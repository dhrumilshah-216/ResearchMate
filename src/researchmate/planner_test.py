from researchmate.planner import plan_research

def main():
    query = input("\nEnter your research question: ").strip()
    print("\nGenerating research plan...\n")
    plan = plan_research(query)
    print("=" * 60)
    print("RESEARCH OBJECTIVE")
    print("=" * 60)
    print(plan.objective)
    print("\nRESEARCH QUESTIONS")
    print("=" * 60)
    for index, question in enumerate(
        plan.research_questions,
        start=1
    ):
        print(f"\n{index}. {question.question}")
        print(f"   Purpose: {question.purpose}")
    print("\nASSUMPTIONS")
    print("=" * 60)
    for assumption in plan.assumptions:
        print(f"- {assumption}")
    print("\nEXCLUDED TOPICS")
    print("=" * 60)
    for topic in plan.excluded_topics:
        print(f"- {topic}")

if __name__ == "__main__":
    main()