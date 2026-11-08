from researchmate.researcher import run_research

def main():
    query = input(
        "\nWhat would you like ResearchMate to investigate?\n> "
    ).strip()
    print("\nResearchMate is investigating...\n")
    result = run_research(query)
    print("=" * 70)
    print("RESEARCH QUESTION")
    print("=" * 70)
    print(result.research_question)
    print("\nRESEARCH SUMMARY")
    print("=" * 70)
    print(result.summary)
    print("\nCOLLECTED SOURCES")
    print("=" * 70)
    for index, source in enumerate(result.sources, start=1):
        print(f"\n{index}. {source.title}")
        print(f"URL: {source.url}")
        print(f"Relevance: {source.score}")
    print("\n" + "=" * 70)
    print(f"Total searches: {result.total_searches}")
    print(f"Unique sources: {len(result.sources)}")

if __name__ == "__main__":
    main()