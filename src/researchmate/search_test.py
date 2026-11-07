from researchmate.search import search_web


def main():

    query = input("\nEnter a search query: ").strip()

    print("\nSearching the web...\n")

    result = search_web(
        query=query,
        max_results=5
    )

    print(f"Query: {result.query}")
    print(f"Sources found: {len(result.sources)}")

    print("\n" + "=" * 70)

    for index, source in enumerate(result.sources, start=1):

        print(f"\nSOURCE {index}")
        print("-" * 70)

        print(f"Title: {source.title}")
        print(f"URL: {source.url}")
        print(f"Relevance score: {source.score}")

        print("\nSnippet:")
        print(source.snippet)

        print("\nExtracted content preview:")
        print(source.content[:1000])

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()