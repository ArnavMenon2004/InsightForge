from tools.web_search import WebSearchTool


def main():

    tool = WebSearchTool()

    results = tool.execute(
        query="Quantum Computing",
        max_results=5
    )

    print("\nSearch Results:\n")

    for source in results:
        print(source.model_dump())
        print("-" * 80)


if __name__ == "__main__":
    main()