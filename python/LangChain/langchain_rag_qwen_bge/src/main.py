from .rag_chain import build_rag_chain


def main():
    qa = build_rag_chain()
    print("\n🌙 Local RAG QA started. Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.strip().lower() == "exit":
            break

        result = qa(query)

        print("\nAnswer:", result["result"])
        print("\nSources:")
        for doc in result["source_documents"]:
            print("-", doc.page_content[:100], "...")
        print()


if __name__ == "__main__":
    main()
