from .rag_chain import build_rag_chain


def main():
    qa = build_rag_chain()
    print("\n🌙 本地 RAG 问答系统已启动。输入 'exit' 退出。\n")

    while True:
        query = input("您: ")
        if query.strip().lower() == "exit":
            break

        result = qa(query)

        print("\n回答:", result["result"])
        print("\n来源:")
        for doc in result["source_documents"]:
            print("-", doc.page_content[:100], "...")
        print()


if __name__ == "__main__":
    main()
