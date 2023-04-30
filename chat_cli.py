from chatbot import DocChatbot
import typer


VECTORDB_PATH = "./data/vector_store"
app = typer.Typer()
docChatbot = DocChatbot()

@app.command()
def ingest(doc_path: str, index_name : str):
    docChatbot.init_vector_db_from_documents(doc_path)
    docChatbot.save_vector_db_to_local(VECTORDB_PATH, index_name)

@app.command()
def chat(index_name : str = "index"):
    
    docChatbot.load_vector_db_from_local(VECTORDB_PATH, index_name)
    docChatbot.init_chatchain()

    chat_history = []

    while True:
        query = input("Question：")
        if query == "exit":
            break
        if query == "reset":
            chat_history = []
            continue

        result_answer, result_source = docChatbot.get_answer_with_source(query, chat_history)
        print(f"Q: {query}\nA: {result_answer}")
        print("Source Documents:")
        for doc in result_source:
            print(doc.metadata)
            # print(doc.page_content)

        # print(chat_history)
        chat_history.append((query, result_answer))
        # print(chat_history)

if __name__ == "__main__":
    app()