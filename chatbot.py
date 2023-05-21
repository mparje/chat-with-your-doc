import os
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import (
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
    UnstructuredFileLoader,
)
from typing import List


class DocChatbot:
    llm: AzureChatOpenAI
    embeddings: OpenAIEmbeddings
    vector_db: FAISS
    chatchain: BaseConversationalRetrievalChain

    def __init__(self) -> None:
        # Init for OpenAI GPT-4 and Embeddings
        load_dotenv()
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview"
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
            temperature=0,
            openai_api_version="2023-03-15-preview"
        )

        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

    def init_chatchain(self, chain_type: str = "stuff") -> None:
        # Init for ConversationalRetrievalChain
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
            """Given the following conversation and a follow up question, rephrase the follow up question.
            Chat History:
            {chat_history}

            Follow Up Input:
            {question}

            Standalone Question:"""
        )

        # Stuff chain_type seems to work better than others
        self.chatchain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(),
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            chain_type=chain_type,
            return_source_documents=True,
            verbose=True,
        )

    @st.cache
    def get_answer_with_source(self, query, chat_history):
        result = self.chatchain(
            {"question": query, "chat_history": chat_history}, return_only_outputs=True
        )

        return result["answer"], result["source_documents"]

    def load_vector_db_from_local(self, path: str, index_name: str):
        self.vector_db = FAISS.load_local(path, self.embeddings, index_name)
        st.write(f"Loaded vector db from local: {path}/{index_name}")

    def save_vector_db_to_local(self, path: str, index_name: str):
        FAISS.save_local(self.vector_db, path, index_name)
        st.write("Vector db saved to local")

    @st.cache
    def init_vector_db_from_documents(self, file_list: List[str]):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        docs = []
        for file in file_list:
            st.write(f"Loading file: {file}")
            ext_name = os.path.splitext(file)[1]
            # print(ext_name)

            if ext_name == ".pptx":
                loader = UnstructuredPowerPointLoader(file)
            elif ext_name == ".docx":
                loader = UnstructuredWordDocumentLoader(file)
            elif ext_name == ".pdf":
                loader = PyPDFLoader(file)
            else:
                # Process .txt, .html
                loader = UnstructuredFileLoader(file)

            doc = loader.load_and_split(text_splitter)
            docs.extend(doc)
            st.write("Processed document: " + file)

        self.vector_db = FAISS.from_documents(docs, OpenAIEmbeddings(chunk_size=1))
        st.write("Generated embeddings and ingested to vector db.")


def main():
    st.title("DocChatbot")

    doc_chatbot = DocChatbot()

    # File upload
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)

    if uploaded_files:
        file_list = [file.name for file in uploaded_files]
        doc_chatbot.init_vector_db_from_documents(file_list)

        # Save vector DB
        if st.button("Save Vector DB"):
            doc_chatbot.save_vector_db_to_local(".", "vector_db")

    # Load vector DB
    vector_db_file = st.text_input("Vector DB File")
    vector_db_index = st.text_input("Vector DB Index")

    if st.button("Load Vector DB"):
        doc_chatbot.load_vector_db_from_local(vector_db_file, vector_db_index)

    # Chat
    query = st.text_input("Query")
    chat_history = st.text_area("Chat History")

    if st.button("Get Answer"):
        answer, source_docs = doc_chatbot.get_answer_with_source(query, chat_history)
        st.write("Answer:", answer)
        st.write("Source Documents:", source_docs)


if __name__ == "__main__":
    main()
