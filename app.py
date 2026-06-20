from dotenv import load_dotenv
load_dotenv()

import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import InMemoryVectorStore

st.title("PDF Q&A")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        pdf_path = f.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    vector_db = InMemoryVectorStore.from_documents(
        docs,
        embeddings
    )

    question = st.text_input("Ask a question")

    if question:

        relevant_docs = vector_db.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            doc.page_content for doc in relevant_docs
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash"
        )

        response = llm.invoke(
            f"""
            Answer the question using the context below.

            Context:
            {context}

            Question:
            {question}
            """
        )

        st.write("### Answer")
        st.write(response.content)