import os
import shutil
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.title("📚 RAG Assistant")

    st.markdown("---")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    create_db = st.button("📥 Create Knowledge Base")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

    st.markdown("---")

    st.info(
        """
        Built Using

        • LangChain

        • Mistral AI

        • ChromaDB

        • HuggingFace Embeddings

        • Streamlit
        """
    )

# ---------------- Title ---------------- #

st.title("📖 Chat With Your PDF")

st.caption("Upload a PDF and ask questions about it.")

# ---------------- Session ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Embeddings ---------------- #

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- Create DB ---------------- #

# ---------------- Create DB ---------------- #

if uploaded_pdf and create_db:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join("uploads", uploaded_pdf.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    with st.spinner("Creating Knowledge Base..."):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        # Create a unique database for each uploaded PDF
        db_path = os.path.join(
            "chroma_db",
            os.path.splitext(uploaded_pdf.name)[0]
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=db_path
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.db_path = db_path

    st.success("Knowledge Base Created Successfully!")
    
# ---------------- Load Vector DB ---------------- #


if "db_path" in st.session_state:

    vectorstore = Chroma(
        persist_directory=st.session_state.db_path,
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, reply:

"I could not find the answer in the document."
"""
            ),
            (
                "human",
                """
Context:

{context}

Question:

{question}
"""
            )
        ]
    )

# ---------------- Chat History ---------------- #

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ---------------- User Question ---------------- #

question = st.chat_input("Ask anything about your PDF...")

if question and "db_path" in st.session_state:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching..."):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = llm.invoke(final_prompt)

    answer = response.content

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander("📄 Source Chunks"):

            for i, doc in enumerate(docs, start=1):

                page = doc.metadata.get("page", "Unknown")

                st.markdown(f"### Chunk {i} | Page {page}")

                st.write(doc.page_content)

                st.divider()