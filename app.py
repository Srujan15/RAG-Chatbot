import os
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


# FREE local embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")
st.title("RAG PDF Chatbot")
st.caption("Upload a PDF and ask questions.")

# ----------------------------
# Helpers
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        pages.append({"page": page_num + 1, "text": text})
    full_text = "\n".join([p["text"] for p in pages])
    return full_text, pages

def build_vector_store(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    texts = []
    metadatas = []

    for p in pages:
        chunks = splitter.split_text(p["text"])
        for c in chunks:
            if c.strip():
                texts.append(c)
                metadatas.append({"page": p["page"]})


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vectorstore

def get_answer(query, vectorstore):
    docs = vectorstore.similarity_search(query, k=4)

    context = "\n\n".join(
        [f"(Page {d.metadata.get('page', '?')}) {d.page_content}" for d in docs]
    )

    system_prompt = (
        "You are a helpful assistant. Answer ONLY using the provided context.\n"
        "If the answer is not present in the context, say: 'I could not find that in the document.'\n"
        "Keep the answer clear and concise."
    )

    llm = ChatOllama(model="llama3", temperature=0)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]

    response = llm.invoke(messages)
    return response.content, docs

# ----------------------------
# Session State
# ----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# ----------------------------
# Sidebar: Upload PDF
# ----------------------------
with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if st.button("Reset Chat & Document"):
        st.session_state.vectorstore = None
        st.session_state.chat = []
        st.rerun()

# ----------------------------
# Build Vector Store
# ----------------------------
if pdf_file and st.session_state.vectorstore is None:
    with st.spinner("Extracting text and building index..."):
        full_text, pages = extract_text_from_pdf(pdf_file)

        if not full_text.strip():
            st.error("Could not extract text from PDF. Try another file.")
            st.stop()

        st.session_state.vectorstore = build_vector_store(pages)
        st.success("PDF indexed successfully! You can now ask questions..")

# ----------------------------
# Chat UI
# ----------------------------
st.subheader("Ask Questions")

query = st.text_input("Enter your question:")

if st.button("Ask") and query:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and index a PDF first.")
    else:
        with st.spinner("Generating answer..."):
            answer, docs = get_answer(query, st.session_state.vectorstore)

            st.session_state.chat.append({"role": "user", "text": query})
            st.session_state.chat.append({"role": "assistant", "text": answer, "sources": docs})

# ----------------------------
# Display Chat
# ----------------------------
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Bot:** {msg['text']}")

        if "sources" in msg:
            with st.expander("Sources"):
                for i, d in enumerate(msg["sources"], start=1):
                    page = d.metadata.get("page", "?")
                    st.markdown(f"**Source {i} (Page {page})**")
                    st.write(d.page_content[:800] + ("..." if len(d.page_content) > 800 else ""))
