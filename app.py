import streamlit as st
from rag_pipeline import load_document, chunk_text, Retriever, generate_answer

st.set_page_config(page_title="Legal RAG System", layout="wide")

st.title("⚖️ Legal Document QA System (RAG)")

# ✅ Session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.chunks = None
    st.session_state.file_name = None
    st.session_state.messages = []

# 📄 Upload
uploaded_file = st.file_uploader("📄 Upload Legal Document (PDF/TXT)", type=["pdf", "txt"])

# ✅ Process document once
if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        with st.spinner("⏳ Processing document..."):
            text = load_document(uploaded_file)
            chunks = chunk_text(text)
            retriever = Retriever(chunks)

            st.session_state.retriever = retriever
            st.session_state.chunks = chunks
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = []

        st.success("✅ Document processed successfully!")

# 📊 Info
if st.session_state.chunks:
    st.write(f"📊 Total Chunks: {len(st.session_state.chunks)}")
    st.write(f"📄 Current File: {st.session_state.file_name}")

# 🔥 DISPLAY CHAT FIRST (IMPORTANT)
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):

            # 🔥 REMOVE CONFIDENCE LINE ONLY
            cleaned_text = "\n".join(
                line for line in msg["content"].split("\n")
                if "confidence" not in line.lower()
            )

            st.write(cleaned_text.strip())

# 🔥 INPUT ALWAYS AT BOTTOM
query = st.chat_input("Ask your legal question...")

# ✅ Handle query
if query and st.session_state.retriever:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate answer
    results = st.session_state.retriever.search(query)
    answer, _ = generate_answer(query, results)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # 🔥 FORCE RERUN → THIS IS THE KEY
    st.rerun()
