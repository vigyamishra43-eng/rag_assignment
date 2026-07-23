import streamlit as st
import tempfile

from functions import (
    load_and_chunk_document,
    create_embeddings,
    search_chunks,
    generate_answer
)

st.title("📄 RAG Assignment")

st.write("Upload a PDF/TXT file and ask questions.")

# Sidebar for API Key
api_key = st.sidebar.text_input(
    "Enter your Cohere API Key",
    type="password"
)

# Upload file
uploaded_file = st.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"]
)

if uploaded_file is not None:

    # Save uploaded file temporarily
    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load document
    chunks = load_and_chunk_document(file_path)

    # Create embeddings
    embeddings = create_embeddings(chunks)

    st.success(f"Document loaded successfully! Total Chunks: {len(chunks)}")

    # User question
    question = st.text_input("Ask your question")

    if st.button("Search"):

        if not api_key:
            st.error("Please enter your Cohere API Key.")
            st.stop()

        if question.strip() == "":
            st.error("Please enter a question.")
            st.stop()

        # Search top 3 chunks
        results = search_chunks(
            question,
            chunks,
            embeddings,
            k=3
        )

        st.subheader("Top 3 Relevant Chunks")

        for i, chunk in enumerate(results):
            st.markdown(f"### Chunk {i+1}")
            st.write(chunk.page_content)

        # Build context
        context = "\n\n".join(
            [chunk.page_content for chunk in results]
        )

        # Generate answer
        answer = generate_answer(
            question,
            context,
            api_key
        )

        st.subheader("Generated Answer")
        st.write(answer)