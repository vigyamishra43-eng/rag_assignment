from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cohere

def load_and_chunk_document(file_path, chunk_size=300, overlap=50):

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = splitter.split_documents(documents)

    return chunks

def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):

    model = SentenceTransformer(model_name)

    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(texts)

    return embeddings.tolist()

def search_chunks(query, chunks, embeddings, k=3):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode([query])

    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = similarities.argsort()[-k:][::-1]

    top_chunks = [chunks[i] for i in top_indices]

    return top_chunks

def generate_answer(query, context, api_key):

    co = cohere.Client(api_key)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the given context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = co.chat(
        model="command-a-03-2025",
        message=prompt,
        temperature=0.3
    )

    return response.text