# RAG Assignment - Document Question Answering System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based Document Question Answering System.

The application allows users to upload PDF/TXT documents, retrieve the most relevant information from the document using semantic similarity, and generate accurate answers using the Cohere language model.

## Features

- Upload PDF and TXT documents
- Automatic document loading and text chunking
- Generate embeddings using Sentence Transformer
- Retrieve top-3 relevant chunks using cosine similarity
- Display retrieved document chunks
- Generate answers using Cohere API
- Interactive Streamlit-based user interface

## Tech Stack

- Python
- Streamlit
- LangChain
- LangChain Community
- LangChain Text Splitters
- Sentence Transformers
- Scikit-learn
- Cohere API
- PyPDF

## Project Structure

```text
rag_assignment/
│
├── app.py              # Streamlit application
├── functions.py        # RAG pipeline functions
├── requirements.txt    # Required dependencies
└── README.md           # Project documentation
```

## Workflow

### 1. Document Upload and Processing

- User uploads a PDF or TXT document.
- The document is loaded using LangChain document loaders.
- The text is divided into smaller chunks using Recursive Character Text Splitter.

### 2. Embedding Generation

- Document chunks are converted into numerical embeddings using:

```
SentenceTransformer (all-MiniLM-L6-v2)
```

### 3. Semantic Search

- User query is converted into an embedding.
- Cosine similarity is calculated between the query embedding and document embeddings.
- The top 3 most relevant chunks are retrieved.

### 4. Answer Generation

- Retrieved chunks are combined to create the context.
- The context and user query are passed to Cohere API.
- Cohere generates an answer based only on the retrieved context.

## Installation

Clone the repository:

```bash
git clone https://github.com/vigyamishra43-eng/rag_assignment.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

## Usage

1. Enter your Cohere API Key in the sidebar.
2. Upload a PDF or TXT document.
3. Enter your question.
4. Click on the Search button.
5. View:
   - Top 3 relevant document chunks
   - Generated answer using Cohere API

## Screenshots

### Document Upload Interface

![Upload Screen](screenshots/upload1.png)
![Upload Screen](screenshots/upload2.png)
![Upload Screen](screenshots/upload3.png)



## Output
![Output Screen](screenshots/output.png)
The application displays:

- Relevant document chunks retrieved using semantic search
- AI-generated answers using Cohere based on the provided context

## Author

**Vigya Mishra**  
B.Tech Computer Science Engineering
