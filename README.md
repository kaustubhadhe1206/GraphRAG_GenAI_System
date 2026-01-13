# GraphRAG_GenAI_System
Got it. This is a **clean, low-fluff, engineer-written README**, not “marketing GenAI”.
You can paste this **as-is**.

---

# GraphRAG GenAI System

GraphRAG GenAI System is a **graph-orchestrated Retrieval-Augmented Generation (RAG) application** built using LangGraph.
It routes user queries across multiple data sources (vector database, Wikipedia, web search, and YouTube transcripts) based on query intent.

The system supports **optional document ingestion**, meaning it works both as:

* a **document-grounded RAG system**, and
* a **tool-augmented QA system** when no documents are provided.

The application is deployed on **Streamlit Cloud**.

---

## What the system does

* Accepts user questions via a Streamlit UI
* Optionally ingests PDFs and stores embeddings in AstraDB
* Uses a graph to decide **how a query should be answered**
* Executes only the required tools for each query
* Returns a grounded response based on the selected route

---

## Architecture Overview

The application uses a **LangGraph state machine** with conditional routing.

### Execution Graph

<img width="712" height="589" alt="Screenshot 2026-01-12 185210" src="https://github.com/user-attachments/assets/89722114-56a5-42a0-a908-945f3d7a3ec1" />

---

## Routing Logic

For every user query, the system follows this logic:

1. **Vector Store Check**

   * If semantically similar content exists in the vector database → use RAG
2. **Router Decision**

   * YouTube link present → YouTube summarization
   * General factual knowledge → Wikipedia
   * Time-sensitive or changing information → Web search

Routing decisions are made by the LLM using a constrained prompt.

---

## Document Processing Flow

PDF ingestion is optional and explicit.

1. User uploads one or more PDFs
2. Documents are loaded using `PyPDFLoader`
3. Text is chunked using `RecursiveCharacterTextSplitter`
4. Embeddings are generated via HuggingFace models
5. Chunks are stored in AstraDB (Cassandra)
6. During queries, relevant chunks are retrieved and passed to the LLM

If no documents are uploaded, the system still functions using external tools.

---

## Tools and Models

* **LLM:** Google Gemini (via LangChain)
* **Embeddings:** HuggingFace MiniLM
* **Vector Store:** AstraDB (Cassandra)
* **Graph Orchestration:** LangGraph
* **Search Tools:** Wikipedia, DuckDuckGo
* **YouTube Processing:** yt-dlp + AssemblyAI
* **UI:** Streamlit

---

## Streamlit Application

The UI provides:

* Optional PDF upload
* Manual document ingestion trigger
* Vector store reset option
* Free-form user query input
* Answer display with optional retrieved context

The UI is intentionally minimal to keep execution predictable and debuggable.

---

## Deployment

The application is deployed on **Streamlit Cloud**, allowing it to run without local setup.
Environment variables are managed through Streamlit secrets.

---

## Why this project

This project was built to explore:

* Graph-based control flow for LLM applications
* Tool routing vs traditional chain-based RAG
* Practical integration of vector databases in production-style apps
* Failure-aware and conditional execution in GenAI systems

---

## Possible Extensions

* Conversation-level memory
* Source citation per response
* Async node execution
* Streaming responses
* Multi-document attribution
