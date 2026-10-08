# 📚 Modern RAG-Based Document Intelligence System

A production-oriented **Retrieval-Augmented Generation (RAG)** system for asking questions about PDF documents.

This project follows a modern RAG architecture with document ingestion, parsing, metadata enrichment, chunking, embeddings, hybrid retrieval, reranking, context assembly, LLM generation, citation verification, evaluation, document lifecycle tracking, and observability.

---

## 🚀 Project Overview

Traditional RAG systems often use only:

**PDF → Embeddings → Vector Search → LLM**

This project implements a more complete pipeline:

**Document → Ingestion → Parsing → Metadata → Chunking → Embeddings → Hybrid Retrieval → Reranking → Context Assembly → Gemini → Citation Verification → Evaluation → Logging**

The goal is to build a RAG system that is more reliable, explainable, and suitable for real-world applications.

---

## 🧠 RAG Architecture

```text
                         ┌─────────────────────┐
                         │      PDF Upload     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Ingestion       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Parsing / Cleaning  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Metadata Enrichment │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Chunking       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────┐
                  │          Indexing Layer           │
                  │                                  │
                  │  Sentence Transformers           │
                  │  ChromaDB                         │
                  │  BM25                             │
                  └───────────────┬──────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │     Hybrid Retrieval        │
                    │  Vector Search + BM25 + RRF │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       Cross-Encoder         │
                    │          Reranking           │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      Context Assembly       │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       Gemini LLM             │
                    │      Answer Generation       │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   Citation Verification     │
                    └──────────────┬──────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
             ┌───────────────┐          ┌────────────────┐
             │   Evaluation  │          │  Observability │
             └───────────────┘          └────────────────┘
```

---

## ✨ Key Features

- 📄 PDF document ingestion
- 🧹 PDF text cleaning and normalization
- 🏷️ Metadata enrichment
- ✂️ Recursive text chunking
- 🔢 Sentence Transformer embeddings
- 🗄️ Persistent ChromaDB vector database
- 🔎 Vector similarity search
- 🔤 BM25 keyword retrieval
- 🔀 Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- 🎯 Cross-Encoder reranking
- 🧠 Context assembly
- 🤖 Gemini-powered answer generation
- 📌 Page-based citation generation
- ✅ Citation verification
- 📊 Retrieval evaluation
- 📈 Groundedness/citation evaluation
- 🕒 Response-time observability
- 🧾 RAG execution logging
- 🔄 Document lifecycle detection using SHA-256 hashes
- ♻️ Persistent vector-index reuse for unchanged documents
- 🌐 Streamlit user interface

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| UI | Streamlit |
| PDF Processing | PyPDF |
| RAG Framework | LangChain |
| Embeddings | Sentence Transformers |
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Keyword Search | BM25 |
| Hybrid Retrieval | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Reranker Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Google Gemini |
| Environment Variables | python-dotenv |
| Evaluation | Custom evaluation modules |
| Logging | JSON-based observability logs |

---

## 📁 Project Structure

```text
AI_PDF_RAG/
│
├── documents/
│   └── Research paper.pdf1.pdf
│
├── uploaded_documents/
│
├── ingestion/
│   ├── loader.py
│   ├── parser.py
│   ├── metadata.py
│   ├── document_hash.py
│   └── document_manager.py
│
├── chunking/
│   └── chunker.py
│
├── embeddings/
│   ├── embedder.py
│   └── vector_store.py
│
├── retrieval/
│   ├── bm25_search.py
│   └── hybrid_search.py
│
├── reranking/
│   └── reranker.py
│
├── context/
│   └── context_builder.py
│
├── generation/
│   ├── generator.py
│   ├── rag_generator.py
│   └── citation_checker.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── retrieval_evaluator.py
│   ├── retrieval_test.py
│   ├── groundedness_evaluator.py
│   └── rag_logger.py
│
├── vectorstore/
│
├── document_registry.json
├── rag_logs.json
├── app.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI_PDF_RAG
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can also run:

```powershell
.venv\Scripts\activate
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# 🔐 API Key Configuration

This project uses the Google Gemini API for answer generation.

Create a file named:

```text
.env
```

Add:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Replace:

```text
YOUR_GEMINI_API_KEY
```

with your actual API key.

### ⚠️ Important

Never upload your `.env` file to GitHub.

The `.gitignore` file should contain:

```text
.env
.venv/
__pycache__/
vectorstore/
```

---

# ▶️ Running the Application

From the project root:

```powershell
streamlit run app.py
```

Streamlit will start the application in your browser.

Typical local address:

```text
http://localhost:8501
```

---

# 📄 How to Use the Application

1. Start the Streamlit application.
2. Upload a PDF document.
3. Enter a question about the document.
4. The system processes the document.
5. Relevant chunks are retrieved.
6. Hybrid retrieval combines vector and keyword search.
7. Cross-Encoder reranks the results.
8. The best context is sent to Gemini.
9. Gemini generates an answer.
10. Citations are checked against retrieved pages.
11. Evaluation and performance information is displayed.
12. The RAG execution is logged.

---

# 🔍 Example RAG Flow

Example question:

```text
What are the cybersecurity challenges in software development?
```

The system performs:

```text
Question
   ↓
Vector Retrieval
   ↓
BM25 Retrieval
   ↓
Hybrid RRF
   ↓
Cross-Encoder Reranking
   ↓
Top Relevant Chunks
   ↓
Context Assembly
   ↓
Gemini
   ↓
Answer + Page Citations
   ↓
Citation Verification
```

---

# 🔀 Hybrid Retrieval

The system combines two retrieval techniques.

## 1. Vector Search

Uses semantic similarity between:

```text
User Question
```

and:

```text
Document Chunks
```

Embeddings are generated using:

```text
all-MiniLM-L6-v2
```

---

## 2. BM25

BM25 provides keyword-based retrieval.

This is useful when the question contains:

- technical terms
- abbreviations
- specific names
- exact keywords
- security standards
- framework names

---

## 3. Reciprocal Rank Fusion

The vector and BM25 results are combined using:

```text
RRF
```

This helps improve retrieval by combining semantic and keyword-based search.

---

# 🎯 Cross-Encoder Reranking

After hybrid retrieval, candidate documents are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker evaluates the relationship between:

```text
Question + Retrieved Document
```

and produces a relevance score.

The highest-scoring documents are selected for context assembly.

---

# 📌 Citation Verification

The generated answer contains page citations such as:

```text
[Page 1]
[Page 5]
[Page 12]
```

The application verifies whether cited pages exist in the retrieved evidence.

Example:

```text
Cited Pages: 5
Verified Pages: 5
Unsupported Pages: 0
Citation Score: 100%
```

This provides a basic verification layer against unsupported citations.

---

# 📊 Evaluation

The project includes evaluation components for retrieval and answer grounding.

## Retrieval Precision

Example:

```text
Retrieved pages:
[1, 5, 1, 12, 11]

Relevant pages:
[1, 5]

Precision@5:
0.60
```

This means 60% of the top-five retrieved results were relevant according to the test set.

---

## Groundedness / Citation Check

The system checks whether cited pages were present in the retrieved evidence.

Example:

```text
Retrieved Pages:
[1, 5, 12]

Cited Pages:
[5, 1]

Verified Pages:
[5, 1]

Unsupported Pages:
[]

Groundedness Score:
1.00
```

---

# ⏱️ Observability

The system records RAG execution information such as:

- PDF loading time
- cleaning time
- chunking time
- vector database time
- BM25 indexing time
- vector retrieval time
- BM25 retrieval time
- hybrid retrieval time
- reranking time
- context assembly time
- Gemini generation time
- citation verification time
- total execution time

Example:

```text
PDF Loading: 1.72 seconds
Cleaning + Metadata: 0.02 seconds
Chunking: 0.03 seconds
Vector Database: 12.74 seconds
BM25 Index: 0.01 seconds
Vector Retrieval: 0.12 seconds
Reranking: 1.78 seconds
Gemini Generation: 100.61 seconds
Citation Verification: 0.00 seconds
Total: 117.03 seconds
```

Execution logs are stored in:

```text
rag_logs.json
```

---

# 🔄 Document Lifecycle Management

The project uses SHA-256 hashing to detect whether a document is:

```text
NEW
CHANGED
UNCHANGED
```

Example:

```text
DOCUMENT LIFECYCLE STATUS
============================================================

NEW DOCUMENTS:

CHANGED DOCUMENTS:

UNCHANGED DOCUMENTS:
- Research paper.pdf1.pdf
```

For unchanged documents, the existing ChromaDB vector index can be reused instead of rebuilding the complete index.

This reduces unnecessary embedding and indexing work.

---

# 🗄️ Persistent Vector Database

The project uses ChromaDB with persistent storage.

Vector data is stored in:

```text
vectorstore/
```

For an unchanged document, the application can reuse the existing vector index.

This demonstrates a more realistic production-style RAG workflow compared with rebuilding embeddings on every question.

---

# 🔒 Security

The project follows basic security practices:

- API keys are stored in `.env`
- `.env` is excluded from Git
- Virtual environments are excluded from Git
- Generated vector data is excluded from Git
- No API keys are hard-coded in Python files

---

# 🧪 Testing and Evaluation

The project contains evaluation modules for:

```text
Retrieval
Groundedness
Citation Verification
RAG Logging
Document Lifecycle
```

Example retrieval evaluation:

```text
Question:
What are the cybersecurity challenges in software development?

Precision@5:
0.60
```

---

# 📈 Current Project Performance

For the sample 14-page research paper:

```text
Pages Loaded:       14
Chunks Created:     62

Vector Results:      5
BM25 Results:        5
Hybrid Results:      8
Reranked Results:    5

Unique Pages:        4

Groundedness:        1.00
Citation Score:      100%
```

The exact execution time can vary depending on:

- CPU/GPU
- internet connection
- Gemini API response time
- model availability
- document size

---

# 🧩 Modern RAG Components Implemented

This project implements the following modern RAG components:

| Component | Status |
|---|---|
| Ingestion | ✅ |
| Parsing / Normalization | ✅ |
| Metadata Enrichment | ✅ |
| Chunking | ✅ |
| Indexing / Embeddings | ✅ |
| Retrieval | ✅ |
| Hybrid Retrieval | ✅ |
| Reranking | ✅ |
| Context Assembly | ✅ |
| Generation | ✅ |
| Verification / Citation Discipline | ✅ |
| Freshness / Document Lifecycle | ✅ |
| Evaluation | ✅ |
| Observability | ✅ |

---

# 🚀 Future Improvements

Possible improvements include:

- [ ] Semantic chunking
- [ ] Parent-child retrieval
- [ ] Query rewriting
- [ ] Multi-query retrieval
- [ ] Context compression
- [ ] Better claim-level groundedness evaluation
- [ ] RAGAS-based evaluation
- [ ] Automated evaluation datasets
- [ ] Document deletion and update workflows
- [ ] Multi-document knowledge base
- [ ] PostgreSQL / pgvector support
- [ ] Authentication
- [ ] Docker deployment
- [ ] CI/CD pipeline
- [ ] Cloud deployment
- [ ] Kubernetes deployment
- [ ] Advanced monitoring
- [ ] Streaming LLM responses

---

# 🎓 What I Learned

This project demonstrates practical understanding of:

- Retrieval-Augmented Generation
- Large Language Models
- Embeddings
- Vector databases
- Semantic search
- Keyword search
- Hybrid retrieval
- Reciprocal Rank Fusion
- Cross-Encoder reranking
- Context engineering
- Prompt engineering
- Citation verification
- Retrieval evaluation
- Document lifecycle management
- API integration
- Streamlit
- Python project architecture
- Observability and performance measurement

---

# 💼 Resume Description

### Modern RAG-Based Document Intelligence System

Built a production-oriented Retrieval-Augmented Generation system for PDF question answering using Python, LangChain, ChromaDB, Sentence Transformers, BM25, Cross-Encoder reranking, and Google Gemini. Implemented document ingestion, metadata enrichment, hybrid retrieval with Reciprocal Rank Fusion, context assembly, citation verification, retrieval evaluation, document lifecycle tracking using SHA-256 hashing, and RAG observability with Streamlit.

---

# ⭐ Resume Highlights

- Built an end-to-end **Modern RAG pipeline** for PDF question answering.
- Implemented **hybrid retrieval** using vector similarity + BM25.
- Used **Reciprocal Rank Fusion (RRF)** to combine retrieval results.
- Added **Cross-Encoder reranking** for improved relevance.
- Implemented **citation verification** and basic groundedness evaluation.
- Added **SHA-256 document lifecycle tracking**.
- Implemented **persistent ChromaDB indexing** for unchanged documents.
- Added **RAG observability and execution-time logging**.
- Built an interactive **Streamlit interface**.
- Integrated **Google Gemini** for answer generation.

---

# 📚 Learning Reference

The architecture was inspired by the concepts discussed in:

**Modern RAG in 2026: The Components That Actually Matter**

The implementation adapts those concepts into a practical, beginner-friendly project that demonstrates how modern RAG systems can be designed beyond simple vector search.

---

# 👨‍💻 Author

**Nikhil Agalcha**

MCA Graduate  
Interested in:

- AI / Generative AI
- RAG Systems
- Python
- Backend Development
- Full Stack Development
- Data Engineering
- DevOps

---

# 🎯 Project Objective

The main objective of this project is to demonstrate how a modern RAG system can be designed as a complete pipeline rather than simply connecting a vector database to an LLM.

The project focuses on:

```text
Better Retrieval
       +
Better Ranking
       +
Better Context
       +
Grounded Answers
       +
Evaluation
       +
Observability
```

This makes the project suitable as a portfolio project for **AI Engineer, Generative AI, RAG Engineer, Python Backend, and Software Engineer** roles.
