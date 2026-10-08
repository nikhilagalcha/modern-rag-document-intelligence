import os
import re

from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ============================================================
# 1. CITATION + EVALUATION FUNCTION
# ============================================================

def evaluate_citations(answer, retrieved_documents):

    # Find citations like [Page 1], [Page 5]
    cited_pages = re.findall(
        r"\[Page\s+(\d+)\]",
        answer,
        re.IGNORECASE
    )

    cited_pages = [
        int(page)
        for page in cited_pages
    ]

    # Get pages that were actually retrieved
    retrieved_pages = [
        document.metadata.get("page_number")
        for document in retrieved_documents
    ]

    verified_pages = []
    unsupported_pages = []

    # Check every citation
    for page in cited_pages:

        if page in retrieved_pages:

            if page not in verified_pages:
                verified_pages.append(page)

        else:

            if page not in unsupported_pages:
                unsupported_pages.append(page)

    # Calculate citation score
    total_unique_citations = len(
        set(cited_pages)
    )

    verified_count = len(
        verified_pages
    )

    if total_unique_citations > 0:

        citation_score = (
            verified_count /
            total_unique_citations
        )

    else:

        citation_score = 0.0

    return {
        "cited_pages": cited_pages,
        "retrieved_pages": retrieved_pages,
        "verified_pages": verified_pages,
        "unsupported_pages": unsupported_pages,
        "citation_score": citation_score
    }


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# ============================================================
# 3. INITIALIZE GEMINI
# ============================================================

client = genai.Client(
    api_key=api_key
)

print("Gemini initialized successfully!")


# ============================================================
# 4. LOAD PDF
# ============================================================

loader = PyPDFDirectoryLoader(
    "documents"
)

documents = loader.load()

print(
    "Pages loaded:",
    len(documents)
)


# ============================================================
# 5. ADD METADATA
# ============================================================

for document in documents:

    document.metadata["file_type"] = "PDF"

    document.metadata["document_id"] = (
        document.metadata.get("source", "")
    )

    document.metadata["page_number"] = (
        document.metadata.get("page", 0) + 1
    )


# ============================================================
# 6. CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(
    documents
)

print(
    "Total chunks:",
    len(chunks)
)


# ============================================================
# 7. BM25 KEYWORD SEARCH
# ============================================================

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_chunks
)

print(
    "BM25 index created!"
)


# ============================================================
# 8. LOAD VECTOR DATABASE
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

print(
    "Vector database loaded!"
)


# ============================================================
# 9. ASK USER QUESTION
# ============================================================

query = input(
    "\nAsk a question about your PDF: "
)

print(
    "\nQuestion:",
    query
)


# ============================================================
# 10. VECTOR SEARCH
# ============================================================

vector_results = vectorstore.similarity_search(
    query,
    k=5
)

print(
    "Vector search completed!"
)


# ============================================================
# 11. BM25 SEARCH
# ============================================================

query_tokens = query.lower().split()

bm25_results = bm25.get_top_n(
    query_tokens,
    chunks,
    n=5
)

print(
    "BM25 search completed!"
)


# ============================================================
# 12. HYBRID SEARCH USING RRF
# ============================================================

rrf_scores = {}

rrf_k = 60


# Vector search results
for rank, document in enumerate(
    vector_results,
    start=1
):

    document_id = id(document)

    if document_id not in rrf_scores:

        rrf_scores[document_id] = {
            "document": document,
            "score": 0
        }

    rrf_scores[document_id]["score"] += (
        1 / (rrf_k + rank)
    )


# BM25 results
for rank, document in enumerate(
    bm25_results,
    start=1
):

    document_id = id(document)

    if document_id not in rrf_scores:

        rrf_scores[document_id] = {
            "document": document,
            "score": 0
        }

    rrf_scores[document_id]["score"] += (
        1 / (rrf_k + rank)
    )


hybrid_results = sorted(
    rrf_scores.values(),
    key=lambda x: x["score"],
    reverse=True
)

print(
    "Hybrid retrieval completed!"
)


# ============================================================
# 13. SELECT TOP CANDIDATES
# ============================================================

candidates = [
    item["document"]
    for item in hybrid_results[:5]
]

print(
    "Candidates for reranking:",
    len(candidates)
)


# ============================================================
# 14. CROSS-ENCODER RERANKING
# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

pairs = [
    (
        query,
        document.page_content
    )
    for document in candidates
]

scores = reranker.predict(
    pairs
)


reranked_results = list(
    zip(
        candidates,
        scores
    )
)

reranked_results.sort(
    key=lambda x: x[1],
    reverse=True
)


# ============================================================
# 15. SELECT FINAL DOCUMENTS
# ============================================================

final_documents = (
    reranked_results[:3]
)

print(
    "Final documents selected:",
    len(final_documents)
)


# ============================================================
# 16. BUILD CONTEXT
# ============================================================

context_parts = []

for i, (document, score) in enumerate(
    final_documents,
    start=1
):

    page_number = document.metadata.get(
        "page_number"
    )

    context_parts.append(
        f"""
SOURCE {i}
Page: {page_number}
Reranker Score: {float(score):.4f}

Content:
{document.page_content}
"""
    )


context = "\n".join(
    context_parts
)


# ============================================================
# 17. CREATE RAG PROMPT
# ============================================================

prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question ONLY using the information
provided in the context below.

Do not use outside knowledge.

If the answer cannot be found in the provided context,
clearly say that the information is not available in
the provided document.

For every important claim, include a page citation
using this exact format:

[Page X]

Context:

{context}


User Question:

{query}


Instructions:

1. Answer clearly and accurately.
2. Use only the provided context.
3. Do not invent information.
4. Include page citations such as [Page 1] or [Page 5].
5. If the evidence is insufficient, say so.
"""


# ============================================================
# 18. GENERATE ANSWER WITH GEMINI
# ============================================================

print(
    "\nGenerating answer with Gemini...\n"
)

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=prompt
)


# ============================================================
# 19. DISPLAY RAG ANSWER
# ============================================================

answer = interaction.output_text

print("\n")
print("=" * 70)
print("RAG ANSWER")
print("=" * 70)

print(answer)


# ============================================================
# 20. PREPARE RETRIEVED DOCUMENTS
# ============================================================

retrieved_documents = [
    document
    for document, score in final_documents
]


# ============================================================
# 21. EVALUATE CITATIONS
# ============================================================

evaluation_result = evaluate_citations(
    answer,
    retrieved_documents
)


# ============================================================
# 22. DISPLAY EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION")
print("=" * 70)

print(
    "Cited pages:",
    evaluation_result["cited_pages"]
)

print(
    "Retrieved pages:",
    evaluation_result["retrieved_pages"]
)

print(
    "Verified pages:",
    evaluation_result["verified_pages"]
)

print(
    "Unsupported pages:",
    evaluation_result["unsupported_pages"]
)

print(
    "Citation Score:",
    f"{evaluation_result['citation_score']:.2f}"
)


# ============================================================
# 23. EVALUATION RESULT
# ============================================================

if evaluation_result["citation_score"] == 1.0:

    print(
        "\n✅ Citation quality: 100%"
    )

elif evaluation_result["citation_score"] >= 0.5:

    print(
        "\n⚠️ Citation quality: Partial"
    )

else:

    print(
        "\n❌ Citation quality: Low"
    )


# ============================================================
# 24. SOURCES USED
# ============================================================

print("\n")
print("=" * 70)
print("SOURCES USED")
print("=" * 70)

for i, (document, score) in enumerate(
    final_documents,
    start=1
):

    print(
        f"Source {i}: "
        f"Page {document.metadata.get('page_number')} "
        f"| Reranker Score: {float(score):.4f}"
    )