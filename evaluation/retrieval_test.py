import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ============================================================
# 1. LOAD PDF
# ============================================================

loader = PyPDFDirectoryLoader(
    "documents"
)

documents = loader.load()

print("Pages loaded:", len(documents))


# ============================================================
# 2. ADD METADATA
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
# 3. CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(
    documents
)

print("Total chunks:", len(chunks))


# ============================================================
# 4. CREATE BM25 INDEX
# ============================================================

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_chunks
)

print("BM25 index created!")


# ============================================================
# 5. LOAD VECTOR DATABASE
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

print("Vector database loaded!")


# ============================================================
# 6. QUESTION
# ============================================================

query = input(
    "\nEnter evaluation question: "
)

print(
    "\nQuestion:",
    query
)


# ============================================================
# 7. VECTOR SEARCH
# ============================================================

vector_results = vectorstore.similarity_search(
    query,
    k=5
)

print("Vector search completed!")


# ============================================================
# 8. BM25 SEARCH
# ============================================================

query_tokens = query.lower().split()

bm25_results = bm25.get_top_n(
    query_tokens,
    chunks,
    n=5
)

print("BM25 search completed!")


# ============================================================
# 9. HYBRID SEARCH — RRF
# ============================================================

rrf_scores = {}

rrf_k = 60


# Vector results
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


print("Hybrid retrieval completed!")


# ============================================================
# 10. SELECT TOP 5
# ============================================================

candidates = [
    item["document"]
    for item in hybrid_results[:5]
]


# ============================================================
# 11. RERANKING
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


print("Reranking completed!")


# ============================================================
# 12. DISPLAY RETRIEVED RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ACTUAL RAG RETRIEVAL RESULTS")
print("=" * 70)


retrieved_documents = []


for i, (document, score) in enumerate(
    reranked_results,
    start=1
):

    retrieved_documents.append(
        document
    )

    print(
        f"\nRank {i}"
    )

    print(
        "Page:",
        document.metadata.get("page_number")
    )

    print(
        "Reranker Score:",
        f"{float(score):.4f}"
    )

    print(
        "Content:",
        document.page_content[:300]
    )


# ============================================================
# 13. MANUAL RELEVANCE
# ============================================================

print("\n")
print("=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)

print(
    "\nEnter the relevant page numbers."
)

print(
    "Example: 1,5"
)

relevant_input = input(
    "Relevant pages: "
)


relevant_pages = [
    int(page.strip())
    for page in relevant_input.split(",")
    if page.strip().isdigit()
]


# ============================================================
# 14. PRECISION@5
# ============================================================

k = min(
    5,
    len(retrieved_documents)
)

top_documents = retrieved_documents[:k]

relevant_count = 0


for document in top_documents:

    page_number = document.metadata.get(
        "page_number"
    )

    if page_number in relevant_pages:

        relevant_count += 1


if k > 0:

    precision = (
        relevant_count / k
    )

else:

    precision = 0.0


# ============================================================
# 15. DISPLAY SCORE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

print(
    "\nRetrieved pages:"
)

print(
    [
        document.metadata.get("page_number")
        for document in retrieved_documents
    ]
)

print(
    "\nRelevant pages:"
)

print(
    relevant_pages
)

print(
    "\nPrecision@5:"
)

print(
    f"{precision:.2f}"
)

print(
    f"\nRetrieval Precision@5: "
    f"{precision * 100:.0f}%"
)