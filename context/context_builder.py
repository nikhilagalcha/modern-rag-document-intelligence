from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ==================================================
# STEP 1: Load PDF
# ==================================================

loader = PyPDFDirectoryLoader("documents")
documents = loader.load()

print("Pages loaded:", len(documents))


# ==================================================
# STEP 2: Add metadata
# ==================================================

for document in documents:

    document.metadata["file_type"] = "PDF"

    document.metadata["document_id"] = document.metadata.get(
        "source", ""
    )

    document.metadata["page_number"] = (
        document.metadata.get("page", 0) + 1
    )


# ==================================================
# STEP 3: Create chunks
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# ==================================================
# STEP 4: Create BM25 index
# ==================================================

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

print("BM25 index created!")


# ==================================================
# STEP 5: Load vector database
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

print("Vector database loaded!")


# ==================================================
# STEP 6: User query
# ==================================================

query = "What are the cybersecurity challenges in software development?"

print("\nQuery:", query)


# ==================================================
# STEP 7: Vector search
# ==================================================

vector_results = vectorstore.similarity_search(
    query,
    k=5
)

print("Vector search completed!")


# ==================================================
# STEP 8: BM25 search
# ==================================================

query_tokens = query.lower().split()

bm25_results = bm25.get_top_n(
    query_tokens,
    chunks,
    n=5
)

print("BM25 search completed!")


# ==================================================
# STEP 9: RRF hybrid retrieval
# ==================================================

rrf_scores = {}

k = 60


for rank, document in enumerate(
    vector_results,
    start=1
):

    text = document.page_content

    if text not in rrf_scores:

        rrf_scores[text] = {
            "score": 0,
            "document": document
        }

    rrf_scores[text]["score"] += 1 / (k + rank)


for rank, document in enumerate(
    bm25_results,
    start=1
):

    text = document.page_content

    if text not in rrf_scores:

        rrf_scores[text] = {
            "score": 0,
            "document": document
        }

    rrf_scores[text]["score"] += 1 / (k + rank)


hybrid_results = sorted(
    rrf_scores.values(),
    key=lambda x: x["score"],
    reverse=True
)


# ==================================================
# STEP 10: Select candidates
# ==================================================

candidate_documents = [
    item["document"]
    for item in hybrid_results[:5]
]

print(
    "\nCandidates for reranking:",
    len(candidate_documents)
)


# ==================================================
# STEP 11: Load reranker
# ==================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranking model loaded!")


# ==================================================
# STEP 12: Create query-document pairs
# ==================================================

pairs = [
    [query, document.page_content]
    for document in candidate_documents
]


# ==================================================
# STEP 13: Rerank
# ==================================================

scores = reranker.predict(pairs)

reranked_results = list(
    zip(candidate_documents, scores)
)

reranked_results = sorted(
    reranked_results,
    key=lambda x: x[1],
    reverse=True
)


# ==================================================
# STEP 14: Select final context
# ==================================================

TOP_K = 3

final_documents = reranked_results[:TOP_K]

print(
    "\nFinal documents selected:",
    len(final_documents)
)


# ==================================================
# STEP 15: Build context
# ==================================================

context_parts = []

for i, (document, score) in enumerate(
    final_documents,
    start=1
):

    page_number = document.metadata.get(
        "page_number",
        "Unknown"
    )

    source = document.metadata.get(
        "file_name",
        document.metadata.get(
            "source",
            "Unknown"
        )
    )

    context_part = f"""
SOURCE {i}
File: {source}
Page: {page_number}
Reranker Score: {float(score):.4f}

CONTENT:
{document.page_content}
"""

    context_parts.append(
        context_part.strip()
    )


context = "\n\n" + "\n\n".join(
    context_parts
)


# ==================================================
# STEP 16: Display final context
# ==================================================

print("\n")
print("=" * 70)
print("FINAL CONTEXT FOR LLM")
print("=" * 70)

print(context)