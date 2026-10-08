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
# STEP 5: Load Vector Database
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
# STEP 7: Vector Search
# ==================================================

vector_results = vectorstore.similarity_search(
    query,
    k=5
)

print("Vector search completed!")


# ==================================================
# STEP 8: BM25 Search
# ==================================================

query_tokens = query.lower().split()

bm25_results = bm25.get_top_n(
    query_tokens,
    chunks,
    n=5
)

print("BM25 search completed!")


# ==================================================
# STEP 9: RRF Fusion
# ==================================================

rrf_scores = {}

k = 60


# Vector results

for rank, document in enumerate(vector_results, start=1):

    text = document.page_content

    if text not in rrf_scores:

        rrf_scores[text] = {
            "score": 0,
            "document": document
        }

    rrf_scores[text]["score"] += 1 / (k + rank)


# BM25 results

for rank, document in enumerate(bm25_results, start=1):

    text = document.page_content

    if text not in rrf_scores:

        rrf_scores[text] = {
            "score": 0,
            "document": document
        }

    rrf_scores[text]["score"] += 1 / (k + rank)


# Sort RRF results

hybrid_results = sorted(
    rrf_scores.values(),
    key=lambda x: x["score"],
    reverse=True
)


# ==================================================
# STEP 10: Select candidates for reranking
# ==================================================

candidate_documents = [
    item["document"]
    for item in hybrid_results[:5]
]

print(
    "\nCandidates sent to reranker:",
    len(candidate_documents)
)


# ==================================================
# STEP 11: Load Cross-Encoder
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
# STEP 13: Calculate relevance scores
# ==================================================

scores = reranker.predict(pairs)


# ==================================================
# STEP 14: Combine documents and scores
# ==================================================

reranked_results = list(
    zip(candidate_documents, scores)
)


# ==================================================
# STEP 15: Sort by reranker score
# ==================================================

reranked_results = sorted(
    reranked_results,
    key=lambda x: x[1],
    reverse=True
)


# ==================================================
# STEP 16: Display final results
# ==================================================

print("\n")
print("=" * 60)
print("FINAL RERANKED RESULTS")
print("=" * 60)


for i, (document, score) in enumerate(
    reranked_results,
    start=1
):

    print("\n" + "=" * 60)

    print("RANK:", i)

    print(
        "Reranker Score:",
        round(float(score), 4)
    )

    print(
        "Page:",
        document.metadata.get("page_number")
    )

    print("\nContent:")

    print(document.page_content[:700])