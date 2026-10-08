from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# STEP 1: Load PDF
# --------------------------------------------------

loader = PyPDFDirectoryLoader("documents")
documents = loader.load()

print("Pages loaded:", len(documents))


# --------------------------------------------------
# STEP 2: Add metadata
# --------------------------------------------------

for document in documents:

    document.metadata["file_type"] = "PDF"

    document.metadata["document_id"] = document.metadata.get(
        "source", ""
    )

    document.metadata["page_number"] = (
        document.metadata.get("page", 0) + 1
    )


# --------------------------------------------------
# STEP 3: Create chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# --------------------------------------------------
# STEP 4: Create BM25 index
# --------------------------------------------------

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

print("BM25 index created!")


# --------------------------------------------------
# STEP 5: Load Vector Database
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

print("Vector database loaded!")


# --------------------------------------------------
# STEP 6: User query
# --------------------------------------------------

query = "What are the cybersecurity challenges in software development?"


# --------------------------------------------------
# STEP 7: Vector Search
# --------------------------------------------------

vector_results = vectorstore.similarity_search(
    query,
    k=5
)

print("\nVector search completed!")


# --------------------------------------------------
# STEP 8: BM25 Search
# --------------------------------------------------

query_tokens = query.lower().split()

bm25_results = bm25.get_top_n(
    query_tokens,
    chunks,
    n=5
)

print("BM25 search completed!")


# --------------------------------------------------
# STEP 9: Reciprocal Rank Fusion (RRF)
# --------------------------------------------------

rrf_scores = {}

k = 60


# Add Vector Search rankings

for rank, document in enumerate(vector_results, start=1):

    document_text = document.page_content

    if document_text not in rrf_scores:
        rrf_scores[document_text] = {
            "score": 0,
            "document": document
        }

    rrf_scores[document_text]["score"] += 1 / (k + rank)


# Add BM25 rankings

for rank, document in enumerate(bm25_results, start=1):

    document_text = document.page_content

    if document_text not in rrf_scores:
        rrf_scores[document_text] = {
            "score": 0,
            "document": document
        }

    rrf_scores[document_text]["score"] += 1 / (k + rank)


# --------------------------------------------------
# STEP 10: Sort by RRF score
# --------------------------------------------------

final_results = sorted(
    rrf_scores.values(),
    key=lambda x: x["score"],
    reverse=True
)


# --------------------------------------------------
# STEP 11: Display final results
# --------------------------------------------------

print("\n" + "=" * 60)
print("HYBRID RETRIEVAL RESULTS")
print("=" * 60)

for i, item in enumerate(final_results[:5], start=1):

    document = item["document"]

    print("\n" + "=" * 60)
    print("RESULT", i)
    print("=" * 60)

    print("RRF Score:", round(item["score"], 6))

    print("Page:", document.metadata.get("page_number"))

    print("\nContent:")
    print(document.page_content[:700])