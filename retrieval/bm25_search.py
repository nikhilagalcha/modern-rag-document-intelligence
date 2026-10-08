from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# STEP 1: Load the PDF
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
# STEP 4: Tokenize the chunks
# --------------------------------------------------

tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]


# --------------------------------------------------
# STEP 5: Create BM25 index
# --------------------------------------------------

bm25 = BM25Okapi(tokenized_chunks)

print("BM25 index created successfully!")


# --------------------------------------------------
# STEP 6: Test BM25 search
# --------------------------------------------------

query = "OWASP 2020"

query_tokens = query.lower().split()

results = bm25.get_top_n(
    query_tokens,
    chunks,
    n=3
)


# --------------------------------------------------
# STEP 7: Display results
# --------------------------------------------------

print("\nTop 3 BM25 results:\n")

for i, result in enumerate(results):

    print("=" * 60)
    print("RESULT", i + 1)
    print("=" * 60)

    print(result.page_content[:500])

    print("\nPage:", result.metadata.get("page_number"))