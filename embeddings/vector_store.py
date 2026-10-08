from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


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
# STEP 3: Split the documents into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# --------------------------------------------------
# STEP 4: Load the embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# --------------------------------------------------
# STEP 5: Create the vector database
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vectorstore"
)

print("Vector database created successfully!")


# --------------------------------------------------
# STEP 6: Test the database
# --------------------------------------------------

results = vectorstore.similarity_search(
    "What are cybersecurity challenges in software development?",
    k=3
)

print("\nTop 3 relevant chunks:\n")

for i, result in enumerate(results):

    print("=" * 60)
    print("RESULT", i + 1)
    print("=" * 60)

    print(result.page_content[:500])

    print("\nPage:", result.metadata.get("page_number"))