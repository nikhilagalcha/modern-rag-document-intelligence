from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
# STEP 3: Create the text splitter
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# --------------------------------------------------
# STEP 4: Split documents into chunks
# --------------------------------------------------

chunks = text_splitter.split_documents(documents)


# --------------------------------------------------
# STEP 5: Display results
# --------------------------------------------------

print("Total chunks created:", len(chunks))


# Show the first 3 chunks
for i, chunk in enumerate(chunks[:3]):

    print("\n" + "=" * 60)
    print("CHUNK", i + 1)
    print("=" * 60)

    print("Text:")
    print(chunk.page_content[:1000])

    print("\nMetadata:")
    print(chunk.metadata)