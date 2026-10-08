import os
from datetime import datetime

from langchain_community.document_loaders import PyPDFDirectoryLoader


# Load PDF documents
loader = PyPDFDirectoryLoader("documents")
documents = loader.load()

print("Pages loaded:", len(documents))


# Add our own metadata
for document in documents:

    # Get the PDF file path
    source = document.metadata.get("source", "")

    # Get only the file name
    file_name = os.path.basename(source)

    # Add custom metadata
    document.metadata["file_name"] = file_name
    document.metadata["file_type"] = "PDF"
    document.metadata["document_id"] = file_name
    document.metadata["page_number"] = document.metadata.get("page", 0) + 1
    document.metadata["ingested_at"] = datetime.now().isoformat()


# Display metadata of first page
if documents:

    print("\nEnriched metadata:")

    for key, value in documents[0].metadata.items():
        print(f"{key}: {value}")