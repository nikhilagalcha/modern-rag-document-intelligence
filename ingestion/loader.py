from langchain_community.document_loaders import PyPDFDirectoryLoader

# Tell the loader where our PDF files are
loader = PyPDFDirectoryLoader("documents")

# Load all PDF files
documents = loader.load()

# Show how many pages were loaded
print("Total pages loaded:", len(documents))

# Show information about the first page
if documents:
    print("\nFirst page content:")
    print(documents[0].page_content[:1000])

    print("\nMetadata:")
    print(documents[0].metadata)