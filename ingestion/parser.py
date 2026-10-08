import re
from langchain_community.document_loaders import PyPDFDirectoryLoader


# Load the PDF documents
loader = PyPDFDirectoryLoader("documents")
documents = loader.load()

print("Pages loaded:", len(documents))


# Function to clean extracted text
def clean_text(text):

    # Join words that are broken with a hyphen at a line break
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace line breaks with spaces
    text = re.sub(r"\s*\n\s*", " ", text)

    # Remove multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces from the beginning and end
    text = text.strip()

    return text


# Clean every page
for document in documents:
    document.page_content = clean_text(document.page_content)


# Display the cleaned first page
if documents:

    print("\nCleaned first page:")
    print(documents[0].page_content[:1500])

    print("\nMetadata:")
    print(documents[0].metadata)