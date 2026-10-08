
import os
import json

from ingestion.document_hash import calculate_file_hash


DOCUMENT_FOLDER = "documents"
REGISTRY_FILE = "document_registry.json"


def check_documents():
    """
    Check documents and identify whether they are:
    NEW, CHANGED, or UNCHANGED.
    """

    # Load existing registry
    if os.path.exists(REGISTRY_FILE):

        with open(REGISTRY_FILE, "r") as file:
            registry = json.load(file)

    else:
        registry = {}


    new_documents = []
    changed_documents = []
    unchanged_documents = []


    # Check every document
    if not os.path.exists(DOCUMENT_FOLDER):
        return {
            "new": [],
            "changed": [],
            "unchanged": []
        }


    for file_name in os.listdir(DOCUMENT_FOLDER):

        file_path = os.path.join(
            DOCUMENT_FOLDER,
            file_name
        )

        # Ignore folders
        if not os.path.isfile(file_path):
            continue


        # Calculate current document hash
        current_hash = calculate_file_hash(
            file_path
        )

        # Get previous hash
        old_hash = registry.get(file_name)


        # Determine document status
        if old_hash is None:

            new_documents.append(file_name)

        elif old_hash != current_hash:

            changed_documents.append(file_name)

        else:

            unchanged_documents.append(file_name)


        # Update registry
        registry[file_name] = current_hash


    # Save updated registry
    with open(REGISTRY_FILE, "w") as file:

        json.dump(
            registry,
            file,
            indent=4
        )


    return {
        "new": new_documents,
        "changed": changed_documents,
        "unchanged": unchanged_documents
    }


# Run test only when this file is executed directly
if __name__ == "__main__":

    result = check_documents()


    print("\n" + "=" * 60)
    print("DOCUMENT LIFECYCLE STATUS")
    print("=" * 60)


    print("\nNEW DOCUMENTS:")

    for document in result["new"]:
        print("-", document)


    print("\nCHANGED DOCUMENTS:")

    for document in result["changed"]:
        print("-", document)


    print("\nUNCHANGED DOCUMENTS:")

    for document in result["unchanged"]:
        print("-", document)


    print("\nDocument registry updated successfully!")

