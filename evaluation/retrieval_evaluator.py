def precision_at_k(retrieved_documents, relevant_pages, k):
    """
    Calculate Precision@K.

    retrieved_documents:
        Documents returned by the RAG system.

    relevant_pages:
        Pages that we know are relevant to the question.

    k:
        Number of top documents to evaluate.
    """

    top_documents = retrieved_documents[:k]

    relevant_count = 0

    for document in top_documents:

        page_number = document.metadata.get(
            "page_number"
        )

        if page_number in relevant_pages:
            relevant_count += 1

    if k == 0:
        return 0.0

    return relevant_count / k


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("RETRIEVAL EVALUATION TEST")
    print("=" * 60)

    # Example retrieved pages
    retrieved_pages = [1, 5, 1, 12, 11]

    # Pages manually identified as relevant
    relevant_pages = [1, 5]

    relevant_count = 0

    for page in retrieved_pages:

        if page in relevant_pages:
            relevant_count += 1

    k = 5

    precision = relevant_count / k

    print("\nRetrieved pages:")
    print(retrieved_pages)

    print("\nRelevant pages:")
    print(relevant_pages)

    print("\nPrecision@5:")
    print(f"{precision:.2f}")

    print(
        f"\nRetrieval Precision@5: "
        f"{precision * 100:.0f}%"
    )