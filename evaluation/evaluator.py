import re


def extract_citations(answer):
    """
    Extract citations such as:
    [Page 1]
    [Page 5]
    """

    citations = re.findall(
        r"\[Page\s+(\d+)\]",
        answer,
        re.IGNORECASE
    )

    return [int(page) for page in citations]


def evaluate_citations(
    answer,
    retrieved_documents
):
    """
    Evaluate whether citations in the answer
    are supported by the retrieved documents.
    """

    cited_pages = extract_citations(answer)

    retrieved_pages = [
        document.metadata.get("page_number")
        for document in retrieved_documents
    ]

    verified_pages = []
    unsupported_pages = []

    for page in cited_pages:

        if page in retrieved_pages:

            if page not in verified_pages:
                verified_pages.append(page)

        else:

            if page not in unsupported_pages:
                unsupported_pages.append(page)

    # Calculate citation score

    total_unique_citations = len(
        set(cited_pages)
    )

    verified_count = len(
        verified_pages
    )

    if total_unique_citations > 0:

        citation_score = (
            verified_count /
            total_unique_citations
        )

    else:

        citation_score = 0.0

    return {
        "cited_pages": cited_pages,
        "retrieved_pages": retrieved_pages,
        "verified_pages": verified_pages,
        "unsupported_pages": unsupported_pages,
        "citation_score": citation_score
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Evaluation module loaded successfully!")

    print("\nThis module will evaluate:")
    print("1. Citation correctness")
    print("2. Retrieved source support")