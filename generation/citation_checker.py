import re


def extract_citations(answer):
    """
    Extract page citations such as:
    [Page 1]
    [Page 5]
    [Page 12]
    """

    citations = re.findall(
        r"\[Page\s+(\d+)\]",
        answer,
        re.IGNORECASE
    )

    return [int(page) for page in citations]


def verify_citations(answer, retrieved_documents):
    """
    Check whether cited pages actually exist
    in the retrieved documents.
    """

    cited_pages = extract_citations(answer)

    retrieved_pages = [
        document.metadata.get("page_number")
        for document in retrieved_documents
    ]

    verified = []
    unsupported = []

    for page in cited_pages:

        if page in retrieved_pages:
            verified.append(page)

        else:
            unsupported.append(page)

    return {
        "cited_pages": cited_pages,
        "retrieved_pages": retrieved_pages,
        "verified_pages": verified,
        "unsupported_pages": unsupported
    }


# ------------------------------------------------------------
import re

def extract_citations(answer):
    """
    Extract page citations such as:
    [Page 1]
    [Page 5]
    [Page 12]
    """

    citations = re.findall(
        r"\[Page\s+(\d+)\]",
        answer,
        re.IGNORECASE
    )

    return [int(page) for page in citations]


def verify_citations(answer, retrieved_documents):
    """
    Check whether cited pages exist
    in the retrieved documents.
    """

    cited_pages = extract_citations(answer)

    retrieved_pages = [
        document.metadata.get("page_number")
        for document in retrieved_documents
    ]

    verified = []
    unsupported = []

    for page in cited_pages:

        if page in retrieved_pages:
            if page not in verified:
                verified.append(page)

        else:
            if page not in unsupported:
                unsupported.append(page)

    return {
        "cited_pages": cited_pages,
        "retrieved_pages": retrieved_pages,
        "verified_pages": verified,
        "unsupported_pages": unsupported
    }