import json
import os
from datetime import datetime


LOG_FILE = "rag_logs.json"


# ============================================================
# 1. SAVE RAG LOG
# ============================================================

def save_rag_log(
    question,
    retrieved_pages,
    reranker_scores,
    answer,
    citation_score,
    retrieval_precision,
    response_time
):

    log_entry = {

        "timestamp": datetime.now().isoformat(),

        "question": question,

        "retrieved_pages": retrieved_pages,

        "reranker_scores": reranker_scores,

        "answer": answer,

        "citation_score": citation_score,

        "retrieval_precision": retrieval_precision,

        "response_time_seconds": response_time
    }


    # Load existing logs
    if os.path.exists(LOG_FILE):

        with open(LOG_FILE, "r") as file:

            logs = json.load(file)

    else:

        logs = []


    # Add new log
    logs.append(log_entry)


    # Save logs
    with open(LOG_FILE, "w") as file:

        json.dump(
            logs,
            file,
            indent=4
        )


    print(
        "\nRAG log saved successfully!"
    )


# ============================================================
# 2. TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RAG OBSERVABILITY TEST")
    print("=" * 60)


    save_rag_log(

        question="What are the cybersecurity challenges in software development?",

        retrieved_pages=[1, 5, 1, 12, 11],

        reranker_scores=[
            6.4444,
            6.3925,
            4.7869,
            4.6066,
            -2.0768
        ],

        answer="Software vulnerabilities are an important cybersecurity challenge [Page 5].",

        citation_score=1.0,

        retrieval_precision=0.60,

        response_time=2.45
    )


    print(
        "\nLog file:",
        LOG_FILE
    )