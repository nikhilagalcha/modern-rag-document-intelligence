import os
import re
import time
import json
import hashlib

import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rank_bm25 import BM25Okapi

from sentence_transformers import CrossEncoder

from google import genai

from ingestion.document_hash import calculate_file_hash


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

UPLOAD_FOLDER = "uploaded_documents"
VECTORSTORE_FOLDER = "vectorstore"
REGISTRY_FILE = "document_registry.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Modern RAG Document Intelligence",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 Modern RAG Document Intelligence System")

st.markdown(
    """
    Upload a PDF and ask questions about its content.

    **Pipeline:**

    PDF → Ingestion → Cleaning → Metadata → Chunking →
    Embeddings → Vector Search + BM25 → Hybrid Retrieval →
    Reranking → Context Assembly → Gemini → Citations → Evaluation
    """
)


# ============================================================
# CHECK API KEY
# ============================================================

if not GEMINI_API_KEY:

    st.error(
        "GEMINI_API_KEY was not found. "
        "Please add it to your .env file."
    )

    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


embeddings = load_embedding_model()
reranker = load_reranker()


# ============================================================
# DOCUMENT LIFECYCLE
# ============================================================

def check_uploaded_document(
    file_path,
    filename
):

    current_hash = calculate_file_hash(
        file_path
    )

    if os.path.exists(REGISTRY_FILE):

        with open(
            REGISTRY_FILE,
            "r"
        ) as file:

            registry = json.load(file)

    else:

        registry = {}


    old_hash = registry.get(
        filename
    )


    if old_hash is None:

        status = "NEW"

    elif old_hash != current_hash:

        status = "CHANGED"

    else:

        status = "UNCHANGED"


    registry[filename] = current_hash


    with open(
        REGISTRY_FILE,
        "w"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )


    return status, current_hash


# ============================================================
# DOCUMENT ID
# ============================================================

def get_document_id(
    file_path
):

    file_hash = calculate_file_hash(
        file_path
    )

    document_id = hashlib.md5(
        file_hash.encode()
    ).hexdigest()[:16]

    return document_id


# ============================================================
# VECTOR DATABASE
# ============================================================

def create_or_load_vectorstore(
    chunks,
    file_path,
    document_status
):

    document_id = get_document_id(
        file_path
    )

    collection_name = (
        f"rag_{document_id}"
    )


    # --------------------------------------------------------
    # UNCHANGED DOCUMENT
    # --------------------------------------------------------

    if document_status == "UNCHANGED":

        try:

            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=VECTORSTORE_FOLDER
            )

            # Check whether collection actually contains data

            collection_count = (
                vectorstore._collection.count()
            )

            if collection_count > 0:

                return vectorstore

        except Exception:

            pass


    # --------------------------------------------------------
    # NEW / CHANGED DOCUMENT
    # --------------------------------------------------------

    # Create collection

    try:

        existing_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=VECTORSTORE_FOLDER
        )

        existing_store.delete_collection()

    except Exception:

        pass


    # Create new persistent vector database

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=VECTORSTORE_FOLDER
    )

    return vectorstore


# ============================================================
# PDF LOADING
# ============================================================

def load_pdf(
    pdf_path
):

    documents = []

    reader = PdfReader(
        pdf_path
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if not text:

            text = ""

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "page_number": page_number,
                    "source": os.path.basename(
                        pdf_path
                    )
                }
            )
        )

    return documents


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    text
):

    # Fix words broken across lines

    text = re.sub(
        r"-\s*\n\s*",
        "",
        text
    )

    # Replace line breaks with spaces

    text = re.sub(
        r"\s*\n\s*",
        " ",
        text
    )

    # Remove repeated spaces

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CLEAN DOCUMENTS + METADATA
# ============================================================

def clean_documents(
    documents
):

    cleaned_documents = []

    for document in documents:

        cleaned = clean_text(
            document.page_content
        )

        metadata = dict(
            document.metadata
        )

        metadata["file_name"] = metadata.get(
            "source",
            "unknown"
        )

        metadata["file_type"] = "PDF"

        metadata["document_id"] = metadata.get(
            "source",
            "unknown"
        )

        cleaned_documents.append(
            Document(
                page_content=cleaned,
                metadata=metadata
            )
        )

    return cleaned_documents


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    documents
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# ============================================================
# BM25
# ============================================================

def create_bm25(
    chunks
):

    tokenized_documents = [
        document.page_content.lower().split()
        for document in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    return bm25


def bm25_search(
    bm25,
    chunks,
    query,
    k=5
):

    tokenized_query = (
        query.lower().split()
    )

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indexes[:k]:

        results.append(
            {
                "document": chunks[index],
                "score": float(
                    scores[index]
                )
            }
        )

    return results


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(
    vectorstore,
    query,
    k=5
):

    results = (
        vectorstore.similarity_search_with_score(
            query,
            k=k
        )
    )

    formatted_results = []

    for document, score in results:

        formatted_results.append(
            {
                "document": document,
                "score": float(score)
            }
        )

    return formatted_results


# ============================================================
# HYBRID RETRIEVAL + RRF
# ============================================================

def hybrid_search(
    vector_results,
    bm25_results,
    k=60
):

    combined = {}

    # --------------------------------------------------------
    # VECTOR RESULTS
    # --------------------------------------------------------

    for rank, result in enumerate(
        vector_results,
        start=1
    ):

        document = result["document"]

        key = (
            document.page_content
        )

        if key not in combined:

            combined[key] = {
                "document": document,
                "score": 0
            }

        combined[key]["score"] += (
            1 / (k + rank)
        )


    # --------------------------------------------------------
    # BM25 RESULTS
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        document = result["document"]

        key = (
            document.page_content
        )

        if key not in combined:

            combined[key] = {
                "document": document,
                "score": 0
            }

        combined[key]["score"] += (
            1 / (k + rank)
        )


    results = list(
        combined.values()
    )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ============================================================
# RERANKING
# ============================================================

def rerank_documents(
    query,
    documents,
    top_k=5
):

    pairs = []

    for result in documents:

        pairs.append(
            [
                query,
                result["document"].page_content
            ]
        )


    if not pairs:

        return []


    scores = reranker.predict(
        pairs
    )


    reranked = []

    for result, score in zip(
        documents,
        scores
    ):

        reranked.append(
            {
                "document": result["document"],
                "score": float(score)
            }
        )


    reranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    return reranked[:top_k]


# ============================================================
# CONTEXT ASSEMBLY
# ============================================================

def build_context(
    reranked_documents,
    top_k=3
):

    selected_documents = (
        reranked_documents[:top_k]
    )

    context_parts = []

    for result in selected_documents:

        document = result["document"]

        page_number = document.metadata.get(
            "page_number",
            "Unknown"
        )

        context_parts.append(
            f"[Page {page_number}]\n"
            f"{document.page_content}"
        )


    context = "\n\n".join(
        context_parts
    )

    return context, selected_documents


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_answer(
    question,
    context
):

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided context.

Important rules:

1. Do not use outside knowledge.
2. If the answer cannot be found in the context, clearly say:
   "I could not find the answer in the provided document."
3. Give a concise but useful answer.
4. Cite the page number after each important claim.
5. Use citations in this exact format:
   [Page X]

Context:

{context}

User Question:

{question}
"""


    max_retries = 3

    for attempt in range(
        max_retries
    ):

        try:

            interaction = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt
            )

            return interaction.output_text


        except Exception as error:

            if attempt < max_retries - 1:

                wait_time = 2 ** attempt

                time.sleep(
                    wait_time
                )

            else:

                raise error


# ============================================================
# CITATION EXTRACTION
# ============================================================

def extract_citations(
    answer
):

    pattern = r"\[Page\s+(\d+)\]"

    pages = re.findall(
        pattern,
        answer,
        flags=re.IGNORECASE
    )

    return [
        int(page)
        for page in pages
    ]


# ============================================================
# CITATION VERIFICATION
# ============================================================

def verify_citations(
    answer,
    retrieved_documents
):

    cited_pages = extract_citations(
        answer
    )

    retrieved_pages = []

    for result in retrieved_documents:

        page = result["document"].metadata.get(
            "page_number"
        )

        if page is not None:

            retrieved_pages.append(
                int(page)
            )


    verified_pages = [
        page
        for page in cited_pages
        if page in retrieved_pages
    ]


    unsupported_pages = [
        page
        for page in cited_pages
        if page not in retrieved_pages
    ]


    if len(cited_pages) == 0:

        citation_score = 0.0

    else:

        citation_score = (
            len(verified_pages)
            /
            len(cited_pages)
        )


    return {
        "cited_pages": cited_pages,
        "retrieved_pages": retrieved_pages,
        "verified_pages": verified_pages,
        "unsupported_pages": unsupported_pages,
        "citation_score": citation_score
    }


# ============================================================
# RETRIEVAL PRECISION
# ============================================================

def calculate_precision(
    retrieved_documents,
    relevant_pages
):

    if not retrieved_documents:

        return 0.0


    retrieved_pages = []

    for result in retrieved_documents:

        page = result["document"].metadata.get(
            "page_number"
        )

        if page is not None:

            retrieved_pages.append(
                int(page)
            )


    relevant_count = sum(
        1
        for page in retrieved_pages
        if page in relevant_pages
    )


    return (
        relevant_count
        /
        len(retrieved_pages)
    )


# ============================================================
# OBSERVABILITY
# ============================================================

def save_rag_log(
    question,
    retrieved_documents,
    answer,
    citation_score,
    retrieval_precision,
    response_time
):

    log_entry = {

        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "question": question,

        "retrieved_pages": [
            result["document"].metadata.get(
                "page_number"
            )
            for result in retrieved_documents
        ],

        "reranker_scores": [
            result["score"]
            for result in retrieved_documents
        ],

        "answer": answer,

        "citation_score": citation_score,

        "retrieval_precision": retrieval_precision,

        "response_time_seconds": response_time
    }


    log_file = "rag_logs.json"


    if os.path.exists(
        log_file
    ):

        try:

            with open(
                log_file,
                "r"
            ) as file:

                logs = json.load(
                    file
                )

        except Exception:

            logs = []

    else:

        logs = []


    logs.append(
        log_entry
    )


    with open(
        log_file,
        "w"
    ) as file:

        json.dump(
            logs,
            file,
            indent=4
        )


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def run_rag(
    pdf_path,
    question,
    document_status
):

    timings = {}

    total_start = time.time()


    # --------------------------------------------------------
    # PDF LOADING
    # --------------------------------------------------------

    start = time.time()

    documents = load_pdf(
        pdf_path
    )

    timings["PDF Loading"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # CLEANING + METADATA
    # --------------------------------------------------------

    start = time.time()

    documents = clean_documents(
        documents
    )

    timings["Cleaning + Metadata"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    start = time.time()

    chunks = create_chunks(
        documents
    )

    timings["Chunking"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # VECTOR DATABASE
    # --------------------------------------------------------

    start = time.time()

    vectorstore = create_or_load_vectorstore(
        chunks,
        pdf_path,
        document_status
    )

    timings["Vector Database"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    start = time.time()

    bm25 = create_bm25(
        chunks
    )

    timings["BM25 Index"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # VECTOR RETRIEVAL
    # --------------------------------------------------------

    start = time.time()

    vector_results = vector_search(
        vectorstore,
        question,
        k=5
    )

    timings["Vector Retrieval"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # BM25 RETRIEVAL
    # --------------------------------------------------------

    start = time.time()

    bm25_results = bm25_search(
        bm25,
        chunks,
        question,
        k=5
    )

    timings["BM25 Retrieval"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # HYBRID RRF
    # --------------------------------------------------------

    start = time.time()

    hybrid_results = hybrid_search(
        vector_results,
        bm25_results
    )

    timings["Hybrid RRF"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # RERANKING
    # --------------------------------------------------------

    start = time.time()

    reranked_documents = rerank_documents(
        question,
        hybrid_results,
        top_k=5
    )

    timings["Reranking"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # CONTEXT ASSEMBLY
    # --------------------------------------------------------

    start = time.time()

    context, selected_documents = build_context(
        reranked_documents,
        top_k=3
    )

    timings["Context Assembly"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    start = time.time()

    answer = generate_answer(
        question,
        context
    )

    timings["Gemini Generation"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # CITATION VERIFICATION
    # --------------------------------------------------------

    start = time.time()

    citation_result = verify_citations(
        answer,
        selected_documents
    )

    timings["Citation Verification"] = (
        time.time() - start
    )


    # --------------------------------------------------------
    # TOTAL TIME
    # --------------------------------------------------------

    total_time = (
        time.time() - total_start
    )

    timings["Total"] = total_time


    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return {

        "answer": answer,

        "documents": documents,

        "chunks": chunks,

        "vector_results": vector_results,

        "bm25_results": bm25_results,

        "hybrid_results": hybrid_results,

        "reranked_documents": reranked_documents,

        "selected_documents": selected_documents,

        "citation_result": citation_result,

        "timings": timings,

        "document_status": document_status
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Pipeline")

    st.markdown(
        """
        **Components**

        1. 📥 Ingestion
        2. 🧹 Parsing
        3. 🏷️ Metadata
        4. ✂️ Chunking
        5. 🧠 Embeddings
        6. 🗄️ ChromaDB
        7. 🔎 BM25
        8. 🔀 Hybrid RRF
        9. 🎯 Reranking
        10. 🧩 Context Assembly
        11. 🤖 Gemini
        12. 📑 Citations
        13. 🔄 Lifecycle
        14. 📊 Evaluation
        15. 👁️ Observability
        """
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


if uploaded_file:

    # --------------------------------------------------------
    # SAVE PDF
    # --------------------------------------------------------

    pdf_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )


    with open(
        pdf_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # --------------------------------------------------------
    # DOCUMENT LIFECYCLE STATUS
    # --------------------------------------------------------

    document_status, document_hash = (
        check_uploaded_document(
            pdf_path,
            uploaded_file.name
        )
    )


    # --------------------------------------------------------
    # STATUS DISPLAY
    # --------------------------------------------------------

    if document_status == "NEW":

        st.success(
            "🆕 New document detected"
        )

    elif document_status == "CHANGED":

        st.warning(
            "🔄 Changed document detected — "
            "vector index will be rebuilt"
        )

    else:

        st.info(
            "♻️ Unchanged document detected — "
            "existing vector index will be reused"
        )


    st.caption(
        f"Document status: **{document_status}**"
    )

    st.caption(
        f"SHA-256: `{document_hash[:16]}...`"
    )


    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = st.text_input(
        "Ask a question about the PDF"
    )


    if question:

        with st.spinner(
            "Running Modern RAG pipeline..."
        ):

            try:

                result = run_rag(
                    pdf_path,
                    question,
                    document_status
                )


            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )

                st.stop()


        # ====================================================
        # ANSWER
        # ====================================================

        st.header("💬 Answer")

        st.write(
            result["answer"]
        )


        # ====================================================
        # CITATION VERIFICATION
        # ====================================================

        citation_result = (
            result["citation_result"]
        )


        st.header(
            "📑 Citation Verification"
        )


        citation_score = (
            citation_result[
                "citation_score"
            ]
        )


        if citation_score == 1.0:

            st.success(
                f"✅ Citation Score: "
                f"{citation_score * 100:.0f}%"
            )

        elif citation_score > 0:

            st.warning(
                f"⚠️ Citation Score: "
                f"{citation_score * 100:.0f}%"
            )

        else:

            st.warning(
                "⚠️ No valid citations found"
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Cited Pages",
                len(
                    citation_result[
                        "cited_pages"
                    ]
                )
            )


        with col2:

            st.metric(
                "Verified Pages",
                len(
                    citation_result[
                        "verified_pages"
                    ]
                )
            )


        with col3:

            st.metric(
                "Unsupported Pages",
                len(
                    citation_result[
                        "unsupported_pages"
                    ]
                )
            )


        # ====================================================
        # PIPELINE PERFORMANCE
        # ====================================================

        st.header(
            "⏱️ Pipeline Performance"
        )


        timings = result[
            "timings"
        ]


        for step, duration in timings.items():

            st.write(
                f"**{step}:** "
                f"{duration:.2f} seconds"
            )


        # ====================================================
        # RETRIEVAL INFORMATION
        # ====================================================

        st.header(
            "🔎 Retrieval Information"
        )


        st.write(
            f"**Pages loaded:** "
            f"{len(result['documents'])}"
        )

        st.write(
            f"**Chunks created:** "
            f"{len(result['chunks'])}"
        )

        st.write(
            f"**Vector results:** "
            f"{len(result['vector_results'])}"
        )

        st.write(
            f"**BM25 results:** "
            f"{len(result['bm25_results'])}"
        )

        st.write(
            f"**Hybrid results:** "
            f"{len(result['hybrid_results'])}"
        )

        st.write(
            f"**Reranked results:** "
            f"{len(result['reranked_documents'])}"
        )


        # ====================================================
        # RETRIEVED SOURCES
        # ====================================================

        st.header(
            "📚 Retrieved Sources"
        )


        for index, item in enumerate(
            result["reranked_documents"],
            start=1
        ):

            document = item[
                "document"
            ]

            score = item[
                "score"
            ]

            page = document.metadata.get(
                "page_number",
                "Unknown"
            )


            with st.expander(
                f"Source {index} — "
                f"Page {page} — "
                f"Score {score:.4f}"
            ):

                st.write(
                    document.page_content
                )


        # ====================================================
        # EVALUATION
        # ====================================================

        st.header(
            "📊 Evaluation"
        )


        # For demonstration, pages appearing
        # in the top retrieved results are used.

        retrieved_pages = []

        for item in result[
            "reranked_documents"
        ]:

            page = item[
                "document"
            ].metadata.get(
                "page_number"
            )

            if page is not None:

                retrieved_pages.append(
                    int(page)
                )


        # Basic retrieval metric

        unique_pages = set(
            retrieved_pages
        )


        st.write(
            f"**Unique retrieved pages:** "
            f"{len(unique_pages)}"
        )


        st.write(
            f"**Groundedness score:** "
            f"{citation_score:.2f}"
        )


        # ====================================================
        # OBSERVABILITY LOG
        # ====================================================

        save_rag_log(
            question=question,

            retrieved_documents=(
                result[
                    "reranked_documents"
                ]
            ),

            answer=result[
                "answer"
            ],

            citation_score=citation_score,

            retrieval_precision=(
                citation_score
            ),

            response_time=(
                timings["Total"]
            )
        )


        st.success(
            "📋 RAG execution logged successfully."
        )


        # ====================================================
        # DOCUMENT LIFECYCLE INFORMATION
        # ====================================================

        st.header(
            "🔄 Document Lifecycle"
        )


        if document_status == "UNCHANGED":

            st.success(
                "♻️ Existing ChromaDB vector "
                "index was reused."
            )

        elif document_status == "NEW":

            st.info(
                "🆕 New document indexed into "
                "ChromaDB."
            )

        elif document_status == "CHANGED":

            st.warning(
                "🔄 Document changed. "
                "A fresh ChromaDB index was created."
            )


        # ====================================================
        # PIPELINE SUMMARY
        # ====================================================

        st.header(
            "🏗️ Modern RAG Architecture"
        )


        st.code(
            """
PDF
 ↓
Document Ingestion
 ↓
Parsing & Cleaning
 ↓
Metadata Enrichment
 ↓
Smart Chunking
 ↓
Embeddings
 ↓
Persistent ChromaDB
 ↓
Dense Vector Search
 +
BM25 Sparse Search
 ↓
Hybrid Retrieval + RRF
 ↓
Cross-Encoder Reranking
 ↓
Context Assembly
 ↓
Gemini Generation
 ↓
Citation Verification
 ↓
Evaluation
 ↓
Observability
            """
        )

else:

    st.info(
        "👆 Upload a PDF document to start."
    )