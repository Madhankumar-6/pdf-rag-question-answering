from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.pdf_loader import extract_text_from_pdf
from app.chunker import chunk_text, create_chunk_records
from app.rag import RAG


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "rag" not in st.session_state:
    st.session_state.rag = RAG()

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "filename" not in st.session_state:
    st.session_state.filename = None


# --------------------------------------------------
# Directories
# --------------------------------------------------

UPLOAD_DIR = Path("data/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Page Header
# --------------------------------------------------

st.title("📄 PDF RAG Question Answering")

st.write(
    "Upload a PDF and ask questions based on its content."
)


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("Process PDF"):

        file_path = (
            UPLOAD_DIR / uploaded_file.name
        )

        try:

            with st.spinner(
                "Processing PDF..."
            ):

                # Create a fresh RAG index
                # for every uploaded document
                st.session_state.rag = RAG()

                # Save PDF
                with open(
                    file_path,
                    "wb"
                ) as f:

                    f.write(
                        uploaded_file.getbuffer()
                    )

                # Extract text
                text = extract_text_from_pdf(
                    str(file_path)
                )

                if not text.strip():

                    st.error(
                        "Could not extract text "
                        "from this PDF."
                    )

                else:

                    # Chunk text
                    chunks = chunk_text(
                        text
                    )

                    # Create records
                    records = create_chunk_records(
                        chunks,
                        uploaded_file.name
                    )

                    # Build FAISS index
                    chunk_count = (
                        st.session_state.rag
                        .index_document(records)
                    )

                    # Save state
                    st.session_state.pdf_processed = True

                    st.session_state.chunk_count = (
                        chunk_count
                    )

                    st.session_state.filename = (
                        uploaded_file.name
                    )

                    st.success(
                        f"PDF processed successfully! "
                        f"{chunk_count} chunks created."
                    )

        except Exception as e:

            st.error(
                f"Error processing PDF: {e}"
            )


# --------------------------------------------------
# Processing Status
# --------------------------------------------------

if st.session_state.pdf_processed:

    st.success(
        f"PDF ready: "
        f"{st.session_state.filename} "
        f"({st.session_state.chunk_count} chunks)"
    )

    if st.button("Clear Document"):

        st.session_state.rag = RAG()

        st.session_state.pdf_processed = False

        st.session_state.chunk_count = 0

        st.session_state.filename = None

        st.rerun()


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

st.divider()

st.header("Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="e.g. What is supervised learning?"
)


if st.button(
    "Ask",
    disabled=not st.session_state.pdf_processed
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Searching the document..."
            ):

                result = (
                    st.session_state.rag.ask(
                        question
                    )
                )


            # ------------------------------------------
            # Answer
            # ------------------------------------------

            st.subheader("Answer")

            st.write(
                result["answer"]
            )


            # ------------------------------------------
            # Sources
            # ------------------------------------------

            sources = result.get(
                "sources",
                []
            )

            if sources:

                st.subheader("Sources")

                for source in sources:

                    source_name = source.get(
                        "source",
                        st.session_state.filename
                    )

                    chunk_id = source.get(
                        "chunk_id",
                        "N/A"
                    )

                    score = source.get(
                        "score",
                        0
                    )

                    with st.expander(
                        f"{source_name} | "
                        f"Chunk {chunk_id} | "
                        f"Score {score:.3f}"
                    ):

                        st.write(
                            source.get(
                                "text",
                                ""
                            )
                        )

        except Exception as e:

            st.error(
                f"Error generating answer: {e}"
            )