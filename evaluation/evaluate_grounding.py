import json
from pathlib import Path

from app.pdf_loader import extract_text_from_pdf
from app.chunker import (
    chunk_text,
    create_chunk_records
)
from app.rag import RAG


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent
)

PDF_PATH = (
    BASE_DIR
    / "data"
    / "uploads"
    / "machine_learning.pdf"
)

TEST_PATH = (
    BASE_DIR
    / "evaluation"
    / "hallucination_tests.json"
)


# -----------------------------
# Load tests
# -----------------------------

with open(
    TEST_PATH,
    "r",
    encoding="utf-8"
) as file:

    tests = json.load(file)


# -----------------------------
# Create RAG
# -----------------------------

text = extract_text_from_pdf(
    str(PDF_PATH)
)

chunks = chunk_text(text)

records = create_chunk_records(
    chunks,
    PDF_PATH.name
)

rag = RAG()

rag.index_document(records)


# -----------------------------
# Run tests
# -----------------------------

for i, test in enumerate(
    tests,
    start=1
):

    question = test["question"]

    expected = test["expected"]


    result = rag.ask(
        question
    )


    print("\n")
    print("=" * 70)

    print(
        f"TEST {i}"
    )

    print(
        "Question:",
        question
    )

    print(
        "Expected:",
        expected
    )

    print(
        "\nAnswer:"
    )

    print(
        result["answer"]
    )


    print(
        "\nRetrieved sources:"
    )

    for source in result["sources"]:

        print(
            f"Chunk {source['chunk_id']} "
            f"| Score: {source['score']:.4f}"
        )