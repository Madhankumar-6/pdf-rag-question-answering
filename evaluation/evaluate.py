import json
from pathlib import Path

from app.pdf_loader import extract_text_from_pdf
from app.chunker import (
    chunk_text,
    create_chunk_records
)
from app.rag import RAG


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = (
    BASE_DIR
    / "data"
    / "uploads"
    / "machine_learning.pdf"
)

QUESTIONS_PATH = (
    BASE_DIR
    / "evaluation"
    / "questions.json"
)


# --------------------------------------------------
# Load evaluation questions
# --------------------------------------------------

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:

    questions = json.load(file)


# --------------------------------------------------
# Load PDF
# --------------------------------------------------

print("Loading PDF...")

text = extract_text_from_pdf(
    str(PDF_PATH)
)

print(
    f"Extracted characters: {len(text)}"
)


# --------------------------------------------------
# Create chunks
# --------------------------------------------------

chunks = chunk_text(
    text
)

records = create_chunk_records(
    chunks,
    PDF_PATH.name
)

print(
    f"Created chunks: {len(records)}"
)


# --------------------------------------------------
# Create RAG system
# --------------------------------------------------

rag = RAG()


# Build FAISS index

rag.index_document(
    records
)

print("FAISS index created.")


# --------------------------------------------------
# Evaluate retrieval
# --------------------------------------------------

correct = 0

total = len(questions)


print("\n")
print("=" * 70)
print("RAG RETRIEVAL EVALUATION")
print("=" * 70)


for i, item in enumerate(
    questions,
    start=1
):

    question = item["question"]

    expected_keywords = [
        keyword.lower()
        for keyword
        in item["expected_keywords"]
    ]


    # Create question embedding

    query_embedding = (
        rag.embedding_model.encode(
            [question]
        )[0]
    )


    # Retrieve top 3 chunks

    results = (
        rag.vector_store.search(
            query_embedding,
            top_k=5
        )
    )


    # Combine retrieved text

    retrieved_text = " ".join(
        result["text"]
        for result in results
    ).lower()


    # Check expected keywords

    found_keywords = [
        keyword
        for keyword
        in expected_keywords
        if keyword in retrieved_text
    ]


    passed = (
        len(found_keywords)
        == len(expected_keywords)
    )


    print("\n")
    print(f"Question {i}:")
    print(question)

    print(
        "\nExpected keywords:"
    )

    print(
        expected_keywords
    )

    print(
        "\nFound keywords:"
    )

    print(
        found_keywords
    )


    if passed:

        print(
            "Result: ✅ PASS"
        )

        correct += 1

    else:

        print(
            "Result: ❌ FAIL"
        )


    print(
        "\nTop retrieved chunks:"
    )


    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            result["text"][:500]
        )

        print("-" * 50)


# --------------------------------------------------
# Final score
# --------------------------------------------------

accuracy = (
    correct / total
    if total > 0
    else 0
)


print("\n")
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    f"Passed: {correct}/{total}"
)

print(
    f"Retrieval Keyword Accuracy: "
    f"{accuracy:.2%}"
)