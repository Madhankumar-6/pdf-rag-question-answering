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


with open(
    TEST_PATH,
    "r",
    encoding="utf-8"
) as file:

    tests = json.load(file)


# -----------------------------
# Build RAG
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
# Evaluate
# -----------------------------

total_unanswerable = 0
correctly_rejected = 0

total_answerable = 0
correctly_answered = 0


for i, test in enumerate(
    tests,
    start=1
):

    question = test["question"]
    expected = test["expected"]


    result = rag.ask(
        question
    )


    answer = (
        result["answer"]
        .strip()
        .lower()
    )


    refusal_phrase = (
        "i couldn't find the answer"
        in answer
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
        "Answer:",
        result["answer"]
    )


    # -------------------------
    # Unanswerable question
    # -------------------------

    if expected == "unanswerable":

        total_unanswerable += 1

        if refusal_phrase:

            correctly_rejected += 1

            print(
                "Result: ✅ Correctly rejected"
            )

        else:

            print(
                "Result: ❌ Potential hallucination"
            )


    # -------------------------
    # Answerable question
    # -------------------------

    elif expected == "answerable":

        total_answerable += 1

        if not refusal_phrase:

            correctly_answered += 1

            print(
                "Result: ✅ Answered"
            )

        else:

            print(
                "Result: ❌ Incorrectly rejected"
            )


# -----------------------------
# Final metrics
# -----------------------------

print("\n")
print("=" * 70)
print("HALLUCINATION EVALUATION")
print("=" * 70)


if total_unanswerable > 0:

    rejection_rate = (
        correctly_rejected
        / total_unanswerable
    )

    hallucination_rate = (
        1 - rejection_rate
    )

    print(
        f"Correct rejection rate: "
        f"{rejection_rate:.2%}"
    )

    print(
        f"Hallucination rate: "
        f"{hallucination_rate:.2%}"
    )


if total_answerable > 0:

    answer_rate = (
        correctly_answered
        / total_answerable
    )

    print(
        f"Correct answerable handling: "
        f"{answer_rate:.2%}"
    )