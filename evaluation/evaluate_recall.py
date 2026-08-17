import json
from pathlib import Path

from app.pdf_loader import extract_text_from_pdf
from app.chunker import (
    chunk_text,
    create_chunk_records
)
from app.rag import RAG


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = (
    BASE_DIR
    / "data"
    / "uploads"
    / "machine_learning.pdf"
)

GROUND_TRUTH_PATH = (
    BASE_DIR
    / "evaluation"
    / "ground_truth.json"
)


# Load ground truth

with open(
    GROUND_TRUTH_PATH,
    "r",
    encoding="utf-8"
) as file:

    questions = json.load(file)


# Load PDF

print("Loading PDF...")

text = extract_text_from_pdf(
    str(PDF_PATH)
)


# Chunk PDF

chunks = chunk_text(
    text
)

records = create_chunk_records(
    chunks,
    PDF_PATH.name
)


# Create RAG

rag = RAG()

rag.index_document(
    records
)


# Evaluate

recall_at_1 = 0
recall_at_3 = 0
recall_at_5 = 0


for item in questions:

    question = item["question"]

    expected_chunk = (
        item["relevant_chunk_id"]
    )


    query_embedding = (
        rag.embedding_model.encode(
            [question]
        )[0]
    )


    results = (
        rag.vector_store.search(
            query_embedding,
            top_k=5
        )
    )


    retrieved_ids = [
        result["chunk_id"]
        for result in results
    ]


    hit_1 = (
        expected_chunk
        in retrieved_ids[:1]
    )

    hit_3 = (
        expected_chunk
        in retrieved_ids[:3]
    )

    hit_5 = (
        expected_chunk
        in retrieved_ids[:5]
    )


    recall_at_1 += int(hit_1)
    recall_at_3 += int(hit_3)
    recall_at_5 += int(hit_5)


    print("\n" + "=" * 60)

    print(
        "Question:",
        question
    )

    print(
        "Expected chunk:",
        expected_chunk
    )

    print(
        "Retrieved:",
        retrieved_ids
    )

    print(
        "Recall@1:",
        "✅" if hit_1 else "❌"
    )

    print(
        "Recall@3:",
        "✅" if hit_3 else "❌"
    )

    print(
        "Recall@5:",
        "✅" if hit_5 else "❌"
    )


total = len(questions)


print("\n" + "=" * 60)
print("FINAL RETRIEVAL EVALUATION")
print("=" * 60)


print(
    f"Recall@1: "
    f"{recall_at_1 / total:.2%}"
)

print(
    f"Recall@3: "
    f"{recall_at_3 / total:.2%}"
)

print(
    f"Recall@5: "
    f"{recall_at_5 / total:.2%}"
)