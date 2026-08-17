import json
from pathlib import Path

from app.pdf_loader import (
    extract_text_from_pdf
)

from app.chunker import (
    chunk_text,
    create_chunk_records
)

from app.rag import RAG

from evaluation.answer_evaluator import (
    evaluate_answer
)


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
    / "answer_tests.json"
)


with open(
    TEST_PATH,
    "r",
    encoding="utf-8"
) as file:

    tests = json.load(file)


# Build RAG

text = extract_text_from_pdf(
    str(PDF_PATH)
)

chunks = chunk_text(text)

records = create_chunk_records(
    chunks,
    PDF_PATH.name
)

rag = RAG()

rag.index_document(
    records
)


# Evaluate

results = []


for i, test in enumerate(
    tests,
    start=1
):

    question = test["question"]

    reference_answer = (
        test["reference_answer"]
    )


    # Ask RAG

    rag_result = rag.ask(
        question
    )


    generated_answer = (
        rag_result["answer"]
    )


    # Build retrieved context

    context = "\n\n".join(
        source["text"]
        for source
        in rag_result["sources"]
    )


    # Judge answer

    evaluation = evaluate_answer(
        question=question,
        context=context,
        reference_answer=reference_answer,
        generated_answer=generated_answer
    )


    print("\n")
    print("=" * 70)

    print(
        f"Question {i}:"
    )

    print(question)

    print(
        "\nGenerated answer:"
    )

    print(generated_answer)

    print(
        "\nEvaluation:"
    )

    print(evaluation)


    results.append(
        {
            "question": question,
            "answer": generated_answer,
            "evaluation": evaluation
        }
    )