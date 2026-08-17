import json
from pathlib import Path

from google import genai
from dotenv import load_dotenv


load_dotenv()


client = genai.Client()


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent
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


def evaluate_answer(
    question,
    context,
    reference_answer,
    generated_answer
):

    prompt = f"""
You are evaluating an answer generated
by a RAG system.

Evaluate the generated answer using ONLY
the provided document context.

QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{generated_answer}

Evaluate three criteria.

1. Faithfulness:
Is the generated answer supported by
the document context?

2. Relevance:
Does the generated answer directly
answer the question?

3. Correctness:
Does the generated answer agree with
the reference answer and document?

Give each score from 1 to 5.

5 = Excellent
4 = Good
3 = Partially correct
2 = Mostly incorrect
1 = Completely incorrect

Return ONLY valid JSON:

{{
    "faithfulness": 1,
    "relevance": 1,
    "correctness": 1,
    "reason": "brief explanation"
}}
"""


    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )


    return response.text