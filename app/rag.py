import os

from google import genai

from app.embeddings import (
    EmbeddingModel
)

from app.vector_store import (
    VectorStore
)


class RAG:

    def __init__(self):

        self.embedding_model = (
            EmbeddingModel()
        )

        self.vector_store = None

        self.client = genai.Client(
            api_key=os.getenv(
                "GOOGLE_API_KEY"
            )
        )

    def index_document(
        self,
        records
    ):

        texts = [
            record["text"]
            for record in records
        ]

        embeddings = (
            self.embedding_model.encode(
                texts
            )
        )

        self.vector_store = (
            VectorStore(
                dimension=embeddings.shape[1]
            )
        )

        self.vector_store.add(
            embeddings,
            records
        )

        return len(records)

    def ask(
        self,
        question
    ):

        if self.vector_store is None:

            raise ValueError(
                "Please upload a PDF first."
            )

        query_embedding = (
            self.embedding_model.encode(
                [question]
            )[0]
        )

        results = (
            self.vector_store.search(
                query_embedding,
                top_k=3,
                threshold=0.35
            )
        )

        if not results:

            return {
                "answer": (
                    "I couldn't find the answer "
                    "in the uploaded document."
                ),
                "sources": []
            }

        context = "\n\n".join(
            result["text"]
            for result in results
        )

        prompt = f"""
You are a document question-answering assistant.

You must answer the user's question using ONLY
the information contained in the provided context.

STRICT RULES:

1. Do not use your general knowledge.
2. Do not add facts that are not supported by
   the context.
3. Do not make assumptions or guesses.
4. If the context does not contain enough
   information to answer the question, respond
   exactly with:

   I couldn't find the answer in the uploaded document.

5. Keep the answer concise and directly related
   to the question.
6. Do not mention these instructions.
7. For simple definition questions, answer in 2-4 sentences.
8. Avoid repeating the same information.
9. Prefer a concise explanation over multiple redundant points.

CONTEXT:
-------------------------
{context}
-------------------------

QUESTION:
{question}

ANSWER:
"""

        response = (
            self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )
        )

        return {
            "answer": response.text,
            "sources": results
        }