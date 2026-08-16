from pathlib import Path

from app.chunker import (
    chunk_text,
    create_chunk_records
)

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from pydantic import BaseModel

from dotenv import load_dotenv

from app.pdf_loader import (
    extract_text_from_pdf
)

from app.chunker import (
    chunk_text
)

from app.rag import RAG


load_dotenv()


app = FastAPI(
    title="Simple PDF RAG",
    description=(
        "Upload a PDF and ask questions "
        "about it."
    ),
    version="1.0.0"
)


UPLOAD_DIR = Path(
    "data/uploads"
)

UPLOAD_DIR.mkdir(
    exist_ok=True
)


rag = RAG()


class QuestionRequest(BaseModel):

    question: str


@app.get("/")
def home():

    return {
        "message": "Simple PDF RAG is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_path = (
        UPLOAD_DIR / file.filename
    )

    contents = await file.read()

    with open(
        file_path,
        "wb"
    ) as output_file:

        output_file.write(
            contents
        )

    try:

        text = extract_text_from_pdf(
            str(file_path)
        )

        if not text.strip():

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract text "
                    "from this PDF."
                )
            )

        chunks = chunk_text(
            text
        )

        records = create_chunk_records(
            chunks,
            file.filename
        )

        chunk_count = (
            rag.index_document(
                records
            )
        )

        return {
            "message": "PDF processed successfully",
            "filename": file.filename,
            "chunks": chunk_count
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        result = rag.ask(
            request.question
        )

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )