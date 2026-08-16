import re


def clean_text(text: str) -> str:

    # Normalize common PDF artifacts
    text = text.replace(
        "\u00a0",
        " "
    )

    # Fix common "Th e" style spacing
    text = re.sub(
        r"\b([A-Za-z]{1,3})\s+e\b",
        r"\1e",
        text
    )

    # Remove repeated whitespace
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Normalize line breaks
    lines = text.splitlines()

    cleaned_lines = []

    previous_line = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Skip exact duplicate consecutive lines
        if line == previous_line:
            continue

        cleaned_lines.append(line)

        previous_line = line


    text = "\n".join(
        cleaned_lines
    )

    return text.strip()

def split_into_sentences(text: str):

    # Simple sentence splitter
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100
):

    text = clean_text(text)

    sentences = split_into_sentences(
        text
    )

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        # If adding the sentence still
        # fits within the chunk size
        if len(
            current_chunk
        ) + len(sentence) + 1 <= chunk_size:

            if current_chunk:

                current_chunk += " "

            current_chunk += sentence

        else:

            # Save current chunk
            if current_chunk:

                chunks.append(
                    current_chunk.strip()
                )

            # Keep last part for overlap
            if overlap > 0:

                overlap_text = (
                    current_chunk[-overlap:]
                )

            else:

                overlap_text = ""

            current_chunk = (
                overlap_text
                + " "
                + sentence
            ).strip()

    # Add final chunk
    if current_chunk:

        chunks.append(
            current_chunk.strip()
        )

    return chunks


def create_chunk_records(
    chunks,
    filename
):

    records = []

    for i, chunk in enumerate(
        chunks
    ):

        records.append(
            {
                "text": chunk,
                "source": filename,
                "chunk_id": i
            }
        )

    return records