from document_processor import load_documents


def chunk_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[str]:
    """
    Split document text into paragraph/section-aware overlapping chunks.

    The function tries to keep headings and their related paragraphs together
    instead of splitting a section in the middle.
    """

    if not text or not text.strip():
        return []

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split into paragraphs while preserving meaningful content
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_parts = []
    current_length = 0

    for paragraph in paragraphs:

        paragraph_length = len(paragraph)

        # If adding this paragraph would exceed the chunk size,
        # save the current chunk first.
        if (
            current_parts
            and current_length + paragraph_length + 2 > chunk_size
        ):
            chunk = "\n\n".join(current_parts).strip()

            if chunk:
                chunks.append(chunk)

            # Keep some overlap from the previous chunk.
            overlap_parts = []
            overlap_length = 0

            for previous in reversed(current_parts):
                if overlap_length + len(previous) + 2 > overlap:
                    break

                overlap_parts.insert(0, previous)
                overlap_length += len(previous) + 2

            current_parts = overlap_parts
            current_length = overlap_length

        # If a single paragraph is larger than the chunk size,
        # split it safely by characters.
        if paragraph_length > chunk_size:

            if current_parts:
                chunk = "\n\n".join(current_parts).strip()

                if chunk:
                    chunks.append(chunk)

                current_parts = []
                current_length = 0

            start = 0

            while start < paragraph_length:
                end = start + chunk_size
                piece = paragraph[start:end].strip()

                if piece:
                    chunks.append(piece)

                if end >= paragraph_length:
                    break

                start += chunk_size - overlap

            continue

        current_parts.append(paragraph)
        current_length += paragraph_length + 2

    # Add final chunk
    if current_parts:
        chunk = "\n\n".join(current_parts).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    documents = load_documents()

    print(f"Loaded {len(documents)} document(s).")

    all_chunks = []

    for document in documents:

        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": index,
                    "text": chunk,
                }
            )

    print(f"Created {len(all_chunks)} chunks.")

    for chunk in all_chunks:

        print(
            f"\n[{chunk['source']} - "
            f"chunk {chunk['chunk_id']}]\n"
            f"{chunk['text'][:300]}..."
        )