from pathlib import Path
from pypdf import PdfReader


def load_documents(documents_dir: str = "documents") -> list[dict]:
    """
    Load academic-support documents from the documents directory.

    Supported formats:
    - Markdown (.md)
    - Text (.txt)
    - PDF (.pdf)
    """

    documents_path = Path(documents_dir)
    documents = []

    if not documents_path.exists():
        print(f"Documents directory not found: {documents_path}")
        return documents

    # Load Markdown files
    for file_path in documents_path.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")

        if text.strip():
            documents.append(
                {
                    "source": file_path.name,
                    "text": text,
                }
            )

    # Load plain-text files
    for file_path in documents_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        if text.strip():
            documents.append(
                {
                    "source": file_path.name,
                    "text": text,
                }
            )

    # Load PDF files
    for file_path in documents_path.glob("*.pdf"):
        try:
            reader = PdfReader(str(file_path))

            pages = []

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    pages.append(page_text)

            text = "\n\n".join(pages)

            if text.strip():
                documents.append(
                    {
                        "source": file_path.name,
                        "text": text,
                    }
                )

        except Exception as error:
            print(
                f"Could not read PDF {file_path.name}: {error}"
            )

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded {len(documents)} document(s).")

    for document in documents:
        print(
            f"- {document['source']}: "
            f"{len(document['text'])} characters"
        )