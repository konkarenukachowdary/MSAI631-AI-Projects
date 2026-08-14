from document_processor import load_documents
from chunker import chunk_text

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def build_chunks():
    """Load documents and split them into overlapping chunks."""

    documents = load_documents()

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

    return all_chunks


def cosine_similarity(query_embedding, document_embeddings):
    """Calculate cosine similarity between a query and document embeddings."""

    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    document_embeddings = document_embeddings / np.linalg.norm(
        document_embeddings,
        axis=1,
        keepdims=True,
    )

    return np.dot(document_embeddings, query_embedding)


def build_retriever():
    """Create embeddings for all document chunks."""

    chunks = build_chunks()

    if not chunks:
        raise ValueError("No document chunks were found.")

    print(f"Created {len(chunks)} chunks.")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
    )

    print(f"Created embeddings with dimension {embeddings.shape[1]}.")

    return model, chunks, embeddings


def retrieve(
    query,
    model,
    chunks,
    embeddings,
    top_k=3,
):
    """Retrieve the most relevant chunks for a query."""

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append(
            {
                "source": chunks[index]["source"],
                "chunk_id": chunks[index]["chunk_id"],
                "text": chunks[index]["text"],
                "score": float(scores[index]),
            }
        )

    return results


if __name__ == "__main__":

    model, chunks, embeddings = build_retriever()

    query = "How can I create a realistic study schedule?"

    results = retrieve(
        query,
        model,
        chunks,
        embeddings,
        top_k=3,
    )

    print(f"\nQuery: {query}\n")

    for i, result in enumerate(results, start=1):

        print(f"--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["text"])
        print()