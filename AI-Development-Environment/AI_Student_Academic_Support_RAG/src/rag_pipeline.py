from retriever import build_retriever, retrieve
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "google/flan-t5-small"

RELEVANCE_THRESHOLD = 0.55

FALLBACK_MESSAGE = (
    "The provided academic support material does not contain "
    "enough information to answer this question."
)


# =========================================================
# Load Language Model
# =========================================================

def load_generator():
    """
    Load the tokenizer and FLAN-T5 language model.
    """

    print(f"Loading language model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    return tokenizer, model


# =========================================================
# Build Context
# =========================================================

def build_context(results):
    """
    Combine retrieved chunks into a single context string.
    """

    context_parts = []

    for result in results:

        context_parts.append(
            f"Source: {result['source']}\n"
            f"Chunk: {result['chunk_id']}\n"
            f"{result['text']}"
        )

    return "\n\n".join(context_parts)


# =========================================================
# Clean Text
# =========================================================

def clean_text(text):
    """
    Remove Markdown formatting and unnecessary whitespace.
    """

    if not text:
        return ""

    text = re.sub(r"#{1,6}\s*", "", text)

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    text = re.sub(r"\*(.*?)\*", r"\1", text)

    text = re.sub(r"`(.*?)`", r"\1", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# Detect Bad Model Output
# =========================================================

def is_bad_answer(answer):
    """
    Detect prompt leakage, incomplete output, and invalid
    FLAN-T5 responses.
    """

    if not answer:
        return True

    cleaned = clean_text(answer)

    if not cleaned:
        return True

    # Exact fallback is valid.
    if cleaned == FALLBACK_MESSAGE:
        return False

    # Too short to be a useful response.
    if len(cleaned.split()) < 6:
        return True

    # Outputs commonly produced when the model copies
    # instructions instead of answering.
    forbidden_phrases = [
        "give direct answer",
        "give a direct answer",
        "answer the student's question",
        "answer the student question",
        "do not copy",
        "do not reproduce",
        "do not use outside knowledge",
        "do not invent",
        "write 2 sentences",
        "write 2 or 3 sentences",
        "write a short",
        "use only information",
        "academic support context",
        "student question:",
        "academic support context:",
        "instructions:",
        "rules:",
    ]

    lower_answer = cleaned.lower()

    for phrase in forbidden_phrases:

        if phrase in lower_answer:
            return True

    # Reject simple numbering such as:
    # 1.
    # 1
    # 1. 2. 3.
    if re.fullmatch(
        r"[\d\.\-\s]+",
        cleaned
    ):
        return True

    # Reject obvious heading-only responses.
    if cleaned.endswith(":") and len(cleaned.split()) < 10:
        return True

    return False


# =========================================================
# Extractive Fallback
# =========================================================

def extractive_fallback(question, results):
    """
    Create a concise answer directly from the most relevant
    retrieved material when the small language model produces
    an unusable response.

    This keeps the answer grounded in the knowledge base.
    """

    if not results:
        return FALLBACK_MESSAGE

    question_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            question.lower()
        )
    )

    candidates = []

    for result in results:

        text = clean_text(result["text"])

        # Split into sentences.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence.split()) < 6:
                continue

            sentence_words = set(
                re.findall(
                    r"\b[a-zA-Z]{4,}\b",
                    sentence.lower()
                )
            )

            overlap = len(
                question_words.intersection(sentence_words)
            )

            score = (
                overlap * 2
                + result["score"]
            )

            candidates.append(
                (score, sentence)
            )

    if not candidates:
        return FALLBACK_MESSAGE

    # Highest relevance sentence first.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = []

    for _, sentence in candidates:

        if sentence not in selected:
            selected.append(sentence)

        if len(selected) == 2:
            break

    answer = " ".join(selected)

    return answer.strip()


# =========================================================
# Generate Grounded Answer
# =========================================================

def generate_answer(
    question,
    context,
    results,
    tokenizer,
    model,
):
    """
    Generate an answer using FLAN-T5.

    If FLAN-T5 produces prompt leakage or an incomplete
    response, use a grounded extractive fallback.
    """

    # -----------------------------------------------------
    # Simple Prompt
    # -----------------------------------------------------

    prompt = (
        "Answer the question using only the information "
        "provided in the context.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )

    # -----------------------------------------------------
    # Tokenization
    # -----------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()

    answer = clean_text(answer)

    # -----------------------------------------------------
    # Validate Model Output
    # -----------------------------------------------------

    if not is_bad_answer(answer):

        return answer

    # -----------------------------------------------------
    # Model Output Was Bad
    # -----------------------------------------------------

    print(
        "\nFLAN-T5 produced an incomplete or "
        "instruction-like response."
    )

    print(
        "Using grounded retrieval fallback."
    )

    return extractive_fallback(
        question,
        results
    )


# =========================================================
# Standalone RAG Pipeline Test
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI STUDENT ACADEMIC SUPPORT - RAG PIPELINE")
    print("=" * 70)


    # -----------------------------------------------------
    # Load Retrieval System
    # -----------------------------------------------------

    print("\nLoading retrieval system...")

    embedding_model, chunks, embeddings = build_retriever()

    print(
        f"Loaded {len(chunks)} chunks into the retrieval system."
    )


    # -----------------------------------------------------
    # Load Language Model
    # -----------------------------------------------------

    print("\nLoading language model...")

    tokenizer, generator = load_generator()

    print("Language model loaded successfully.")


    # -----------------------------------------------------
    # Test Question
    # -----------------------------------------------------

    question = (
        "What are some effective study strategies?"
    )

    print("\n" + "=" * 70)
    print("STUDENT QUESTION")
    print("=" * 70)

    print(question)


    # -----------------------------------------------------
    # Retrieve Documents
    # -----------------------------------------------------

    results = retrieve(
        question,
        embedding_model,
        chunks,
        embeddings,
        top_k=3,
    )


    # -----------------------------------------------------
    # Display Sources
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    if not results:

        print("No documents were retrieved.")

    else:

        for index, result in enumerate(
            results,
            start=1
        ):

            print(f"\nResult {index}")
            print("-" * 50)

            print(
                f"Source: {result['source']}"
            )

            print(
                f"Chunk: {result['chunk_id']}"
            )

            print(
                f"Similarity: {result['score']:.4f}"
            )

            print(
                f"Content: {result['text'][:500]}"
            )


    # -----------------------------------------------------
    # Relevance Check
    # -----------------------------------------------------

    if not results:

        print(
            "\nNo relevant documents were retrieved."
        )

        answer = FALLBACK_MESSAGE

    elif results[0]["score"] < RELEVANCE_THRESHOLD:

        print(
            "\nTop retrieval score is below "
            "the relevance threshold."
        )

        print(
            f"Threshold: {RELEVANCE_THRESHOLD:.2f}"
        )

        print(
            f"Top score: {results[0]['score']:.4f}"
        )

        answer = FALLBACK_MESSAGE

    else:

        # -------------------------------------------------
        # Build Context
        # -------------------------------------------------

        context = build_context(results)

        # -------------------------------------------------
        # Generate Answer
        # -------------------------------------------------

        answer = generate_answer(
            question,
            context,
            results,
            tokenizer,
            generator,
        )


    # -----------------------------------------------------
    # Display Answer
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)

    print(answer)


    # -----------------------------------------------------
    # Complete
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)