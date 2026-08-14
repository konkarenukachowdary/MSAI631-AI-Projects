import gradio as gr

from rag_pipeline import (
    build_retriever,
    retrieve,
    load_generator,
    generate_answer,
    build_context,
)


# =========================================================
# Configuration
# =========================================================

TOP_K = 3

# Minimum similarity required before the system sends
# retrieved information to the language model.
RELEVANCE_THRESHOLD = 0.55


# =========================================================
# Load Retrieval System
# =========================================================

print("=" * 70)
print("AI STUDENT ACADEMIC SUPPORT ASSISTANT")
print("=" * 70)

print("\nLoading retrieval system...")

embedding_model, chunks, embeddings = build_retriever()

print(
    f"Loaded {len(chunks)} document chunks."
)


# =========================================================
# Load Language Model
# =========================================================

print("\nLoading language model...")

tokenizer, generator = load_generator()

print("\nAcademic Support Assistant is ready.")


# =========================================================
# Answer Student Question
# =========================================================

def answer_question(question):
    """
    Process a student question through the RAG pipeline.

    Flow:

    Student Question
          ↓
    Semantic Retrieval
          ↓
    Relevance Check
          ↓
    Context Construction
          ↓
    Language Model
          ↓
    Grounded Answer
    """

    # -----------------------------------------------------
    # Validate Question
    # -----------------------------------------------------

    if not question or not question.strip():

        return (
            "Please enter a question.",
            ""
        )

    question = question.strip()

    # -----------------------------------------------------
    # Retrieve Relevant Chunks
    # -----------------------------------------------------

    results = retrieve(
        question,
        embedding_model,
        chunks,
        embeddings,
        top_k=TOP_K,
    )

    # -----------------------------------------------------
    # Handle No Retrieval Results
    # -----------------------------------------------------

    if not results:

        answer = (
            "The provided academic support material does not "
            "contain enough information to answer this question."
        )

        return answer, ""


    # -----------------------------------------------------
    # Check Relevance
    # -----------------------------------------------------

    top_score = results[0]["score"]

    if top_score < RELEVANCE_THRESHOLD:

        answer = (
            "The provided academic support material does not "
            "contain enough information to answer this question."
        )

        source_lines = [
            "### 📚 Retrieved Sources",
            "",
            (
                "The retrieved information did not meet the "
                "minimum relevance threshold."
            ),
            "",
        ]

        for result in results:

            source_lines.append(
                f"- **{result['source']}** "
                f"({get_file_type(result['source'])})"
            )

            source_lines.append(
                f"  - Chunk: {result['chunk_id']}"
            )

            source_lines.append(
                f"  - Similarity: {result['score']:.4f}"
            )

            source_lines.append("")

        return answer, "\n".join(source_lines)


    # -----------------------------------------------------
    # Build Retrieved Context
    # -----------------------------------------------------

    context = build_context(results)


    # -----------------------------------------------------
    # Generate Grounded Answer
    # -----------------------------------------------------

    answer = generate_answer(
    question,
    context,
    results,
    tokenizer,
    generator,
    )


    # -----------------------------------------------------
    # Display Retrieved Sources
    # -----------------------------------------------------

    source_lines = [
        "### 📚 Retrieved Sources",
        "",
    ]

    for result in results:

        source = result["source"]

        source_lines.append(
            f"- **{source}** "
            f"({get_file_type(source)})"
        )

        source_lines.append(
            f"  - Chunk: {result['chunk_id']}"
        )

        source_lines.append(
            f"  - Similarity: {result['score']:.4f}"
        )

        source_lines.append("")


    source_text = "\n".join(source_lines)

    return answer, source_text


# =========================================================
# File Type Helper
# =========================================================

def get_file_type(filename):
    """
    Return a readable document type for the UI.
    """

    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        return "PDF"

    if filename_lower.endswith(".md"):
        return "Markdown"

    if filename_lower.endswith(".txt"):
        return "Text"

    return "Document"


# =========================================================
# Gradio User Interface
# =========================================================

with gr.Blocks(
    title="AI Student Academic Support Assistant"
) as demo:

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    gr.Markdown(
        """
# 🎓 AI Student Academic Support Assistant

Ask questions about **academic support, study skills,
time management, assignments, and related topics.**

The assistant uses **Retrieval-Augmented Generation (RAG)**
to retrieve relevant information from the academic-support
knowledge base before generating an answer.

The knowledge base supports **multiple document files,
including Markdown (`.md`) and PDF (`.pdf`) documents.**
"""
    )


    # -----------------------------------------------------
    # Student Question
    # -----------------------------------------------------

    question = gr.Textbox(
        label="Student Question",
        placeholder=(
            "Example: How can I improve my academic "
            "time management?"
        ),
        lines=3,
    )


    # -----------------------------------------------------
    # Ask Button
    # -----------------------------------------------------

    ask_button = gr.Button(
        "Ask Assistant",
        variant="primary",
    )


    # -----------------------------------------------------
    # Assistant Answer
    # -----------------------------------------------------

    gr.Markdown(
        "### 💬 Assistant Answer"
    )

    answer = gr.Textbox(
        label="Answer",
        lines=6,
        interactive=False,
    )


    # -----------------------------------------------------
    # Retrieved Sources
    # -----------------------------------------------------

    sources = gr.Markdown(
        label="Retrieved Sources"
    )


    # -----------------------------------------------------
    # Button Event
    # -----------------------------------------------------

    ask_button.click(
        fn=answer_question,
        inputs=question,
        outputs=[
            answer,
            sources,
        ],
    )


    # -----------------------------------------------------
    # Enter Key Event
    # -----------------------------------------------------

    question.submit(
        fn=answer_question,
        inputs=question,
        outputs=[
            answer,
            sources,
        ],
    )


# =========================================================
# Launch Application
# =========================================================

if __name__ == "__main__":

    demo.launch()