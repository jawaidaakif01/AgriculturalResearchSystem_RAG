"""
Report generation stage — built with LangChain (matches the rest of the project stack).

Takes the output of augment_with_full_text() — retrieved records where some
have full_text and some only have an abstract — and generates a synthesized,
cited research report via Gemini through LangChain. Explicitly instructed to
distinguish what it actually knows in detail vs. what it only has a summary
of, and to say so rather than inventing specifics that aren't in the
retrieved material.

Usage:
    from generate_report import generate_report

    report = generate_report(user_query, augmented_records)
    print(report)

Install deps (already in requirements.txt):
    langchain, langchain-google-genai, python-dotenv
Requires: GOOGLE_API_KEY set in your environment or .env file
    (this is the env var name langchain-google-genai looks for by default)
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"  # swap for a different Gemini model if you prefer

SYSTEM_INSTRUCTIONS = """You are an agricultural research assistant. You generate research reports \
for users based ONLY on the sources provided to you below — never from general/prior knowledge.

Rules you must follow:
1. Every claim in your report must be traceable to a specific numbered source. Cite sources \
inline like [Source 2].
2. Sources marked (FULL TEXT) may contain detailed methods, exact figures, application rates, \
or experimental parameters — use these when present and cite them specifically.
3. Sources marked (ABSTRACT ONLY) only give you a high-level summary of that paper's findings. \
Do NOT invent specific numbers, dosages, or technical parameters that aren't explicitly stated in \
the abstract text — if the abstract doesn't contain that level of detail, say so plainly rather \
than guessing.
4. If the retrieved sources collectively do not contain enough information to fully answer the \
user's question, say so explicitly and describe what IS covered versus what is missing, rather \
than filling gaps with unstated assumptions.
5. Write the report in clear, structured prose aimed at someone doing agricultural research — \
organize by theme where multiple sources overlap, don't just list sources one by one."""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTIONS),
    ("human", "User's research question: {query}\n\nRetrieved sources:\n{context}\n\n"
              "Now write the research report."),
])


def _build_context_block(records):
    """Turns retrieved records into a labeled context block for the prompt,
    clearly marking which sources have full text vs. abstract-only."""
    blocks = []
    for i, record in enumerate(records, 1):
        has_full_text = bool(record.get("full_text"))
        content = record["full_text"] if has_full_text else record["abstract"]
        detail_level = "FULL TEXT" if has_full_text else "ABSTRACT ONLY"

        block = (
            f"[Source {i}] ({detail_level})\n"
            f"Title: {record['title']}\n"
            f"Date: {record.get('date', 'unknown')}\n"
            f"Content: {content}\n"
            f"Reference URL: {record.get('source_id', 'unknown')}\n"
        )
        blocks.append(block)
    return "\n---\n".join(blocks)


def generate_report(user_query, augmented_records):
    """
    user_query: the original research question (str)
    augmented_records: list of retrieved records, each optionally containing
        a 'full_text' field from augment_with_full_text() — may be None per record
    Returns: the generated report text (str)
    """
    context_block = _build_context_block(augmented_records)

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)
    chain = PROMPT_TEMPLATE | llm | StrOutputParser()

    response = chain.invoke({"query": user_query, "context": context_block})
    return response


if __name__ == "__main__":
    # Minimal manual test with fabricated records — swap for real retrieved
    # records from your FAISS pipeline once Phase 2/3 are wired up.
    test_query = "What is known about Azoxystrobin's effectiveness against Cercospora leaf spot?"
    test_records = [
        {
            "title": "Fungicide trials for Cercospora leaf spot control in sugar beet",
            "date": "2021",
            "abstract": "Azoxystrobin application significantly reduced Cercospora leaf spot "
                         "severity compared to untreated control plots across two growing seasons.",
            "source_id": "https://example.org/paper1",
            "full_text": None,  # simulating an abstract-only source
        },
    ]
    print(generate_report(test_query, test_records))