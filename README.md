
# AgriSearch AI — Evidence-Backed Agricultural Research System (RAG)

An enterprise-grade, retrieval-augmented generation (RAG) platform that ingests peer-reviewed agricultural literature from the FAO AGRIS corpus, dynamically discovers and extracts full-text scientific PDFs at query time, and autonomously triggers live web search when research questions fall outside the local index boundary[cite: 1].

---

## Core Capabilities

* **FAO AGRIS Literature Ingestion**: Crawls and structures thousands of peer-reviewed records containing metadata, subjects, and publication dates[cite: 1].
* **Vector Semantic Retrieval**: Encodes scientific text using `BAAI/bge-base-en-v1.5` embeddings into a local FAISS vector store[cite: 1].
* **Query-Time Full-Text Harvesting**: Dynamically extracts full-text scientific PDFs (via `pdfplumber`) for retrieved chunks rather than relying solely on abstracts[cite: 1].
* **Calibrated Web Fallback Routing**: Evaluates semantic relevance ($0.75$ L2 distance threshold); out-of-domain queries trigger Tavily Live Web Search automatically.
* **Custom Document Upload**: Supports direct upload and text parsing of user-supplied agricultural PDFs using `pypdf`.
* **Multi-Format Export**: Generates downloadable styled PDF reports via `reportlab`, alongside raw Markdown and JSON retrieval metadata.
* **Interactive Streamlit Workspace**: Clean pastel interface featuring state persistence across exports, diagnostic telemetry, and route provenance indicators.

---

## System Architecture

```text
                                [ User Query / Custom PDF ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ Custom PDF Upload ]                       [ Local FAISS Index ]
                       │                         (BAAI/bge-base-en-v1.5 Embeddings)
                       │                                           │
                       │                              L2 Distance Calculation
                       │                                           │
                       │                       ┌───────────────────┴───────────────────┐
                       │                       ▼                                       ▼
                       │               Score ≤ 0.75 (In-Domain)              Score > 0.75 (Out-of-Domain)
                       │                       │                                       │
                       │             [ Full-Text PDF Fetch ]                 [ Tavily Web Fallback ]
                       │             (Dynamic Web Scraping)                  (Live Web Intelligence)
                       │                       │                                       │
                       └───────────────────────┼───────────────────────────────────────┘
                                               │
                                               ▼
                              [ Gemini 2.0 Flash Synthesis ]
                             (Evidence-Backed Numbered Citations)
                                               │
                                               ▼
                          [ Interactive Workspace / PDF Export ]

```

---

## Repository Structure

```text
├── app.py                         # Streamlit interactive application with state management & ReportLab PDF export
├── style.css                      # Custom stylesheet for typography, cards, and upload elements
├── crawl_agris.py                 # DCAT XML crawler and parser for the FAO AGRIS repository
├── evaluate_fulltext_coverage.py  # Diagnostic tool evaluating open-access PDF harvesting success rates
├── full_text_fetch.py             # Query-time discovery and extraction of full-text research PDFs
├── generate_report.py             # Gemini 2.0 Flash prompt orchestrator with numbered citations
├── ingest.py                      # Text chunking, embedding generation, and FAISS indexing pipeline
├── query.py                       # CLI query interface with automated Tavily fallback routing
├── retrieval_utils.py             # Cross-platform embedding device selector (CUDA/MPS/CPU) and adapters
├── web_search.py                  # Tavily Search API client and schema mapping
├── pyproject.toml                 # uv project configuration and dependency specifications
└── .env.example                   # Environment configuration template

```

---

## Installation & Setup

### Prerequisites

* Python `>= 3.12`

* [`uv`](https://docs.astral.sh/uv/) package manager

### 1. Clone the Repository

```bash
git clone [https://github.com/jawaidaakif01/AgriculturalResearchSystem_RAG.git](https://github.com/jawaidaakif01/AgriculturalResearchSystem_RAG.git)
cd AgriculturalResearchSystem_RAG

```

### 2. Configure Environment Variables

Copy the template file to `.env`:

```bash
cp .env.example .env

```

Populate `.env` with your API credentials:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
HF_TOKEN=your_huggingface_read_token_here

```

### 3. Install Dependencies

```bash
uv sync

```

---

## Usage

### Interactive Web Application

Launch the Streamlit dashboard:

```bash
uv run streamlit run app.py

```

### Command-Line Interface (CLI)

Run queries directly in the terminal:

```bash
# In-Domain Query (Uses Local FAISS Index + Full-Text PDF Augmentation)
uv run python query.py "What are effective methods for controlling Cercospora leaf spot in sugar beets?"

# Out-of-Domain Query (Triggers Tavily Live Web Fallback)
uv run python query.py "What are the latest advancements in quantum computing superconducting qubits?"

```

---

## Ingestion & Diagnostics

### Build Local FAISS Index

To chunk, embed, and index an AGRIS dataset:

```bash
uv run python ingest.py --input agris_filtered.json --index_dir faiss_index

```

### Evaluate Full-Text Harvesting Coverage

To benchmark PDF extraction yield across open-access repositories:

```bash
uv run python evaluate_fulltext_coverage.py --input agris_filtered.json --sample_size 50

```

---

