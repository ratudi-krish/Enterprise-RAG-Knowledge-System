# Enterprise RAG Knowledge System

An advanced Retrieval-Augmented Generation (RAG) system for intelligent document question-answering with **Hybrid Retrieval (Dense + Sparse BM25 via Reciprocal Rank Fusion)**, **Conversational Memory**, **Verifiable Page Citations**, and an **Empirical Evaluation Dashboard**.

Submitted in partial fulfilment of the requirements for the degree of Bachelor of Technology in Computer Science & Engineering – AIML at Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow.

---

## ⚡ Tech Stack

- **Orchestration:** LangChain (LCEL Expression Language)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local, zero API cost, 384-dimensional dense vectors)
- **Sparse Indexing:** Native Okapi BM25 Lexical Keyword Engine
- **Fusion Algorithm:** Reciprocal Rank Fusion (RRF: $\sum \frac{w}{60 + \text{rank}}$)
- **Vector Database:** ChromaDB (Persisted to disk)
- **LLM Inference:** Groq LPU Acceleration (`openai/gpt-oss-20b`, `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)
- **Frontend:** Streamlit (Cyberpunk glassmorphic 3-tab dashboard)

---

## 🚀 Setup & Launch

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Groq API Key**
   - Get a free key at [console.groq.com/keys](https://console.groq.com/keys)
   - Copy `.env.example` to `.env` and paste your key:
     ```bash
     GROQ_API_KEY=your_key_here
     ```

3. **Launch the Web Dashboard**
   ```bash
   streamlit run app.py
   ```

4. **Alternative (CLI Batch Ingestion)**
   Drop `.pdf`, `.txt`, `.md`, or `.csv` files into `data/` and run:
   ```bash
   python ingest.py
   ```

---

## 🌟 What Makes This "Enterprise-Grade" (Winning Criteria)

1. **Hybrid Retrieval (Dense Semantic + Sparse BM25 with RRF)**:
   - Pure vector search fails on alphanumeric identifiers (e.g. roll numbers, product codes, section numbers).
   - Our system runs dense vector search and sparse BM25 keyword matching concurrently, merging candidate lists using Reciprocal Rank Fusion.
2. **Conversational Memory with Contextual Query Rewriting**:
   - Multi-turn conversational context is maintained. Follow-up queries (e.g. *"What about his guide?"*) are reformulated into standalone queries before retrieval.
3. **Audit-Ready Source Attribution**:
   - Citations show the exact source document, 1-indexed physical page, match confidence percentage, dense/sparse ranks, and a verbatim chunk excerpt quote.
4. **100% Grounded Refusal against Hallucination**:
   - Strict prompt guardrails force the model to output *"I don't know based on the provided documents"* when facts are absent, visually highlighted in the UI.
5. **Zero-Downtime Multi-Model Fallback**:
   - Resilient cascade between Groq models (`openai/gpt-oss-20b` $\rightarrow$ `llama-3.3-70b-versatile` $\rightarrow$ `llama-3.1-8b-instant`) to prevent 404 access disruptions.
6. **Live Empirical Evaluation Dashboard**:
   - Tab 2 visualizes quantitative benchmarks (Faithfulness, Relevance, Latency, and Refusal Accuracy) across 9 hyperparameter configurations based on RAGAs methodology.
7. **Viva Voce Defense Guide**:
   - Tab 3 provides an interactive architectural blueprint and defense cheat sheet with model answers for academic evaluators.

---

## 📊 Evaluation & Benchmarking

Run the automated LLM-as-a-judge evaluation harness:
```bash
python eval.py
```

This tests 9 configuration permutations (Chunk sizes 300, 500, 800 $\times$ Top-K 2, 3, 5) and generates `eval_results.csv`, validating:
- **Faithfulness (1–5)**: Does the answer strictly use retrieved context without hallucination?
- **Relevance (1–5)**: Does the answer address the question?
- **Latency (seconds)**: Retrieval + generation time.
- **Refusal Accuracy**: Does it correctly refuse out-of-domain control queries?

---

## 📁 Project Structure

```
RAG Project/
├── app.py                  # 3-Tab Streamlit Dashboard (Q&A, Evaluation, Architecture & Viva)
├── rag_chain.py            # Conversational RAG chain with Hybrid RRF & Multi-Model Fallback
├── vectorstore_utils.py    # Multi-format loader, BM25 Index, ChromaDB, and RRF search
├── ingest.py               # Headless CLI batch ingestion tool
├── eval.py                 # Automated RAGAs-style quantitative benchmarking harness
├── eval_results.csv        # Pre-computed benchmark results powering Tab 2
├── requirements.txt        # Python dependency manifest
├── .env.example            # Environment configuration template
├── .gitignore              # Git ignore rules protecting secrets and caches
├── chroma_db/              # Persisted vector storage
└── data/                   # Document repository (.pdf, .txt, .md, .csv)
```
