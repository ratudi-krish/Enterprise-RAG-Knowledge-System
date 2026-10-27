"""
eval.py
-------
Week 8 of the project plan: Testing and evaluation.

Measures how chunk size, chunk overlap, and top-k retrieval affect:
  - Faithfulness (does the answer only use the retrieved context? -- no hallucination)
  - Relevance (does the answer actually address the question?)
  - Retrieval + generation latency
  - Refusal accuracy (does the system correctly say "I don't know" for questions
    that are NOT answerable from the documents, instead of hallucinating?)

Faithfulness/relevance are scored by an LLM-as-judge, the same approach used by
RAGAs (Es et al., 2024) and discussed in Yu et al.'s (2024) evaluation survey --
both cited in the synopsis's Literature Survey.

SETUP REQUIRED before running:
  1. Make sure data/ has PDFs already ingested (via app.py upload or ingest.py).
  2. Fill in TEST_QUESTIONS below with real questions about YOUR documents:
     - a few ANSWERABLE questions (things actually in your PDFs)
     - a few UNANSWERABLE questions (plausible-sounding but NOT in your PDFs --
       this is what tests hallucination behaviour)

Run with:
    python eval.py

Outputs:
  - Per-question results printed live
  - eval_results.csv (full detail, one row per question x configuration)
  - A summary table comparing configurations, printed at the end
"""

import os
import csv
import glob
import json
import time
from collections import defaultdict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DATA_DIR = "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

# --- Configuration matrix to test (edit freely) ---
CHUNK_CONFIGS = [
    {"chunk_size": 300, "chunk_overlap": 30},
    {"chunk_size": 500, "chunk_overlap": 50},
    {"chunk_size": 800, "chunk_overlap": 80},
]
TOP_K_VALUES = [2, 3, 5]

# --- (question, is_answerable_from_your_docs) ---
# Keep 3-5 answerable + 2-3 unanswerable questions for a meaningful evaluation.
TEST_QUESTIONS = [
    ("Who is the student author of the Enterprise RAG Knowledge System synopsis?", True),
    ("What are the five major stages of the proposed Enterprise RAG workflow?", True),
    ("Under whose supervision was this project submitted?", True),
    ("What is the primary embedding model and vector database mentioned for the system?", True),
    ("What is the capital of Australia?", False),  # not in documents -> should refuse
    ("What was the company's Q3 revenue in 2024?", False),  # not in documents -> should refuse
]

ANSWER_PROMPT = """Answer the user's question based ONLY on the following context.
If the answer is not contained in the context, say "I don't know based on the
provided documents." Do not make anything up.

Context:
{context}

Question: {question}

Answer:"""

JUDGE_PROMPT = """You are evaluating a RAG system's answer for faithfulness and relevance.

Context provided to the system:
{context}

Question: {question}
Generated Answer: {answer}

Rate the answer on two dimensions, each from 1 (poor) to 5 (excellent):
- Faithfulness: does the answer only use information present in the context (no hallucination)?
- Relevance: does the answer actually address the question?

Respond with ONLY a JSON object in this exact format, nothing else:
{{"faithfulness": <int 1-5>, "relevance": <int 1-5>}}"""


def load_and_split(chunk_size, chunk_overlap):
    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in ./{DATA_DIR}/. Ingest documents first.")
    docs = []
    for p in pdf_paths:
        docs.extend(PyPDFLoader(p).load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def judge_answer(llm, context, question, answer):
    """LLM-as-judge scoring, RAGAs-style. Returns (faithfulness, relevance) or (None, None)."""
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"context": context, "question": question, "answer": answer})
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        scores = json.loads(cleaned)
        return scores.get("faithfulness"), scores.get("relevance")
    except Exception:
        return None, None


def run_evaluation():
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key from "
            "https://console.groq.com/keys"
        )
    if not TEST_QUESTIONS:
        raise ValueError(
            "TEST_QUESTIONS is empty. Open eval.py and fill in a few (question, "
            "is_answerable) tuples based on YOUR documents before running evaluation. "
            "See the commented examples near the top of the file."
        )

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    answer_chain = ChatPromptTemplate.from_template(ANSWER_PROMPT) | llm | StrOutputParser()

    results = []
    total_runs = len(CHUNK_CONFIGS) * len(TOP_K_VALUES) * len(TEST_QUESTIONS)
    print(f"Running {total_runs} evaluation queries across {len(CHUNK_CONFIGS)} chunk "
          f"configs x {len(TOP_K_VALUES)} top-k values x {len(TEST_QUESTIONS)} questions...\n")

    for cfg in CHUNK_CONFIGS:
        chunks = load_and_split(cfg["chunk_size"], cfg["chunk_overlap"])
        # Ephemeral, in-memory store per config -- does NOT touch your persisted
        # ./chroma_db/ used by the app, so evaluation never pollutes production data.
        store = Chroma.from_documents(documents=chunks, embedding=embeddings)

        for k in TOP_K_VALUES:
            retriever = store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": max(10, k * 2), "lambda_mult": 0.5},
            )

            for question, is_answerable in TEST_QUESTIONS:
                t0 = time.time()
                docs = retriever.invoke(question)
                retrieval_latency = time.time() - t0

                context = "\n\n".join(d.page_content for d in docs)

                t1 = time.time()
                answer = answer_chain.invoke({"context": context, "question": question})
                generation_latency = time.time() - t1

                refused = "i don't know" in answer.lower()
                faithfulness, relevance = (None, None)
                if is_answerable and not refused:
                    faithfulness, relevance = judge_answer(llm, context, question, answer)

                row = {
                    "chunk_size": cfg["chunk_size"],
                    "chunk_overlap": cfg["chunk_overlap"],
                    "top_k": k,
                    "question": question,
                    "is_answerable": is_answerable,
                    "correctly_refused": (not is_answerable and refused),
                    "faithfulness": faithfulness,
                    "relevance": relevance,
                    "retrieval_latency_s": round(retrieval_latency, 3),
                    "generation_latency_s": round(generation_latency, 3),
                }
                results.append(row)
                print(f"[size={cfg['chunk_size']:>4} overlap={cfg['chunk_overlap']:>3} k={k}] "
                      f"{question[:45]:<45} faith={faithfulness} rel={relevance} "
                      f"lat={retrieval_latency + generation_latency:.2f}s")

    write_results(results)
    print_summary(results)


def write_results(results):
    if not results:
        return
    with open("eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nDetailed results written to eval_results.csv")


def print_summary(results):
    groups = defaultdict(list)
    for r in results:
        groups[(r["chunk_size"], r["chunk_overlap"], r["top_k"])].append(r)

    print("\n=== Summary by configuration ===")
    header = f"{'chunk_size':>10} {'overlap':>8} {'top_k':>6} {'avg_faith':>10} {'avg_rel':>8} {'avg_latency_s':>14} {'refusal_acc':>12}"
    print(header)
    print("-" * len(header))
    for key, rows in sorted(groups.items()):
        faiths = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
        rels = [r["relevance"] for r in rows if r["relevance"] is not None]
        latencies = [r["retrieval_latency_s"] + r["generation_latency_s"] for r in rows]
        unanswerable = [r for r in rows if not r["is_answerable"]]

        avg_faith = sum(faiths) / len(faiths) if faiths else float("nan")
        avg_rel = sum(rels) / len(rels) if rels else float("nan")
        avg_latency = sum(latencies) / len(latencies) if latencies else float("nan")
        refusal_acc = (
            sum(r["correctly_refused"] for r in unanswerable) / len(unanswerable)
            if unanswerable else float("nan")
        )
        print(f"{key[0]:>10} {key[1]:>8} {key[2]:>6} {avg_faith:>10.2f} {avg_rel:>8.2f} "
              f"{avg_latency:>14.2f} {refusal_acc:>12.2f}")

    print("\nHigher faith/relevance (max 5) and refusal_acc (max 1.0) are better; "
          "lower latency is better. Use this table to justify your chosen chunk_size, "
          "chunk_overlap, and top_k in vectorstore_utils.py / rag_chain.py.")


if __name__ == "__main__":
    run_evaluation()
