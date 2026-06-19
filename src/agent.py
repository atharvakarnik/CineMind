"""Runs the RAG pipeline: retrieve context, then call Gemini with a grounding prompt."""

from __future__ import annotations

import argparse
from typing import Any

from llama_index.llms.google_genai import GoogleGenAI

from config import GOOGLE_API_KEY
from src.retriever import DEFAULT_TOP_K, retrieve

MODEL_NAME = "gemini-3.5-flash"

GROUNDING_PROMPT_TEMPLATE = """You are a movie question-answering assistant. Answer the question \
using ONLY the context below. If the answer is not present in the context, say clearly that you \
don't know — do not invent or guess movie facts.

Context:
{context}

Question: {question}

Answer:"""

llm = GoogleGenAI(model=MODEL_NAME, api_key=GOOGLE_API_KEY)


def answer(question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Retrieve context for the question, ground Gemini's answer in it, and report sources used."""
    nodes = retrieve(question, top_k=top_k)
    context = "\n\n---\n\n".join(node.node.get_content() for node in nodes)
    prompt = GROUNDING_PROMPT_TEMPLATE.format(context=context, question=question)

    response = llm.complete(prompt)

    sources = [
        {"title": node.node.metadata.get("title"), "content_type": node.node.metadata.get("content_type")}
        for node in nodes
    ]
    return {"answer": str(response), "sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask CineMind a grounded question about a movie.")
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    result = answer(args.question, top_k=args.top_k)

    print(f"\nQuestion: {args.question}\n")
    print(f"Answer: {result['answer']}\n")
    print("Sources:")
    for source in result["sources"]:
        print(f"  - {source['title']} ({source['content_type']})")


if __name__ == "__main__":
    main()
