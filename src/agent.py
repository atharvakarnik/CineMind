"""Runs the RAG pipeline: retrieve context, then call Gemini with a grounding prompt."""

from __future__ import annotations

import argparse
import time
from typing import Any

from google.genai.errors import ServerError
from llama_index.llms.google_genai import GoogleGenAI

from config import GEMINI_FALLBACK_MODEL, GEMINI_MODEL, GOOGLE_API_KEY
from src.retriever import DEFAULT_TOP_K, retrieve

RETRY_MAX_ATTEMPTS = 4
RETRY_BASE_DELAY_SECONDS = 4.0  # waits ~4s, 8s, 16s between attempts (~28s total) before giving up
UNAVAILABLE_MESSAGE = "The model is temporarily unavailable, please try again."

GROUNDING_PROMPT_TEMPLATE = """You are a movie question-answering assistant. Answer the question \
using ONLY the context below. If the answer is not present in the context, say clearly that you \
don't know — do not invent or guess movie facts.

Context:
{context}

Question: {question}

Answer:"""

_llm_cache: dict[str, GoogleGenAI] = {}


def _get_llm(model_name: str) -> GoogleGenAI:
    """Build (and cache) a GoogleGenAI client for the given model."""
    if model_name not in _llm_cache:
        _llm_cache[model_name] = GoogleGenAI(model=model_name, api_key=GOOGLE_API_KEY)
    return _llm_cache[model_name]


def _complete_with_retries(prompt: str, model_name: str, max_attempts: int) -> str | None:
    """Call llm.complete with exponential backoff on server overload. Returns None if every attempt fails."""
    llm = _get_llm(model_name)
    for attempt in range(max_attempts):
        try:
            return str(llm.complete(prompt))
        except ServerError as exc:
            if attempt == max_attempts - 1:
                print(f"  ! {model_name} still unavailable after {max_attempts} attempts: {exc}")
                return None
            delay = RETRY_BASE_DELAY_SECONDS * (2**attempt)
            print(
                f"  ! {model_name} returned a server error (attempt {attempt + 1}/{max_attempts}); "
                f"retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
    return None


def _generate_answer(prompt: str) -> str:
    """Call Gemini with retries, falling back to a secondary model if the primary stays unavailable."""
    result = _complete_with_retries(prompt, GEMINI_MODEL, RETRY_MAX_ATTEMPTS)
    if result is not None:
        return result

    print(f"  ! Falling back to {GEMINI_FALLBACK_MODEL}...")
    result = _complete_with_retries(prompt, GEMINI_FALLBACK_MODEL, max_attempts=1)
    return result if result is not None else UNAVAILABLE_MESSAGE


def answer(question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Retrieve context for the question, ground Gemini's answer in it, and report sources used."""
    nodes = retrieve(question, top_k=top_k)
    context = "\n\n---\n\n".join(node.node.get_content() for node in nodes)
    prompt = GROUNDING_PROMPT_TEMPLATE.format(context=context, question=question)

    answer_text = _generate_answer(prompt)

    sources = [
        {"title": node.node.metadata.get("title"), "content_type": node.node.metadata.get("content_type")}
        for node in nodes
    ]
    return {"answer": answer_text, "sources": sources}


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
