"""Performs query-time retrieval of relevant chunks from the Chroma vector store."""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

CHROMA_STORE_DIR = Path("chroma_store")
COLLECTION_NAME = "cinemind"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

_index: VectorStoreIndex | None = None


def _get_index() -> VectorStoreIndex:
    """Load the existing Chroma collection as a VectorStoreIndex, without re-embedding or rebuilding."""
    global _index
    if _index is not None:
        return _index

    # Must match the embed model used at index time, or query vectors won't be
    # comparable to the stored ones and retrieval will return nonsense.
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_STORE_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except ValueError as exc:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found in {CHROMA_STORE_DIR}. "
            "Run `python -m src.index` first."
        ) from exc

    vector_store = ChromaVectorStore(chroma_collection=collection)
    _index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)
    return _index


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[NodeWithScore]:
    """Embed a question with MiniLM and return the top_k most relevant chunks."""
    index = _get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks for a question (no LLM call).")
    parser.add_argument("question", help="The question to retrieve context for.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    nodes = retrieve(args.question, top_k=args.top_k)

    print(f"\nQuery: {args.question}")
    print(f"Retrieved {len(nodes)} chunks:\n")
    for i, node in enumerate(nodes, start=1):
        metadata = node.node.metadata
        preview = node.node.get_content().strip().replace("\n", " ")[:160]
        print(
            f"[{i}] score={node.score:.4f} | "
            f"{metadata.get('title')} ({metadata.get('year')}) | {metadata.get('content_type')}"
        )
        print(f"    {preview}...\n")


if __name__ == "__main__":
    main()
