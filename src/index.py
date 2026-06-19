"""Chunks and embeds normalized documents, then persists them to a Chroma collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

DATA_DIR = Path("data")
DOCUMENTS_PATH = DATA_DIR / "documents.json"
CHROMA_STORE_DIR = Path("chroma_store")
COLLECTION_NAME = "cinemind"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def load_documents() -> list[Document]:
    """Load normalized documents from data/documents.json into LlamaIndex Document objects."""
    raw_documents = json.loads(DOCUMENTS_PATH.read_text())
    return [Document(text=doc["text"], metadata=doc["metadata"]) for doc in raw_documents]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chroma vector index from data/documents.json.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the collection from scratch.")
    args = parser.parse_args()

    if not DOCUMENTS_PATH.exists():
        raise SystemExit(f"{DOCUMENTS_PATH} not found. Run `python -m src.ingest` first.")

    client = chromadb.PersistentClient(path=str(CHROMA_STORE_DIR))

    if args.rebuild and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() > 0 and not args.rebuild:
        print(
            f"Warning: collection '{COLLECTION_NAME}' already has {collection.count()} items. "
            "Pass --rebuild to delete and recreate it. Exiting without changes."
        )
        return

    # No LLM is needed to build embeddings; only set the embed model so LlamaIndex
    # never falls back to trying to instantiate OpenAI's default.
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = None

    documents = load_documents()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
    )
    node_count = len(index.docstore.docs)

    print("\n--- Indexing summary ---")
    print(f"Documents loaded: {len(documents)}")
    print(f"Nodes/chunks embedded: {node_count}")
    print(f"Final collection count: {collection.count()}")


if __name__ == "__main__":
    main()