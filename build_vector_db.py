#!/usr/bin/env python3
"""
build_vector_db.py

Builds a local Chroma vector database in the ./db folder
using the extracted chunks from the PDF knowledge base.
"""

import os
import json
import argparse
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Universal LangChain imports
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


def get_embedding_function(provider: str = "auto"):
    """
    Returns the appropriate embedding model based on user preference and environment.
    - 'openai': Uses OpenAI text-embedding-3-small (requires OPENAI_API_KEY in .env)
    - 'huggingface': Uses local all-MiniLM-L6-v2 (100% free, runs offline, no key needed)
    - 'auto': Uses OpenAI if OPENAI_API_KEY is configured, else local HuggingFace.
    """
    openai_key = os.getenv("OPENAI_API_KEY")

    if provider == "openai" or (provider == "auto" and openai_key):
        if not openai_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in environment or .env file.\n"
                "Please add OPENAI_API_KEY=your_key to .env, or run with --provider huggingface"
            )
        from langchain_openai import OpenAIEmbeddings
        print("Using OpenAI Embeddings (text-embedding-3-small)...")
        return OpenAIEmbeddings(model="text-embedding-3-small"), "openai", "text-embedding-3-small"
    else:
        print("Using local HuggingFace Embeddings (all-MiniLM-L6-v2, no API key required)...")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), "huggingface", "all-MiniLM-L6-v2"


def load_chunks_for_indexing() -> List[Document]:
    """Loads chunks from chunks_output.json or directly from pdf_data.py."""
    chunks_file = "chunks_output.json"
    documents: List[Document] = []

    if os.path.exists(chunks_file):
        print(f"Loading chunks from {chunks_file}...")
        with open(chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Prefer recursive character chunks
        raw_chunks = data.get("chunks", {}).get("recursive_character", [])
        if not raw_chunks:
            # Fallback to whatever chunks are present
            first_key = list(data.get("chunks", {}).keys())[0]
            raw_chunks = data["chunks"][first_key]

        for item in raw_chunks:
            doc = Document(
                page_content=item["text"],
                metadata={
                    "source": "Graphs-Complete-Course.pdf",
                    "chunk_id": item.get("chunk_id", 0),
                    "char_count": item.get("char_count", len(item["text"])),
                    "strategy": item.get("strategy", "recursive")
                }
            )
            documents.append(doc)
    else:
        print("chunks_output.json not found. Extracting directly from pdf_data.py...")
        import pdf_data
        from chunking import recursive_character_chunking
        full_text = pdf_data.get_full_text()
        raw_chunks = recursive_character_chunking(full_text, chunk_size=800, overlap=150)
        for item in raw_chunks:
            documents.append(
                Document(
                    page_content=item.text,
                    metadata={
                        "source": "Graphs-Complete-Course.pdf",
                        "chunk_id": item.chunk_id,
                        "char_count": item.char_count
                    }
                )
            )

    return documents


def build_database(db_path: str = "./db", provider: str = "auto"):
    """Creates or updates the Chroma vector database in db_path."""
    os.makedirs(db_path, exist_ok=True)
    embeddings, prov_name, model_name = get_embedding_function(provider)

    docs = load_chunks_for_indexing()
    print(f"Total documents to embed and index: {len(docs)}")

    print(f"Building Chroma vector database at '{db_path}'...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )

    # Save embedding configuration info so server.py knows which model was used
    info_path = os.path.join(db_path, "embedding_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({
            "provider": prov_name,
            "model_name": model_name,
            "total_documents": len(docs)
        }, f, indent=2)

    print(f"Vector database successfully created in '{db_path}' with {len(docs)} documents!")
    print(f"Saved configuration to: {info_path}")
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Chroma vector database from PDF chunks")
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "huggingface"],
        default="auto",
        help="Embedding provider (default: auto; uses openai if key is set, else huggingface)"
    )
    parser.add_argument("--db", default="./db", help="Destination folder for Chroma DB (default: ./db)")
    args = parser.parse_args()

    build_database(db_path=args.db, provider=args.provider)
