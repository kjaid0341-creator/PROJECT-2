#!/usr/bin/env python3
"""
main.py

End-to-End Pipeline:
1. Verifies/loads data from the Python file (`pdf_data.py`) generated from the PDF.
2. Demonstrates and evaluates multiple chunking methods:
   - Strategy 1: Fixed-Size Character Chunking with Overlap
   - Strategy 2: Recursive Character Chunking
   - Strategy 3: Sentence-Based Chunking
   - Strategy 4: Section-Based / Semantic Chunking
   - Strategy 5: Token-Based Chunking (tiktoken)
3. Outputs statistics and saves chunks to JSON for downstream RAG / vector DB usage.
"""

import os
import json
import argparse
from typing import Dict, List, Any

# Step 1: Load data from the generated Python file
try:
    import pdf_data
except ImportError:
    print("pdf_data.py not found. Running conversion first...")
    from convert_pdf_to_py import convert_pdf_to_python
    convert_pdf_to_python("Graphs-Complete-Course.pdf", "pdf_data.py")
    import pdf_data

from chunking import (
    fixed_size_chunking,
    recursive_character_chunking,
    sentence_chunking,
    section_based_chunking,
    token_based_chunking,
    Chunk
)


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()} ")
    print("=" * 70)


def display_strategy_results(strategy_name: str, chunks: List[Chunk]):
    print(f"\n--- Strategy: {strategy_name} ---")
    print(f"Total Chunks Generated : {len(chunks)}")
    if not chunks:
        return

    avg_chars = sum(c.char_count for c in chunks) / len(chunks)
    avg_tokens = sum(c.token_count or 0 for c in chunks) / len(chunks)
    print(f"Average Character Length: {avg_chars:.1f} chars")
    print(f"Average Token Length    : {avg_tokens:.1f} tokens")

    # Sample preview
    sample_chunk = chunks[0]
    print(f"\n[Sample Preview - Chunk #1 | {sample_chunk.char_count} chars | {sample_chunk.token_count} tokens]:")
    print("-" * 50)
    preview_lines = sample_chunk.text.splitlines()[:6]
    print("\n".join(preview_lines))
    if len(sample_chunk.text.splitlines()) > 6:
        print("... [truncated] ...")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="PDF to Python & Text Chunking Pipeline")
    parser.add_argument(
        "--strategy",
        choices=["all", "fixed", "recursive", "sentence", "section", "token"],
        default="all",
        help="Chunking strategy to run (default: all)"
    )
    parser.add_argument("--chunk_size", type=int, default=800, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap in characters")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens for token chunking")
    parser.add_argument("--output", type=str, default="chunks_output.json", help="Path to export chunks JSON")
    args = parser.parse_args()

    print_banner("1. Loading Data from Python File (pdf_data.py)")
    metadata = pdf_data.METADATA
    pages = pdf_data.PAGES
    full_text = pdf_data.get_full_text()

    print(f"Source Document : {metadata.get('source_file')}")
    print(f"Total Pages      : {metadata.get('total_pages')}")
    print(f"Total Characters : {len(full_text):,}")
    print(f"First Page Excerpt:")
    print("  " + pages[0]['text'].replace("\n", " ")[:120] + "...")

    results: Dict[str, List[Dict[str, Any]]] = {}

    print_banner("2. Applying Chunking Strategies")

    # Strategy 1: Fixed-Size
    if args.strategy in ["all", "fixed"]:
        fixed_chunks = fixed_size_chunking(
            full_text,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            metadata={"source": metadata.get("source_file")}
        )
        display_strategy_results("Fixed-Size Chunking", fixed_chunks)
        results["fixed_size"] = [c.to_dict() for c in fixed_chunks]

    # Strategy 2: Recursive Character
    if args.strategy in ["all", "recursive"]:
        rec_chunks = recursive_character_chunking(
            full_text,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            metadata={"source": metadata.get("source_file")}
        )
        display_strategy_results("Recursive Character Chunking", rec_chunks)
        results["recursive_character"] = [c.to_dict() for c in rec_chunks]

    # Strategy 3: Sentence-Based
    if args.strategy in ["all", "sentence"]:
        sent_chunks = sentence_chunking(
            full_text,
            max_chunk_size=args.chunk_size,
            overlap_sentences=1,
            metadata={"source": metadata.get("source_file")}
        )
        display_strategy_results("Sentence-Based Chunking", sent_chunks)
        results["sentence_based"] = [c.to_dict() for c in sent_chunks]

    # Strategy 4: Section-Based
    if args.strategy in ["all", "section"]:
        sec_chunks = section_based_chunking(
            full_text,
            metadata={"source": metadata.get("source_file")}
        )
        display_strategy_results("Section / Structural Chunking", sec_chunks)
        results["section_based"] = [c.to_dict() for c in sec_chunks]

    # Strategy 5: Token-Based
    if args.strategy in ["all", "token"]:
        tok_chunks = token_based_chunking(
            full_text,
            max_tokens=args.max_tokens,
            overlap_tokens=32,
            metadata={"source": metadata.get("source_file")}
        )
        display_strategy_results("Token-Based Chunking (tiktoken)", tok_chunks)
        results["token_based"] = [c.to_dict() for c in tok_chunks]

    # Export to JSON
    print_banner("3. Exporting Chunks")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "document_metadata": metadata,
                "summary": {k: len(v) for k, v in results.items()},
                "chunks": results
            },
            f,
            indent=2
        )
    print(f"All generated chunks saved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.2f} KB")


if __name__ == "__main__":
    main()
