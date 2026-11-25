"""
demo_quickstart.py

A short, simple example demonstrating how to load the PDF data
from the Python file and perform chunking in just a few lines of code.
"""

# 1. Load the data directly from the converted Python file
import pdf_data
from chunking import recursive_character_chunking, fixed_size_chunking, sentence_chunking

print("=== 1. Loaded PDF Data from Python File ===")
print(f"Document : {pdf_data.METADATA['source_file']}")
print(f"Pages    : {pdf_data.METADATA['total_pages']}")
print(f"Chars    : {pdf_data.METADATA['total_characters']:,}")

# Access full text or individual pages
full_text = pdf_data.get_full_text()
page_10 = pdf_data.get_page(10)
print(f"\nPage 10 length: {page_10['char_count']} characters")

# 2. Perform Chunking (e.g. Recursive Character Chunking)
print("\n=== 2. Performing Recursive Character Chunking ===")
chunks = recursive_character_chunking(full_text, chunk_size=500, overlap=100)
print(f"Generated {len(chunks)} chunks!")

# 3. Inspect the first 3 chunks
for c in chunks[:3]:
    print(f"\n--- [Chunk #{c.chunk_id} | Chars: {c.char_count} | Tokens: {c.token_count}] ---")
    print(c.text[:180] + ("..." if len(c.text) > 180 else ""))
