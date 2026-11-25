#!/usr/bin/env python3
"""
convert_pdf_to_py.py

Extracts text and metadata from a PDF file (Graphs-Complete-Course.pdf)
and converts it directly into a standalone Python file (`pdf_data.py`).
"""

import sys
import os
import reprlib
import pypdf

def convert_pdf_to_python(pdf_path: str, output_py_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Reading PDF: {pdf_path}")
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages found: {total_pages}")

    pages_data = []
    full_text_parts = []

    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages_data.append({
            "page_number": idx,
            "char_count": len(text),
            "text": text
        })
        full_text_parts.append(text)

    full_text = "\n\n--- Page Break ---\n\n".join(full_text_parts)

    print(f"Generating Python file: {output_py_path}")
    with open(output_py_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""\n')
        f.write(f"Auto-generated Python representation of {os.path.basename(pdf_path)}\n")
        f.write("Contains extracted page-level data and full document text.\n")
        f.write('"""\n\n')

        # Metadata
        f.write("METADATA = {\n")
        f.write(f'    "source_file": {repr(os.path.basename(pdf_path))},\n')
        f.write(f'    "total_pages": {total_pages},\n')
        f.write(f'    "total_characters": {len(full_text)},\n')
        f.write("}\n\n")

        # Pages list
        f.write("# List of all extracted pages with page_number and text\n")
        f.write("PAGES = [\n")
        for p in pages_data:
            f.write("    {\n")
            f.write(f'        "page_number": {p["page_number"]},\n')
            f.write(f'        "char_count": {p["char_count"]},\n')
            f.write(f'        "text": {repr(p["text"])},\n')
            f.write("    },\n")
        f.write("]\n\n")

        # Function to retrieve full text
        f.write("def get_full_text() -> str:\n")
        f.write('    """Returns the full text of all pages joined by page breaks."""\n')
        f.write('    return "\\n\\n--- Page Break ---\\n\\n".join(page["text"] for page in PAGES)\n\n')

        f.write("def get_page(page_number: int) -> dict:\n")
        f.write('    """Returns the data dictionary for a specific page (1-indexed)."""\n')
        f.write('    if 1 <= page_number <= len(PAGES):\n')
        f.write('        return PAGES[page_number - 1]\n')
        f.write('    raise IndexError(f"Page number {page_number} out of range (1 to {len(PAGES)})")\n\n')

        # Self-test block if executed directly
        f.write('if __name__ == "__main__":\n')
        f.write('    print(f"Loaded {METADATA[\'source_file\']} with {METADATA[\'total_pages\']} pages.")\n')
        f.write('    print(f"Total characters: {METADATA[\'total_characters\']}")\n')
        f.write('    print("Page 1 preview:")\n')
        f.write('    print(PAGES[0]["text"][:200])\n')

    file_size_kb = os.path.getsize(output_py_path) / 1024
    print(f"Successfully converted PDF to {output_py_path} ({file_size_kb:.2f} KB)")


if __name__ == "__main__":
    pdf_file = "Graphs-Complete-Course.pdf"
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    output_file = "pdf_data.py"
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    convert_pdf_to_python(pdf_file, output_file)
