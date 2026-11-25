"""
chunking.py

A modular library providing multiple text chunking strategies for document processing:
1. Fixed-Size Chunking (sliding window with character overlap)
2. Recursive Character Chunking (hierarchical delimiter splitting via LangChain/custom)
3. Sentence-Based Chunking (natural language sentence boundary splitting)
4. Section / Semantic Chunking (document structure & chapter header splitting)
5. Token-Based Chunking (tiktoken-aligned chunks for LLM context windows)
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Tiktoken integration with caching for high performance
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    TIKTOKEN_AVAILABLE = False
    _TIKTOKEN_ENCODING = None

# LangChain text splitter integration
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False


@dataclass
class Chunk:
    chunk_id: int
    strategy: str
    text: str
    char_count: int
    token_count: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_token_count(text: str) -> int:
    """Calculates token count using cached tiktoken encoding, or 4-char heuristic fallback."""
    if _TIKTOKEN_ENCODING is not None:
        try:
            return len(_TIKTOKEN_ENCODING.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


# -------------------------------------------------------------------------
# 1. Fixed-Size Chunking (Sliding Window)
# -------------------------------------------------------------------------
def fixed_size_chunking(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Chunk]:
    """
    Splits text into fixed character chunks with a sliding window overlap.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap")

    step = chunk_size - overlap
    chunks: List[Chunk] = []
    chunk_id = 1

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    strategy="Fixed-Size",
                    text=chunk_text,
                    char_count=len(chunk_text),
                    token_count=get_token_count(chunk_text),
                    metadata={
                        **(metadata or {}),
                        "start_char": start,
                        "end_char": end,
                        "chunk_size": chunk_size,
                        "overlap": overlap
                    }
                )
            )
            chunk_id += 1
        if end >= len(text):
            break

    return chunks


# -------------------------------------------------------------------------
# 2. Recursive Character Chunking
# -------------------------------------------------------------------------
def recursive_character_chunking(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    separators: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Chunk]:
    """
    Hierarchical text splitting: tries paragraphs -> lines -> sentences -> words.
    Uses LangChain RecursiveCharacterTextSplitter when available, with a clean fallback.
    """
    if separators is None:
        separators = ["\n\n--- Page Break ---\n\n", "\n\n", "\n", ". ", " ", ""]

    if LANGCHAIN_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators,
            length_function=len
        )
        splits = splitter.split_text(text)
    else:
        splits = []
        current = []
        cur_len = 0
        for block in re.split(r'\n{2,}', text):
            block_len = len(block)
            if cur_len + block_len <= chunk_size:
                current.append(block)
                cur_len += block_len + 2
            else:
                if current:
                    splits.append("\n\n".join(current))
                current = [block]
                cur_len = block_len
        if current:
            splits.append("\n\n".join(current))

    chunks: List[Chunk] = []
    for idx, c_text in enumerate(splits, start=1):
        clean_text = c_text.strip()
        if clean_text:
            chunks.append(
                Chunk(
                    chunk_id=idx,
                    strategy="Recursive-Character",
                    text=clean_text,
                    char_count=len(clean_text),
                    token_count=get_token_count(clean_text),
                    metadata={
                        **(metadata or {}),
                        "chunk_size": chunk_size,
                        "overlap": overlap
                    }
                )
            )

    return chunks


# -------------------------------------------------------------------------
# 3. Sentence-Based Chunking
# -------------------------------------------------------------------------
def sentence_chunking(
    text: str,
    max_chunk_size: int = 1000,
    overlap_sentences: int = 1,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Chunk]:
    """
    Splits text on sentence boundaries and bundles complete sentences into chunks.
    Ensures thoughts are never truncated mid-sentence.
    """
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
    raw_sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]

    chunks: List[Chunk] = []
    chunk_id = 1
    i = 0
    total_sentences = len(raw_sentences)

    while i < total_sentences:
        current_sentences = []
        current_len = 0

        j = i
        while j < total_sentences:
            s = raw_sentences[j]
            if current_len + len(s) + 1 <= max_chunk_size or not current_sentences:
                current_sentences.append(s)
                current_len += len(s) + 1
                j += 1
            else:
                break

        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    strategy="Sentence-Based",
                    text=chunk_text,
                    char_count=len(chunk_text),
                    token_count=get_token_count(chunk_text),
                    metadata={
                        **(metadata or {}),
                        "start_sentence_idx": i,
                        "end_sentence_idx": j - 1,
                        "sentence_count": len(current_sentences),
                        "overlap_sentences": overlap_sentences
                    }
                )
            )
            chunk_id += 1

        if j >= total_sentences:
            break
        i = max(i + 1, j - overlap_sentences)

    return chunks


# -------------------------------------------------------------------------
# 4. Section / Semantic Header-Based Chunking
# -------------------------------------------------------------------------
def section_based_chunking(
    text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Chunk]:
    """
    Splits text along document structural sections (e.g. 'Part 1', 'Part 2', 'Contents', etc.).
    Preserves logical pedagogical modules.
    """
    part_pattern = re.compile(r'(?=\n(?:Part\s+\d+|Contents|How to Use This Material))', re.IGNORECASE)
    parts = part_pattern.split(text)

    chunks: List[Chunk] = []
    chunk_id = 1

    for part in parts:
        clean_part = part.strip()
        if not clean_part:
            continue

        lines = clean_part.splitlines()
        title = lines[0].strip() if lines else "Untitled Section"
        if len(lines) > 1 and lines[0].lower().startswith("part"):
            title = f"{lines[0].strip()} - {lines[1].strip()}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                strategy="Section-Based",
                text=clean_part,
                char_count=len(clean_part),
                token_count=get_token_count(clean_part),
                metadata={
                    **(metadata or {}),
                    "section_title": title,
                }
            )
        )
        chunk_id += 1

    return chunks


# -------------------------------------------------------------------------
# 5. Token-Based Chunking
# -------------------------------------------------------------------------
def token_based_chunking(
    text: str,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Chunk]:
    """
    Splits text precisely based on token counts using tiktoken (ideal for LLM prompt limits).
    """
    if _TIKTOKEN_ENCODING is None:
        words = text.split()
        words_per_token = 0.75
        word_chunk = int(max_tokens * words_per_token)
        word_overlap = int(overlap_tokens * words_per_token)
        step = max(1, word_chunk - word_overlap)

        chunks: List[Chunk] = []
        chunk_id = 1
        for i in range(0, len(words), step):
            subset = words[i:i + word_chunk]
            c_text = " ".join(subset)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    strategy="Token-Based (Approx)",
                    text=c_text,
                    char_count=len(c_text),
                    token_count=len(subset),
                    metadata={**(metadata or {}), "token_strategy": "word_approximation"}
                )
            )
            chunk_id += 1
            if i + word_chunk >= len(words):
                break
        return chunks

    tokens = _TIKTOKEN_ENCODING.encode(text)
    total_tokens = len(tokens)
    step = max(1, max_tokens - overlap_tokens)
    chunks = []
    chunk_id = 1

    for start in range(0, total_tokens, step):
        end = min(start + max_tokens, total_tokens)
        chunk_token_slice = tokens[start:end]
        decoded_text = _TIKTOKEN_ENCODING.decode(chunk_token_slice).strip()

        if decoded_text:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    strategy="Token-Based",
                    text=decoded_text,
                    char_count=len(decoded_text),
                    token_count=len(chunk_token_slice),
                    metadata={
                        **(metadata or {}),
                        "token_start": start,
                        "token_end": end,
                        "max_tokens": max_tokens,
                        "overlap_tokens": overlap_tokens
                    }
                )
            )
            chunk_id += 1

        if end >= total_tokens:
            break

    return chunks
