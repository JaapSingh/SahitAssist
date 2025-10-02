from __future__ import annotations
import regex
from typing import List, Tuple

DANDA = "।"

SENTENCE_DELIMS = regex.compile(r"([.!?%s])+" % DANDA)

def split_into_paragraphs(text: str) -> List[str]:
    # Paragraph split on blank lines or heavy whitespace breaks
    paras = []
    buf = []
    for line in text.splitlines():
        if line.strip() == "":
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        paras.append(" ".join(buf).strip())
    return paras

def split_paragraph_into_sentences(paragraph: str) -> List[str]:
    # Use regex split while preserving boundaries
    parts = SENTENCE_DELIMS.split(paragraph)
    sentences = []
    cur = []
    for part in parts:
        if part is None:
            continue
        if SENTENCE_DELIMS.fullmatch(part):
            cur.append(part)
            sent = "".join(cur).strip()
            if sent:
                sentences.append(sent)
            cur = []
        else:
            cur.append(part)
    if cur:
        trailing = "".join(cur).strip()
        if trailing:
            sentences.append(trailing)
    # Clean up whitespace
    return [regex.sub(r"\s+", " ", s).strip() for s in sentences if s.strip()]

def segment_text(normalized_text: str) -> List[Tuple[int, int, str]]:
    """
    Returns list of tuples: (paragraph_index, sentence_index_in_paragraph, sentence)
    """
    paragraphs = split_into_paragraphs(normalized_text)
    results = []
    for p_idx, para in enumerate(paragraphs):
        sentences = split_paragraph_into_sentences(para)
        for s_idx, sentence in enumerate(sentences):
            results.append((p_idx, s_idx, sentence))
    return results