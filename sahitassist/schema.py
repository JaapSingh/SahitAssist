from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
import hashlib

class TokenAnnotation(BaseModel):
    surface: str
    lemma: Optional[str] = None
    pos: Optional[str] = None
    literal_en: Optional[str] = None
    note: Optional[str] = None

class SentenceRecord(BaseModel):
    book_id: str
    page_number: int
    paragraph_index: int
    sentence_index: int
    original: str
    normalized: str
    tokens: List[TokenAnnotation]
    literal_sentence_en: Optional[str]
    sentence_translation_en: Optional[str]
    context_explanation_en: Optional[str]
    metadata: dict = Field(default_factory=dict)

    def compute_checksum(self) -> str:
        h = hashlib.sha256()
        h.update(self.normalized.encode("utf-8"))
        h.update(self.original.encode("utf-8"))
        return "sha256:" + h.hexdigest()

    @classmethod
    def create_base(cls, *, book_id: str, page_number: int,
                    paragraph_index: int, sentence_index: int,
                    original: str, normalized: str) -> "SentenceRecord":
        return cls(
            book_id=book_id,
            page_number=page_number,
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            original=original,
            normalized=normalized,
            tokens=[],
            literal_sentence_en=None,
            sentence_translation_en=None,
            context_explanation_en=None,
            metadata={"uuid": str(uuid.uuid4())},
        )