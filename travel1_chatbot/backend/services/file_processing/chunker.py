from typing import List, Dict, Optional

class Chunker:
    """Utility for splitting long text into overlapping chunks and
    expanding extracted segments into sub-chunks while preserving metadata.
    """

    def __init__(self, max_chars: int = 220, overlap: int = 20) -> None:
        self.max_chars = max_chars
        self.overlap = overlap

    def split_long_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= self.max_chars:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - self.overlap)
        return chunks

    def chunk_segments(self, segments: List[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
        """Expand segments from text_extractor into sub-chunks with metadata.
        Each returned dict contains: {"text_chunk": str, "source": str|None, "page": int|None}
        """
        out: List[Dict[str, Optional[str]]] = []
        for seg in segments:
            source = seg.get("source")
            page = seg.get("page")
            for sub in self.split_long_text(seg.get("text", "")):
                out.append({
                    "text_chunk": sub,
                    "source": source,
                    "page": page,
                })
        return out



