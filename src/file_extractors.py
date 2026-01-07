"""Utilities to extract text content from uploaded files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional


def extract_text_from_file(file_path: str) -> Tuple[bool, str]:
    """Extract text from a supported file.

    Supported:
    - .txt (utf-8 with fallback)
    - .docx (python-docx)
    - .pdf (pypdf)

    Returns:
        (ok, text_or_error)
    """
    p = Path(file_path)
    suffix = p.suffix.lower()

    try:
        if suffix == ".txt":
            return True, p.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".docx":
            try:
                import docx  # type: ignore
            except Exception as e:
                return False, f"python-docx not installed: {e}"
            d = docx.Document(str(p))
            text = "\n".join([para.text for para in d.paragraphs if getattr(para, "text", "").strip()])
            return True, text

        if suffix == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore
            except Exception as e:
                return False, f"pypdf not installed: {e}"
            reader = PdfReader(str(p))
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            return True, "\n".join(pages)

        return False, f"Unsupported file type: {suffix}"
    except Exception as e:
        return False, str(e)

