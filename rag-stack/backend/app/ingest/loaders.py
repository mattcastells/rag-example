from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".html", ".htm"}


@dataclass
class LoadedDocument:
    path: Path
    content: str
    mime: str
    metadata: dict


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    return soup.get_text(separator="\n")


def detect_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".md", ".markdown"}:
        return "text/markdown"
    if suffix in {".html", ".htm"}:
        return "text/html"
    return "text/plain"


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        content = load_pdf(path)
    elif suffix in {".md", ".markdown", ".txt"}:
        content = load_text(path)
    elif suffix in {".html", ".htm"}:
        content = load_html(path)
    else:
        raise ValueError(f"Unsupported extension: {suffix}")
    return LoadedDocument(path=path, content=content, mime=detect_mime(path), metadata={})


def iter_documents(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_from_directory(root: Path) -> List[LoadedDocument]:
    return [load_document(path) for path in iter_documents(root)]
