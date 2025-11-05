from __future__ import annotations

from typing import Iterable

from ..config import Settings
from ..models import RagChunk


SYSTEM_PROMPT = """Eres un asistente experto que responde únicamente con la información proporcionada en los documentos."
NO_FALLBACK = "Si la respuesta no se encuentra en los documentos, responde que no tienes suficiente información."
FORMAT_INSTRUCTIONS = "Formatea la respuesta en español utilizando viñetas concisas y cita la fuente entre paréntesis usando el formato path#fragment."
"""


def build_prompt(settings: Settings, question: str, chunks: Iterable[RagChunk]) -> str:
    bullet_lines = []
    for chunk in chunks:
        path = chunk.path or "desconocido"
        preview = chunk.content.replace("\n", " ")
        preview = preview[:500]
        bullet_lines.append(f"Fuente: {path}\nContenido: {preview}")
    context = "\n\n".join(bullet_lines)
    prompt = (
        f"{SYSTEM_PROMPT}\n{NO_FALLBACK}\n{FORMAT_INSTRUCTIONS}\n"
        f"Idioma objetivo: {settings.response_language}.\n"
        f"Documentos de contexto:\n{context}\n\nPregunta: {question}"
    )
    return prompt
