from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except Exception:  # pragma: no cover
    from sqlalchemy.types import TypeDecorator

    class Vector(TypeDecorator):
        impl = JSON
        cache_ok = True

        def process_bind_param(self, value, dialect):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return list(value)
            raise TypeError("Vector column expects a sequence")

        def process_result_value(self, value, dialect):
            return value


class Base(DeclarativeBase):
    pass


try:
    from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY  # type: ignore

    ACLType = PG_ARRAY(String)  # type: ignore
except Exception:  # pragma: no cover
    ACLType = JSON


class RagChunk(Base):
    __tablename__ = "rag_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Sequence[float] | None] = mapped_column(Vector(1024), nullable=True)
    path: Mapped[str | None] = mapped_column(String, nullable=True)
    mime: Mapped[str | None] = mapped_column(String, nullable=True)
    repo: Mapped[str | None] = mapped_column(String, nullable=True)
    tag: Mapped[str | None] = mapped_column(String, nullable=True)
    version: Mapped[str | None] = mapped_column(String, nullable=True)
    acl: Mapped[list[str] | None] = mapped_column(ACLType, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

