"""create rag_chunks table

Revision ID: 0001
Revises: 
Create Date: 2024-01-01 00:00:00

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "rag_chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", postgresql.VECTOR(1024), nullable=True),
        sa.Column("path", sa.Text(), nullable=True),
        sa.Column("mime", sa.Text(), nullable=True),
        sa.Column("repo", sa.Text(), nullable=True),
        sa.Column("tag", sa.Text(), nullable=True),
        sa.Column("version", sa.Text(), nullable=True),
        sa.Column("acl", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "idx_chunks_embedding",
        "rag_chunks",
        ["embedding"],
        postgresql_using="ivfflat",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("idx_chunks_meta", "rag_chunks", ["meta"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("idx_chunks_meta", table_name="rag_chunks")
    op.drop_index("idx_chunks_embedding", table_name="rag_chunks")
    op.drop_table("rag_chunks")
