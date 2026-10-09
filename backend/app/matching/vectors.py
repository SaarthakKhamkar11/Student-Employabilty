"""Embedding similarity for matching: cosine between provider vectors, and similarity against stored vectors.

Stored vectors are compared only with a query vector from the SAME model name and version. A model
switch without re-embedding fails loudly instead of mixing vector spaces (Phase 6: raw similarities of
different models are not comparable).
"""

from typing import Protocol

import numpy as np
import psycopg

from app.db.profile_repository import EmbeddingRow
from app.embeddings.provider import Vector


def to_dense(vector: Vector, dimension: int) -> np.ndarray:
    if vector.indices is None:
        return np.asarray(vector.values, dtype=np.float64)
    out = np.zeros(dimension, dtype=np.float64)
    out[vector.indices] = vector.values
    return out


def cosine(a: Vector | None, b: Vector | None, dimension: int) -> float | None:
    """Cosine of two vectors (None when either carries no signal). Provider vectors are L2-normalised."""
    if a is None or b is None:
        return None
    if a.indices is not None and b.indices is not None:
        bmap = dict(zip(b.indices, b.values))
        return float(sum(v * bmap.get(i, 0.0) for i, v in zip(a.indices, a.values)))
    return float(to_dense(a, dimension) @ to_dense(b, dimension))


class VectorIndexError(RuntimeError):
    """Stored vectors are missing for the configured model (run the re-embedding backfill)."""


class VectorIndex(Protocol):
    def similarities(self, entity_type: str, section: str, query: Vector, model_name: str, model_version: str,
                     dimension: int, ids: list[str] | None = None) -> dict[str, float]: ...

    def count(self, entity_type: str, model_name: str, model_version: str) -> int: ...


class PostgresVectorIndex:
    """pgvector cosine distance (<=>) on the embeddings table; sparse or dense by the model's storage."""

    def __init__(self, dsn: str, connect_timeout: int = 5):
        self._dsn, self._timeout = dsn, connect_timeout

    def similarities(self, entity_type, section, query, model_name, model_version, dimension, ids=None):
        dense_lit, sparse_lit = EmbeddingRow(section, model_name, model_version, dimension, query.values,
                                             query.indices, True).sql_literal()
        column, literal, cast = (("embedding_sparse", sparse_lit, "sparsevec") if sparse_lit
                                 else ("embedding", dense_lit, "vector"))
        sql = (f"SELECT entity_id, 1 - ({column} <=> %s::{cast}) FROM embeddings "
               "WHERE entity_type = %s AND section = %s AND model_name = %s AND model_version = %s "
               f"AND {column} IS NOT NULL" + (" AND entity_id = ANY(%s)" if ids is not None else ""))
        params = [literal, entity_type, section, model_name, model_version] + ([ids] if ids is not None else [])
        with psycopg.connect(self._dsn, connect_timeout=self._timeout) as conn:
            return {row[0]: float(row[1]) for row in conn.execute(sql, params)}

    def count(self, entity_type, model_name, model_version):
        with psycopg.connect(self._dsn, connect_timeout=self._timeout) as conn:
            return conn.execute("SELECT count(DISTINCT entity_id) FROM embeddings WHERE entity_type = %s "
                                "AND model_name = %s AND model_version = %s",
                                (entity_type, model_name, model_version)).fetchone()[0]


class InMemoryVectorIndex:
    """For tests and offline runs: vectors held in memory, same contract as the Postgres index."""

    def __init__(self):
        self._vectors: dict[tuple[str, str, str, str], dict[str, Vector]] = {}

    def add(self, entity_type: str, entity_id: str, section: str, vector: Vector, model_name: str,
            model_version: str) -> None:
        self._vectors.setdefault((entity_type, section, model_name, model_version), {})[entity_id] = vector

    def similarities(self, entity_type, section, query, model_name, model_version, dimension, ids=None):
        stored = self._vectors.get((entity_type, section, model_name, model_version), {})
        wanted = stored if ids is None else {i: stored[i] for i in ids if i in stored}
        return {i: cosine(query, v, dimension) for i, v in wanted.items()}

    def count(self, entity_type, model_name, model_version):
        return len({i for (t, _, m, v), vs in self._vectors.items()
                    if t == entity_type and m == model_name and v == model_version for i in vs})
