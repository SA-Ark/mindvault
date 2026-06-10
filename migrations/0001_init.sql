-- MindVault schema: memories + embeddings + knowledge graph.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id              BIGSERIAL PRIMARY KEY,
    content_hash    TEXT NOT NULL UNIQUE,
    content         TEXT NOT NULL,
    memory_type     TEXT,
    tags            TEXT[],
    importance      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    access_count    BIGINT NOT NULL DEFAULT 0,
    last_accessed   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at      TIMESTAMPTZ,
    search_vector   TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING gin (search_vector);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type) WHERE deleted_at IS NULL;

-- Embeddings live in their own table so dimensionality / model swaps
-- never rewrite the memories table.
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id   BIGINT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    embedding   VECTOR NOT NULL
);

-- HNSW index: build AFTER deciding dimensionality, e.g.
--   CREATE INDEX idx_embeddings_hnsw ON memory_embeddings
--   USING hnsw (embedding vector_cosine_ops);

-- Knowledge graph: typed entities with observations, linked to memories.
CREATE TABLE IF NOT EXISTS kg_entities (
    id           BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    entity_type  TEXT,
    context      TEXT NOT NULL DEFAULT 'default',
    observations TEXT[] NOT NULL DEFAULT '{}',
    importance   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at   TIMESTAMPTZ,
    UNIQUE (name, context)
);

CREATE TABLE IF NOT EXISTS kg_relations (
    id            BIGSERIAL PRIMARY KEY,
    from_entity   BIGINT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    to_entity     BIGINT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (from_entity, to_entity, relation_type)
);

CREATE TABLE IF NOT EXISTS kg_entity_memories (
    entity_id   BIGINT NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    memory_id   BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, memory_id)
);

-- Memory-to-memory graph edges (similarity / co-access links) used by
-- the recall pipeline's neighbor boost.
CREATE TABLE IF NOT EXISTS memory_edges (
    source_id  BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id  BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    weight     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    edge_type  TEXT NOT NULL DEFAULT 'related',
    PRIMARY KEY (source_id, target_id, edge_type)
);
