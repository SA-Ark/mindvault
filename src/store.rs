//! PostgreSQL + pgvector persistence and the hybrid recall pipeline.

use crate::fuse::{self, RRF_K};
use crate::model::{Memory, ScoredMemory};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use pgvector::Vector;
use sha2::{Digest, Sha256};
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use std::collections::HashMap;

/// Over-fetch factor per retrieval leg before fusion.
const LEG_FETCH_FACTOR: usize = 3;

pub struct MemoryStore {
    pool: PgPool,
}

const MEMORY_COLS: &str = "m.id, m.content_hash, m.content, m.memory_type, m.tags, \
     m.importance, m.access_count, \
     extract(epoch from m.last_accessed)::bigint AS last_accessed, \
     extract(epoch from m.created_at)::bigint AS created_at, \
     extract(epoch from m.updated_at)::bigint AS updated_at";

fn row_to_memory(row: &sqlx::postgres::PgRow) -> Memory {
    Memory {
        id: row.get("id"),
        content_hash: row.get("content_hash"),
        content: row.get("content"),
        memory_type: row.get("memory_type"),
        tags: row.try_get::<Vec<String>, _>("tags").unwrap_or_default(),
        importance: row.get("importance"),
        access_count: row.get("access_count"),
        last_accessed: row.try_get("last_accessed").ok(),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }
}

pub fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

impl MemoryStore {
    /// Connect and apply the schema migration (idempotent).
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(8)
            .connect(database_url)
            .await
            .context("failed to connect to PostgreSQL")?;
        sqlx::raw_sql(include_str!("../migrations/0001_init.sql"))
            .execute(&pool)
            .await
            .context("schema migration failed")?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Store a memory with its embedding. Content-addressed: identical
    /// content upserts rather than duplicating. Returns the memory id.
    pub async fn store(
        &self,
        content: &str,
        memory_type: Option<&str>,
        tags: &[String],
        importance: f64,
        embedding: &[f32],
    ) -> Result<i64> {
        let hash = content_hash(content);
        let row = sqlx::query(
            "INSERT INTO memories (content_hash, content, memory_type, tags, importance) \
             VALUES ($1, $2, $3, $4, $5) \
             ON CONFLICT (content_hash) DO UPDATE \
                 SET memory_type = EXCLUDED.memory_type, \
                     tags = EXCLUDED.tags, \
                     importance = EXCLUDED.importance, \
                     updated_at = now(), \
                     deleted_at = NULL \
             RETURNING id",
        )
        .bind(&hash)
        .bind(content)
        .bind(memory_type)
        .bind(tags)
        .bind(importance)
        .fetch_one(&self.pool)
        .await?;
        let id: i64 = row.get("id");

        sqlx::query(
            "INSERT INTO memory_embeddings (memory_id, embedding) VALUES ($1, $2) \
             ON CONFLICT (memory_id) DO UPDATE SET embedding = EXCLUDED.embedding",
        )
        .bind(id)
        .bind(Vector::from(embedding.to_vec()))
        .execute(&self.pool)
        .await?;

        Ok(id)
    }

    pub async fn get_by_hash(&self, hash: &str) -> Result<Option<Memory>> {
        let row = sqlx::query(&format!(
            "SELECT {MEMORY_COLS} FROM memories m \
             WHERE m.content_hash = $1 AND m.deleted_at IS NULL"
        ))
        .bind(hash)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.as_ref().map(row_to_memory))
    }

    /// Soft delete. Returns whether a live memory was deleted.
    pub async fn delete(&self, hash: &str) -> Result<bool> {
        let result = sqlx::query(
            "UPDATE memories SET deleted_at = now() \
             WHERE content_hash = $1 AND deleted_at IS NULL",
        )
        .bind(hash)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn count(&self) -> Result<i64> {
        let row = sqlx::query("SELECT count(*) AS n FROM memories WHERE deleted_at IS NULL")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.get("n"))
    }

    /// Semantic leg: cosine ANN over pgvector, best-first.
    pub async fn search_semantic(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(Memory, f64)>> {
        let rows = sqlx::query(&format!(
            "SELECT {MEMORY_COLS}, 1.0 - (e.embedding <=> $1::vector) AS similarity \
             FROM memories m \
             JOIN memory_embeddings e ON e.memory_id = m.id \
             WHERE m.deleted_at IS NULL \
             ORDER BY e.embedding <=> $1::vector \
             LIMIT $2"
        ))
        .bind(Vector::from(embedding.to_vec()))
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|row| {
                let similarity: f64 = row.get("similarity");
                (row_to_memory(row), similarity)
            })
            .collect())
    }

    /// Keyword leg: PostgreSQL full-text search ranked by `ts_rank`.
    pub async fn search_keyword(&self, query: &str, limit: usize) -> Result<Vec<(Memory, f64)>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let rows = sqlx::query(&format!(
            "SELECT {MEMORY_COLS}, \
                    ts_rank(m.search_vector, plainto_tsquery('english', $1)) AS rank \
             FROM memories m \
             WHERE m.search_vector @@ plainto_tsquery('english', $1) \
               AND m.deleted_at IS NULL \
             ORDER BY rank DESC \
             LIMIT $2"
        ))
        .bind(query)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|row| {
                let rank: f32 = row.get("rank");
                (row_to_memory(row), rank as f64)
            })
            .collect())
    }

    /// Graph neighbors of a memory (both directions), strongest first.
    pub async fn neighbors(&self, memory_id: i64, limit: usize) -> Result<Vec<Memory>> {
        let rows = sqlx::query(&format!(
            "SELECT {MEMORY_COLS} FROM memories m \
             JOIN ( \
                 SELECT target_id AS other, weight FROM memory_edges WHERE source_id = $1 \
                 UNION ALL \
                 SELECT source_id AS other, weight FROM memory_edges WHERE target_id = $1 \
             ) edges ON edges.other = m.id \
             WHERE m.deleted_at IS NULL \
             ORDER BY edges.weight DESC \
             LIMIT $2"
        ))
        .bind(memory_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(row_to_memory).collect())
    }

    /// Link two memories in the recall graph (undirected semantics:
    /// stored once, queried both ways).
    pub async fn link(&self, source_id: i64, target_id: i64, weight: f64) -> Result<()> {
        sqlx::query(
            "INSERT INTO memory_edges (source_id, target_id, weight) VALUES ($1, $2, $3) \
             ON CONFLICT (source_id, target_id, edge_type) \
             DO UPDATE SET weight = EXCLUDED.weight",
        )
        .bind(source_id)
        .bind(target_id)
        .bind(weight)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Memories linked to KG entities whose name matches `query`.
    pub async fn entity_linked_memories(&self, query: &str, limit: usize) -> Result<Vec<Memory>> {
        let rows = sqlx::query(&format!(
            "SELECT DISTINCT {MEMORY_COLS} FROM memories m \
             JOIN kg_entity_memories em ON em.memory_id = m.id \
             JOIN kg_entities e ON e.id = em.entity_id \
             WHERE e.deleted_at IS NULL AND m.deleted_at IS NULL \
               AND e.name ILIKE '%' || $1 || '%' \
             LIMIT $2"
        ))
        .bind(query)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.iter().map(row_to_memory).collect())
    }

    /// The full hybrid recall pipeline:
    ///
    /// 1. semantic leg (pgvector cosine) — over-fetched 3×
    /// 2. keyword leg (FTS `ts_rank`) — over-fetched 3×
    /// 3. RRF fusion of both rankings (`k = 60`)
    /// 4. graph-neighbor boost for neighbors of top semantic hits
    /// 5. entity-aware boost for memories linked to query-matched entities
    /// 6. time-decay multiplier + importance bonus, deterministic sort
    pub async fn hybrid_recall(
        &self,
        query: &str,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let fetch = limit.saturating_mul(LEG_FETCH_FACTOR).max(1);
        let semantic = self.search_semantic(embedding, fetch).await?;
        let keyword = self.search_keyword(query, fetch).await?;

        let mut fused: HashMap<i64, (Memory, f64)> = HashMap::new();
        fuse::fuse_ranked(&mut fused, &semantic, RRF_K);
        fuse::fuse_ranked(&mut fused, &keyword, RRF_K);

        for (rank, (memory, _)) in semantic.iter().take(limit).enumerate() {
            for neighbor in self.neighbors(memory.id, 5).await? {
                fuse::boost_neighbor(&mut fused, &neighbor, rank);
            }
        }

        for memory in self.entity_linked_memories(query, fetch).await? {
            fuse::boost_entity(&mut fused, &memory);
        }

        let results = fuse::finalize(fused, Utc::now().timestamp(), limit);
        self.record_access(&results).await?;
        Ok(results)
    }

    /// Touch access stats for returned memories (drives decay freshness
    /// and importance signals).
    async fn record_access(&self, results: &[ScoredMemory]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }
        let ids: Vec<i64> = results.iter().map(|r| r.memory.id).collect();
        sqlx::query(
            "UPDATE memories \
             SET access_count = access_count + 1, last_accessed = now() \
             WHERE id = ANY($1)",
        )
        .bind(&ids)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Timestamp helper for diagnostics.
    pub fn now() -> DateTime<Utc> {
        Utc::now()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_is_stable_sha256() {
        assert_eq!(content_hash("hello"), content_hash("hello"));
        assert_ne!(content_hash("hello"), content_hash("hello "));
        assert_eq!(content_hash("hello").len(), 64);
    }
}
