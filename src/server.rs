//! HTTP API + embedded demo UI for MindVault.
//!
//! A single self-contained binary that serves a JSON API over the recall
//! pipeline and a static playground UI from one port — the pattern used by
//! the other single-binary Ark services. Self-hosted end to end: the default
//! embedder is the deterministic [`crate::embed::HashEmbedder`], so the demo
//! needs nothing beyond PostgreSQL + pgvector (no third-party embedding SaaS).
//!
//! Routes:
//!   GET  /health                 → liveness
//!   GET  /api/stats              → counts, type breakdown, edge/entity counts
//!   POST /api/ingest             → store one memory {content, memory_type?, importance?}
//!   POST /api/recall             → hybrid recall {query, k?} → scored hits + per-leg explain
//!   GET  /api/recent?limit=      → most-recent memories
//!   GET  /api/graph              → memory + KG graph (nodes + edges) for visualisation
//!   POST /api/seed               → (re)seed the demo corpus (idempotent)
//!   GET  /                       → the playground UI

use crate::embed::Embedder;
use crate::seed;
use crate::store::{content_hash, MemoryStore};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

pub struct AppState {
    pub store: MemoryStore,
    pub embedder: Box<dyn Embedder>,
    pub embedder_name: String,
    pub embed_dim: usize,
}

pub type SharedState = Arc<AppState>;

const INDEX_HTML: &str = include_str!("../static/index.html");

pub fn router(state: SharedState) -> Router {
    Router::new()
        .route("/", get(|| async { Html(INDEX_HTML) }))
        .route("/health", get(health))
        .route("/api/stats", get(stats))
        .route("/api/ingest", post(ingest))
        .route("/api/recall", post(recall))
        .route("/api/recent", get(recent))
        .route("/api/graph", get(graph))
        .route("/api/seed", post(seed_handler))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok", "service": "mindvault" }))
}

#[derive(Serialize)]
struct StatsResponse {
    memories: i64,
    types: Vec<TypeCount>,
    edges: usize,
    entities: usize,
    relations: usize,
    embedder: String,
    embed_dim: usize,
}

#[derive(Serialize)]
struct TypeCount {
    name: String,
    count: i64,
}

async fn stats(State(st): State<SharedState>) -> Result<Json<StatsResponse>, ApiError> {
    let memories = st.store.count().await?;
    let types = st
        .store
        .type_counts()
        .await?
        .into_iter()
        .map(|(name, count)| TypeCount { name, count })
        .collect();
    let edges = st.store.all_edges().await?.len();
    let entities = st.store.all_entities().await?.len();
    let relations = st.store.all_relations().await?.len();
    Ok(Json(StatsResponse {
        memories,
        types,
        edges,
        entities,
        relations,
        embedder: st.embedder_name.clone(),
        embed_dim: st.embed_dim,
    }))
}

#[derive(Deserialize)]
struct IngestRequest {
    content: String,
    #[serde(default)]
    memory_type: Option<String>,
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default)]
    importance: Option<f64>,
}

#[derive(Serialize)]
struct IngestResponse {
    id: i64,
    content_hash: String,
}

async fn ingest(
    State(st): State<SharedState>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, ApiError> {
    let content = req.content.trim();
    if content.is_empty() {
        return Err(ApiError::bad_request("content must not be empty"));
    }
    let emb = st.embedder.embed(content).await?;
    let tags = req.tags.unwrap_or_default();
    let id = st
        .store
        .store(
            content,
            req.memory_type.as_deref(),
            &tags,
            req.importance.unwrap_or(0.5),
            &emb,
        )
        .await?;
    Ok(Json(IngestResponse {
        id,
        content_hash: content_hash(content),
    }))
}

#[derive(Deserialize)]
struct RecallRequest {
    query: String,
    #[serde(default)]
    k: Option<usize>,
}

#[derive(Serialize)]
struct RecallHit {
    id: i64,
    content: String,
    memory_type: Option<String>,
    importance: f64,
    score: f64,
    /// Which legs/boosts surfaced this hit (for the explainability panel).
    signals: Vec<String>,
}

#[derive(Serialize)]
struct RecallResponse {
    query: String,
    k: usize,
    hits: Vec<RecallHit>,
    /// Per-leg diagnostics so the UI can show the pipeline working.
    explain: RecallExplain,
}

#[derive(Serialize)]
struct RecallExplain {
    semantic_top: Vec<LegHit>,
    keyword_top: Vec<LegHit>,
    entity_matches: Vec<String>,
}

#[derive(Serialize)]
struct LegHit {
    id: i64,
    content: String,
    native_score: f64,
}

async fn recall(
    State(st): State<SharedState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<RecallResponse>, ApiError> {
    let query = req.query.trim();
    if query.is_empty() {
        return Err(ApiError::bad_request("query must not be empty"));
    }
    let k = req.k.unwrap_or(5).clamp(1, 25);
    let emb = st.embedder.embed(query).await?;

    // Per-leg diagnostics (separate, lightweight queries purely for the
    // explainability panel — the real fused result comes from hybrid_recall).
    let fetch = k * 3;
    let semantic = st.store.search_semantic(&emb, fetch).await?;
    let keyword = st.store.search_keyword(query, fetch).await?;
    let entity_linked = st.store.entity_linked_memories(query, fetch).await?;
    let entity_ids: std::collections::HashSet<i64> = entity_linked.iter().map(|m| m.id).collect();
    let semantic_ids: std::collections::HashSet<i64> = semantic.iter().map(|(m, _)| m.id).collect();
    let keyword_ids: std::collections::HashSet<i64> = keyword.iter().map(|(m, _)| m.id).collect();

    let hits_raw = st.store.hybrid_recall(query, &emb, k).await?;
    let hits = hits_raw
        .into_iter()
        .map(|sm| {
            let mut signals = Vec::new();
            if semantic_ids.contains(&sm.memory.id) {
                signals.push("semantic".to_string());
            }
            if keyword_ids.contains(&sm.memory.id) {
                signals.push("keyword".to_string());
            }
            if entity_ids.contains(&sm.memory.id) {
                signals.push("entity".to_string());
            }
            if signals.is_empty() {
                signals.push("graph".to_string());
            }
            RecallHit {
                id: sm.memory.id,
                content: sm.memory.content,
                memory_type: sm.memory.memory_type,
                importance: sm.memory.importance,
                score: sm.score,
                signals,
            }
        })
        .collect();

    let explain = RecallExplain {
        semantic_top: semantic
            .iter()
            .take(5)
            .map(|(m, s)| LegHit {
                id: m.id,
                content: m.content.clone(),
                native_score: *s,
            })
            .collect(),
        keyword_top: keyword
            .iter()
            .take(5)
            .map(|(m, s)| LegHit {
                id: m.id,
                content: m.content.clone(),
                native_score: *s,
            })
            .collect(),
        entity_matches: entity_linked.iter().map(|m| m.content.clone()).collect(),
    };

    Ok(Json(RecallResponse {
        query: query.to_string(),
        k,
        hits,
        explain,
    }))
}

#[derive(Deserialize)]
struct RecentParams {
    #[serde(default)]
    limit: Option<usize>,
}

async fn recent(
    State(st): State<SharedState>,
    Query(p): Query<RecentParams>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let limit = p.limit.unwrap_or(20).clamp(1, 200);
    let memories = st.store.recent(limit).await?;
    Ok(Json(json!({ "memories": memories })))
}

#[derive(Serialize)]
struct GraphNode {
    id: String,
    label: String,
    kind: String, // "memory" | "entity"
    memory_type: Option<String>,
}

#[derive(Serialize)]
struct GraphEdge {
    source: String,
    target: String,
    weight: f64,
    kind: String, // "memory" | "entity_link" | "relation"
    label: String,
}

async fn graph(State(st): State<SharedState>) -> Result<Json<serde_json::Value>, ApiError> {
    // Use the recent memories as graph nodes (cap for a readable demo view).
    let memories = st.store.recent(60).await?;
    let mut nodes: Vec<GraphNode> = memories
        .iter()
        .map(|m| GraphNode {
            id: format!("m{}", m.id),
            label: truncate(&m.content, 60),
            kind: "memory".into(),
            memory_type: m.memory_type.clone(),
        })
        .collect();

    let mut edges: Vec<GraphEdge> = st
        .store
        .all_edges()
        .await?
        .into_iter()
        .map(|(s, t, w, et)| GraphEdge {
            source: format!("m{s}"),
            target: format!("m{t}"),
            weight: w,
            kind: "memory".into(),
            label: et,
        })
        .collect();

    // Entities as a second node class + their memory links + relations.
    let entities = st.store.all_entities().await?;
    for (id, name, etype, _obs) in &entities {
        nodes.push(GraphNode {
            id: format!("e{id}"),
            label: name.clone(),
            kind: "entity".into(),
            memory_type: etype.clone(),
        });
    }
    let member_ids: std::collections::HashSet<i64> = memories.iter().map(|m| m.id).collect();
    for m in &memories {
        for eid in st.store.entities_for_memory(m.id).await? {
            edges.push(GraphEdge {
                source: format!("e{eid}"),
                target: format!("m{}", m.id),
                weight: 0.5,
                kind: "entity_link".into(),
                label: "mentions".into(),
            });
        }
    }
    let _ = member_ids;
    for (from, to, rel) in st.store.all_relations().await? {
        edges.push(GraphEdge {
            source: format!("e{from}"),
            target: format!("e{to}"),
            weight: 1.0,
            kind: "relation".into(),
            label: rel,
        });
    }

    Ok(Json(json!({ "nodes": nodes, "edges": edges })))
}

async fn seed_handler(State(st): State<SharedState>) -> Result<Json<serde_json::Value>, ApiError> {
    let count = seed::seed(&st.store, st.embedder.as_ref()).await?;
    Ok(Json(json!({ "seeded": true, "memories": count })))
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push('…');
        out
    }
}

/// Uniform JSON error envelope.
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(msg: &str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.to_string(),
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: e.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}
