//! Embedding providers. MindVault is model-agnostic: anything that can
//! turn text into `Vec<f32>` plugs in via [`Embedder`].

use anyhow::{Context, Result};
use serde::Deserialize;

#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dimensions(&self) -> usize;
}

/// OpenAI-compatible `/v1/embeddings` HTTP embedder. Works with any
/// self-hosted inference server exposing that contract (TEI,
/// llama.cpp-server, vLLM, Ollama via compat layer, ...).
pub struct HttpEmbedder {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    dimensions: usize,
}

impl HttpEmbedder {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>, dimensions: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint: endpoint.into(),
            model: model.into(),
            api_key: None,
            dimensions,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
}

#[async_trait::async_trait]
impl Embedder for HttpEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut req = self.client.post(&self.endpoint).json(&serde_json::json!({
            "model": self.model,
            "input": text,
        }));
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp: EmbeddingResponse = req
            .send()
            .await
            .context("embedding request failed")?
            .error_for_status()
            .context("embedding endpoint returned an error status")?
            .json()
            .await
            .context("embedding response was not valid JSON")?;
        let embedding = resp
            .data
            .into_iter()
            .next()
            .context("embedding response contained no data")?
            .embedding;
        anyhow::ensure!(
            embedding.len() == self.dimensions,
            "expected {} dimensions, got {}",
            self.dimensions,
            embedding.len()
        );
        Ok(embedding)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Deterministic hashing embedder for tests and offline evaluation.
/// Buckets token hashes into a fixed-width L2-normalized vector — not a
/// semantic model, but stable, fast, and dependency-free, which is what
/// integration tests and harness plumbing need.
pub struct HashEmbedder {
    dimensions: usize,
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

#[async_trait::async_trait]
impl Embedder for HashEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(hash_embed(text, self.dimensions))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// FNV-1a token bucketing into a normalized vector.
pub fn hash_embed(text: &str, dimensions: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dimensions.max(1)];
    for token in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        let mut hash: u64 = 0xcbf29ce484222325;
        for b in token.to_lowercase().bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        let idx = (hash % dimensions.max(1) as u64) as usize;
        let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
        v[idx] += sign;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embed_is_deterministic_and_normalized() {
        let a = hash_embed("the quick brown fox", 64);
        let b = hash_embed("the quick brown fox", 64);
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn similar_texts_are_closer_than_dissimilar() {
        let cos = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let q = hash_embed("rust borrow checker memory", 256);
        let near = hash_embed("memory safety and the rust borrow checker", 256);
        let far = hash_embed("simmer the tomato sauce slowly", 256);
        assert!(cos(&q, &near) > cos(&q, &far));
    }

    #[tokio::test]
    async fn hash_embedder_implements_trait() {
        let e = HashEmbedder::new(32);
        let v = e.embed("hello world").await.unwrap();
        assert_eq!(v.len(), 32);
        assert_eq!(e.dimensions(), 32);
    }
}
