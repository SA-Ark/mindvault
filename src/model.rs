//! Core data model: memories, scores, and the time-decay policy.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: i64,
    pub content_hash: String,
    pub content: String,
    pub memory_type: Option<String>,
    pub tags: Vec<String>,
    pub importance: f64,
    pub access_count: i64,
    /// Unix seconds of last read access, if any.
    pub last_accessed: Option<i64>,
    /// Unix seconds.
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub score: f64,
}

/// Per-type hourly decay rate. Stable knowledge (facts, lessons,
/// architecture) decays slowly; operational chatter (notes, logs) decays
/// fast. These constants were tuned in production against real recall
/// quality, not picked aesthetically.
pub fn decay_lambda(memory_type: Option<&str>) -> f64 {
    match memory_type.unwrap_or_default() {
        "fact" | "architecture" | "reference" => 0.01,
        "decision" => 0.02,
        "lesson" => 0.005,
        "note" | "observation" => 0.03,
        "workflow" | "log_summary" => 0.05,
        "digest" => 0.02,
        _ => 0.02,
    }
}

/// Multiplier in `(0, 1]` applied to a memory's fused score:
/// `(1 - lambda)^hours_since_last_touch`. A freshly touched memory is
/// unaffected; a stale operational note fades out of the top-k.
pub fn decay_multiplier(memory: &Memory, now_unix: i64) -> f64 {
    let last_touch = memory.last_accessed.unwrap_or(memory.updated_at);
    let hours = ((now_unix - last_touch).max(0)) as f64 / 3600.0;
    let lambda = decay_lambda(memory.memory_type.as_deref());
    (1.0 - lambda).powf(hours)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory(memory_type: &str, last_accessed: i64) -> Memory {
        Memory {
            id: 1,
            content_hash: "h".into(),
            content: "c".into(),
            memory_type: Some(memory_type.into()),
            tags: vec![],
            importance: 0.0,
            access_count: 0,
            last_accessed: Some(last_accessed),
            created_at: 0,
            updated_at: 0,
        }
    }

    #[test]
    fn lessons_decay_slowest_workflows_fastest() {
        assert!(decay_lambda(Some("lesson")) < decay_lambda(Some("fact")));
        assert!(decay_lambda(Some("fact")) < decay_lambda(Some("workflow")));
    }

    #[test]
    fn fresh_memory_multiplier_is_one() {
        let now = 1_700_000_000;
        let m = memory("note", now);
        assert!((decay_multiplier(&m, now) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn week_old_note_fades_week_old_lesson_persists() {
        let now = 1_700_000_000;
        let week_ago = now - 7 * 24 * 3600;
        let note = decay_multiplier(&memory("note", week_ago), now);
        let lesson = decay_multiplier(&memory("lesson", week_ago), now);
        assert!(note < 0.01, "note multiplier {note}");
        assert!(lesson > 0.4, "lesson multiplier {lesson}");
    }

    #[test]
    fn future_timestamps_clamp_to_one() {
        let now = 1_700_000_000;
        let m = memory("note", now + 9999);
        assert!((decay_multiplier(&m, now) - 1.0).abs() < 1e-9);
    }
}
