use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Mutex;

/// Node score used for reputation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeScore {
    /// Overall reputation score (0-1)
    pub score: f32,
    /// Overall score (0-1) (same as score for compatibility)
    pub overall_score: f32,
    /// Response time metrics
    pub response_time: f32,
    /// Last update time
    pub last_update: SystemTime,
    /// Last updated time (same as last_update for compatibility)
    pub last_updated: SystemTime,
    /// Connection reliability (0-1)
    pub reliability: f32,
    /// Number of fulfilled requests
    pub fulfilled_requests: u64,
    /// Number of failed requests
    pub failed_requests: u64,
    /// Peer ID
    pub peer_id: String,
    /// Device health score (0-1)
    pub device_health_score: f32,
    /// Network score (0-1)
    pub network_score: f32,
    /// Storage score (0-1)
    pub storage_score: f32,
    /// Engagement score (0-1)
    pub engagement_score: f32,
    /// AI behavior score (0-1)
    pub ai_behavior_score: f32,
    /// Score history (newest first)
    pub history: Vec<(SystemTime, f32)>,
}

impl NodeScore {
    /// Create a new node score with default values
    pub fn new(peer_id: &str) -> Self {
        let now = SystemTime::now();
        Self {
            score: 0.5, // Start with neutral score
            overall_score: 0.5,
            response_time: 0.0,
            last_update: now,
            last_updated: now,
            reliability: 1.0,
            fulfilled_requests: 0,
            failed_requests: 0,
            peer_id: peer_id.to_string(),
            device_health_score: 0.7,
            network_score: 0.7,
            storage_score: 0.7,
            engagement_score: 0.7,
            ai_behavior_score: 0.7,
            history: Vec::new(),
        }
    }

    /// Update the score based on successful operation
    pub fn record_success(&mut self, response_time: f32) {
        let now = SystemTime::now();
        self.fulfilled_requests += 1;
        self.response_time = (self.response_time * 0.9) + (response_time * 0.1); // Weighted average
        self.reliability = self.fulfilled_requests as f32
            / (self.fulfilled_requests + self.failed_requests) as f32;
        self.score = self.score * 0.95 + 0.05; // Slowly increase score with each success
        self.overall_score = self.score;
        self.last_update = now;
        self.last_updated = now;
        self.history.push((now, self.score));
    }

    /// Update the score based on failed operation
    pub fn record_failure(&mut self) {
        let now = SystemTime::now();
        self.failed_requests += 1;
        self.reliability = self.fulfilled_requests as f32
            / (self.fulfilled_requests + self.failed_requests) as f32;
        // More significant reduction for failures
        self.score = self.score * 0.9;
        self.overall_score = self.score;
        self.last_update = now;
        self.last_updated = now;
        self.history.push((now, self.score));
    }
}

/// Handler for network message processing
#[derive(Default)]
pub struct MessageHandler {
    /// Node scores for connected peers
    pub peer_scores: Arc<Mutex<HashMap<String, NodeScore>>>,
    /// Known peers
    pub known_peers: Arc<Mutex<HashSet<String>>>,
}

impl MessageHandler {
    /// Create a new message handler
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a peer's score
    pub async fn get_peer_score(&self, peer_id: &str) -> Option<NodeScore> {
        let scores = self.peer_scores.lock().await;
        scores.get(peer_id).cloned()
    }

    /// Update a peer's score with a successful interaction
    pub async fn update_peer_success(&self, peer_id: &str, response_time: f32) {
        let mut scores = self.peer_scores.lock().await;
        let score = scores
            .entry(peer_id.to_string())
            .or_insert_with(|| NodeScore::new(peer_id));
        score.record_success(response_time);
    }

    /// Update a peer's score with a failed interaction
    pub async fn update_peer_failure(&self, peer_id: &str) {
        let mut scores = self.peer_scores.lock().await;
        let score = scores
            .entry(peer_id.to_string())
            .or_insert_with(|| NodeScore::new(peer_id));
        score.record_failure();
    }

    pub fn decay_score(&mut self, score: &mut f32) {
        *score *= 0.9;
    }
}
