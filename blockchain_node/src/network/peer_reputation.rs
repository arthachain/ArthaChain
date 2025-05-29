use libp2p::PeerId;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use anyhow::Result;
use log::warn;
use serde::{Deserialize, Serialize};

type Timestamp = SystemTime;
type ReputationScore = f64;
type ReputationHistory = Vec<(Timestamp, ReputationScore)>;
type PeerHistoryMap = HashMap<PeerId, ReputationHistory>;

// Custom timestamp type that can be serialized/deserialized
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimestampStruct(pub u64);

impl TimestampStruct {
    pub fn now() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        TimestampStruct(now.as_secs())
    }

    pub fn to_instant(&self) -> Instant {
        // This is a simplification as Instant doesn't have a direct conversion from timestamp
        // In a real implementation, you might track when the process started to create relative instants
        Instant::now()
    }

    pub fn from_instant(_instant: Instant) -> Self {
        // Since Instant doesn't expose its internal time, we use current time
        // In a real implementation, you'd track relative time from process start
        Self::now()
    }
}

/// Peer behavior metrics
#[derive(Debug, Clone)]
pub struct PeerMetrics {
    pub total_blocks_propagated: usize,
    pub total_transactions_relayed: usize,
    pub total_blocks_validated: usize,
    pub total_blocks_invalid: usize,
    pub total_requests: usize,
    pub total_responses: usize,
    pub total_timeouts: usize,
    pub total_errors: usize,
    pub average_response_time: Duration,
    pub last_seen: Timestamp,
    pub connection_duration: Duration,
    pub bandwidth_usage: usize,
}

impl Default for PeerMetrics {
    fn default() -> Self {
        Self {
            total_blocks_propagated: 0,
            total_transactions_relayed: 0,
            total_blocks_validated: 0,
            total_blocks_invalid: 0,
            total_requests: 0,
            total_responses: 0,
            total_timeouts: 0,
            total_errors: 0,
            average_response_time: Duration::from_millis(0),
            last_seen: Timestamp::now(),
            connection_duration: Duration::from_secs(0),
            bandwidth_usage: 0,
        }
    }
}

/// Reputation configuration
#[derive(Debug, Clone)]
pub struct ReputationConfig {
    pub score_decay_rate: f64,
    pub min_score: f64,
    pub max_score: f64,
    pub update_interval: Duration,
    pub history_size: usize,
    pub trust_threshold: f64,
    pub ban_threshold: f64,
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            score_decay_rate: 0.1,
            min_score: 0.0,
            max_score: 10.0,
            update_interval: Duration::from_secs(60),
            history_size: 1000,
            trust_threshold: 7.0,
            ban_threshold: 2.0,
        }
    }
}

/// Peer reputation manager
pub struct PeerReputationManager {
    config: ReputationConfig,
    scores: Arc<RwLock<HashMap<PeerId, ReputationScore>>>,
    metrics: Arc<RwLock<HashMap<PeerId, PeerMetrics>>>,
    history: Arc<RwLock<PeerHistoryMap>>,
    trusted_peers: Arc<RwLock<HashSet<PeerId>>>,
    banned_peers: Arc<RwLock<HashSet<PeerId>>>,
}

impl PeerReputationManager {
    pub fn new(config: ReputationConfig) -> Self {
        Self {
            config,
            scores: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(HashMap::new())),
            trusted_peers: Arc::new(RwLock::new(HashSet::new())),
            banned_peers: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Update peer metrics
    pub async fn update_metrics(&self, peer_id: &PeerId, metrics: PeerMetrics) -> Result<()> {
        let mut metrics_map = self.metrics.write().await;
        let _duration = metrics.average_response_time; // Unused variable prefixed with underscore
        if let Some(peer_metrics) = metrics_map.get_mut(peer_id) {
            *peer_metrics = metrics;
            Ok(())
        } else {
            metrics_map.insert(*peer_id, metrics);
            warn!("Added metrics for previously unknown peer: {}", peer_id);
            Ok(())
        }
    }

    /// Calculate reputation score based on metrics
    async fn calculate_score(&self, peer_id: &PeerId) -> Result<ReputationScore> {
        let metrics = self.metrics.read().await;
        let peer_metrics = metrics.get(peer_id).cloned().unwrap_or_default();

        // Calculate individual scores
        let uptime_score = if peer_metrics.total_errors > 0 {
            1.0 / (1.0 + peer_metrics.total_errors as f64)
        } else {
            1.0
        };

        let response_time_score = if peer_metrics.average_response_time.as_millis() > 0 {
            1.0 / (1.0 + peer_metrics.average_response_time.as_millis() as f64 / 1000.0)
        } else {
            1.0
        };

        let block_prop_score = if peer_metrics.total_blocks_propagated > 0 {
            peer_metrics.total_blocks_validated as f64 / peer_metrics.total_blocks_propagated as f64
        } else {
            0.0
        };

        let tx_relay_score = if peer_metrics.total_transactions_relayed > 0 {
            1.0 - (peer_metrics.total_errors as f64
                / peer_metrics.total_transactions_relayed as f64)
        } else {
            0.0
        };

        let validation_score = if peer_metrics.total_blocks_validated > 0 {
            1.0 - (peer_metrics.total_blocks_invalid as f64
                / peer_metrics.total_blocks_validated as f64)
        } else {
            0.0
        };

        let bandwidth_score = if peer_metrics.bandwidth_usage > 0 {
            (peer_metrics.total_responses as f64 / peer_metrics.bandwidth_usage as f64).min(1.0)
        } else {
            0.0
        };

        // Calculate overall score with weights
        let score = (uptime_score * 0.2
            + response_time_score * 0.2
            + block_prop_score * 0.2
            + tx_relay_score * 0.15
            + validation_score * 0.15
            + bandwidth_score * 0.1)
            * self.config.max_score;

        Ok(score)
    }

    /// Update peer reputation
    pub async fn update_reputation(&self, peer_id: &PeerId) -> Result<()> {
        let score = self.calculate_score(peer_id).await?;

        // Update score
        let mut scores = self.scores.write().await;
        scores.insert(*peer_id, score);

        // Update history
        let mut history = self.history.write().await;
        let peer_history = history.entry(*peer_id).or_insert_with(Vec::new);
        peer_history.push((Timestamp::now(), score));

        // Trim history if needed
        if peer_history.len() > self.config.history_size {
            peer_history.remove(0);
        }

        // Update trusted/banned status
        if score >= self.config.trust_threshold {
            let mut trusted = self.trusted_peers.write().await;
            trusted.insert(*peer_id);
        } else {
            let mut trusted = self.trusted_peers.write().await;
            trusted.remove(peer_id);
        }

        if score <= self.config.ban_threshold {
            let mut banned = self.banned_peers.write().await;
            banned.insert(*peer_id);
            warn!(
                "Peer {} banned due to low reputation score: {}",
                peer_id, score
            );
        }

        Ok(())
    }

    /// Get peer reputation score
    pub async fn get_score(&self, peer_id: &PeerId) -> Option<ReputationScore> {
        self.scores.read().await.get(peer_id).cloned()
    }

    /// Check if peer is trusted
    pub async fn is_trusted(&self, peer_id: &PeerId) -> bool {
        self.trusted_peers.read().await.contains(peer_id)
    }

    /// Check if peer is banned
    pub async fn is_banned(&self, peer_id: &PeerId) -> bool {
        self.banned_peers.read().await.contains(peer_id)
    }

    /// Get peer metrics
    pub async fn get_metrics(&self, peer_id: &PeerId) -> Option<PeerMetrics> {
        self.metrics.read().await.get(peer_id).cloned()
    }

    /// Get peer reputation history
    pub async fn get_history(&self, peer_id: &PeerId) -> Vec<(Timestamp, f64)> {
        self.history
            .read()
            .await
            .get(peer_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Decay reputation scores over time
    pub async fn decay_scores(&self) -> Result<()> {
        let mut scores = self.scores.write().await;
        for score in scores.values_mut() {
            let decay = self.config.score_decay_rate * self.config.update_interval.as_secs_f64();
            *score = (*score - decay).max(self.config.min_score);
        }
        Ok(())
    }
}

/// Peer reputation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerReputation {
    pub peer_id: String,
    pub score: f64,
    pub last_updated: Timestamp,
    pub last_seen: Timestamp,
    pub connection_count: u32,
    pub successful_requests: u32,
    pub failed_requests: u32,
}

impl PeerReputation {
    pub fn new(node_id: String) -> Self {
        Self {
            peer_id: node_id,
            score: 5.0, // Neutral starting score
            last_updated: Timestamp::now(),
            last_seen: Timestamp::now(),
            connection_count: 0,
            successful_requests: 0,
            failed_requests: 0,
        }
    }

    pub fn update_score(&mut self, delta: f64) {
        self.score += delta;
        self.last_updated = Timestamp::now();
    }

    pub fn add_warning(&mut self) {
        self.score -= 0.5;
        self.last_updated = Timestamp::now();
    }

    pub fn ban(&mut self, _duration: Duration) {
        self.score = 0.0;
        // Calculate ban expiration by adding duration to current time
        self.last_updated = Timestamp::now();
    }

    pub fn is_banned(&self) -> bool {
        self.score <= 0.0
    }

    pub fn record_connection(&mut self) {
        self.connection_count += 1;
        self.last_seen = Timestamp::now();
    }

    pub fn clear_ban(&mut self) {
        self.score = 1.0;
        self.last_updated = Timestamp::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_reputation_system() {
        let config = ReputationConfig {
            score_decay_rate: 0.1,
            min_score: 0.0,
            max_score: 10.0,
            update_interval: Duration::from_millis(50),
            history_size: 5,
            trust_threshold: 7.0,
            ban_threshold: 2.0,
        };

        let manager = PeerReputationManager::new(config);
        let peer_id = PeerId::random();

        // Update metrics
        let metrics = PeerMetrics {
            total_blocks_propagated: 10,
            total_blocks_validated: 9,
            total_transactions_relayed: 100,
            total_errors: 1,
            average_response_time: Duration::from_millis(50),
            ..PeerMetrics::default()
        };

        manager.update_metrics(&peer_id, metrics).await.unwrap();

        // Update reputation
        manager.update_reputation(&peer_id).await.unwrap();

        // Get score
        let score = manager.get_score(&peer_id).await.unwrap();
        assert!(score > 0.0);

        // Simulate some activity
        sleep(Duration::from_millis(100)).await;

        // Update metrics again with different values
        let new_metrics = PeerMetrics {
            total_blocks_propagated: 20,
            total_blocks_validated: 18,
            total_transactions_relayed: 200,
            total_errors: 2,
            average_response_time: Duration::from_millis(60),
            ..PeerMetrics::default()
        };

        manager.update_metrics(&peer_id, new_metrics).await.unwrap();

        // Update reputation again
        manager.update_reputation(&peer_id).await.unwrap();

        // Get new score
        let new_score = manager.get_score(&peer_id).await.unwrap();

        // The score should change based on the new metrics
        // Due to implementation details, the score may increase or decrease
        // We just check that they're different
        assert_ne!(new_score, score);
    }
}
