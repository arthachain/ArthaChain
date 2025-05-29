use anyhow::Result;
use log::info;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use crate::sharding::ShardId;

/// Reputation score (0.0 to 1.0)
pub type ReputationScore = f64;

/// Default minimum reputation threshold
pub const DEFAULT_MIN_REPUTATION: ReputationScore = 0.3;

/// Reputation update reason
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReputationUpdateReason {
    /// Good behavior (successful contribution)
    SuccessfulContribution,
    /// Bad behavior (failed validation)
    FailedValidation,
    /// Malicious behavior (invalid block)
    MaliciousBlock,
    /// Invalid transaction
    InvalidTransaction,
    /// Timeout (slow response)
    Timeout,
    /// Peer disconnected
    Disconnected,
    /// Peer reconnected
    Reconnected,
    /// Cross-shard validation success
    CrossShardValidationSuccess,
    /// Cross-shard validation failure
    CrossShardValidationFailure,
    /// Custom reason
    Custom(String),
}

/// Reputation update
#[derive(Debug, Clone)]
pub struct ReputationUpdate {
    /// Peer ID
    pub peer_id: String,
    /// Shard ID
    pub shard_id: ShardId,
    /// Score delta
    pub score_delta: f64,
    /// Update reason
    pub reason: ReputationUpdateReason,
    /// Timestamp
    pub timestamp: Instant,
}

/// Reputation manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationConfig {
    /// Minimum reputation score to participate
    pub min_reputation: ReputationScore,
    /// Initial reputation score for new peers
    pub initial_reputation: ReputationScore,
    /// Maximum reputation score adjustment
    pub max_adjustment: ReputationScore,
    /// Decay factor for reputation
    pub decay_factor: f64,
    /// Decay interval
    pub decay_interval_secs: u64,
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            min_reputation: DEFAULT_MIN_REPUTATION,
            initial_reputation: 0.5,
            max_adjustment: 0.1,
            decay_factor: 0.99,
            decay_interval_secs: 3600, // 1 hour
        }
    }
}

/// Reputation entry for a peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationEntry {
    /// Peer ID
    pub peer_id: String,
    /// Shard ID
    pub shard_id: ShardId,
    /// Reputation score
    pub score: ReputationScore,
    /// Last update timestamp as duration since UNIX_EPOCH
    pub last_update: u64,
    /// History of updates
    pub updates: Vec<(u64, ReputationUpdateReason, f64)>,
}

/// Reputation manager
pub struct ReputationManager {
    /// Reputation scores by peer ID and shard ID
    scores: Arc<RwLock<HashMap<String, HashMap<ShardId, ReputationEntry>>>>,
    /// Configuration
    config: ReputationConfig,
    /// Last decay time
    last_decay: Arc<RwLock<Instant>>,
    /// Pending updates
    pending_updates: Arc<RwLock<Vec<ReputationUpdate>>>,
    /// Running
    running: Arc<RwLock<bool>>,
}

impl ReputationManager {
    /// Create a new reputation manager
    pub fn new(config: ReputationConfig) -> Self {
        Self {
            scores: Arc::new(RwLock::new(HashMap::new())),
            config,
            last_decay: Arc::new(RwLock::new(Instant::now())),
            pending_updates: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the reputation manager
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = true;

        // Start background tasks for processing updates and decay

        Ok(())
    }

    /// Stop the reputation manager
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = false;

        Ok(())
    }

    /// Get the reputation score for a peer in a specific shard
    pub async fn get_score(&self, peer_id: &str, shard_id: ShardId) -> ReputationScore {
        let scores = self.scores.read().await;

        match scores.get(peer_id) {
            Some(shard_scores) => match shard_scores.get(&shard_id) {
                Some(entry) => entry.score,
                None => self.config.initial_reputation,
            },
            None => self.config.initial_reputation,
        }
    }

    /// Check if a peer has sufficient reputation to participate
    pub async fn is_allowed(&self, peer_id: &str, shard_id: ShardId) -> bool {
        let score = self.get_score(peer_id, shard_id).await;
        score >= self.config.min_reputation
    }

    /// Update the reputation score for a peer
    pub async fn update_score(
        &self,
        peer_id: &str,
        shard_id: ShardId,
        reason: ReputationUpdateReason,
        score_delta: f64,
    ) -> Result<()> {
        // Add to pending updates
        {
            let mut pending = self.pending_updates.write().await;
            pending.push(ReputationUpdate {
                peer_id: peer_id.to_string(),
                shard_id,
                score_delta,
                reason: reason.clone(),
                timestamp: Instant::now(),
            });
        }

        // Process the update immediately for now
        self.apply_update(peer_id, shard_id, reason, score_delta)
            .await
    }

    /// Apply a reputation update
    async fn apply_update(
        &self,
        peer_id: &str,
        shard_id: ShardId,
        reason: ReputationUpdateReason,
        score_delta: f64,
    ) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut scores = self.scores.write().await;

        // Get or create the shard map for this peer
        let shard_scores = scores
            .entry(peer_id.to_string())
            .or_insert_with(HashMap::new);

        // Get or create the reputation entry for this shard
        let entry = shard_scores
            .entry(shard_id)
            .or_insert_with(|| ReputationEntry {
                peer_id: peer_id.to_string(),
                shard_id,
                score: self.config.initial_reputation,
                last_update: now,
                updates: Vec::new(),
            });

        // Apply bounded score delta
        let bounded_delta = score_delta
            .max(-self.config.max_adjustment)
            .min(self.config.max_adjustment);

        entry.score = (entry.score + bounded_delta).clamp(0.0, 1.0);
        entry.last_update = now;

        // Add to history, keeping last 10 updates
        entry.updates.push((now, reason, bounded_delta));
        if entry.updates.len() > 10 {
            entry.updates.remove(0);
        }

        Ok(())
    }

    /// Process pending updates
    pub async fn process_pending_updates(&self) -> Result<()> {
        let mut pending = self.pending_updates.write().await;

        let updates_to_process = std::mem::take(&mut *pending);

        for update in updates_to_process {
            self.apply_update(
                &update.peer_id,
                update.shard_id,
                update.reason,
                update.score_delta,
            )
            .await?;
        }

        Ok(())
    }

    /// Apply decay to all reputation scores
    pub async fn apply_decay(&self) -> Result<()> {
        let mut last_decay = self.last_decay.write().await;
        let now = Instant::now();

        // Check if it's time to decay
        if now.duration_since(*last_decay).as_secs() < self.config.decay_interval_secs {
            return Ok(());
        }

        // Update last decay time
        *last_decay = now;

        // Apply decay to all scores
        let mut scores = self.scores.write().await;
        let mut total_decayed = 0;

        for peer_scores in scores.values_mut() {
            for entry in peer_scores.values_mut() {
                entry.score = entry.score * self.config.decay_factor;
                total_decayed += 1;
            }
        }

        info!("Applied reputation decay to {} peers", total_decayed);

        Ok(())
    }

    /// Get peers with scores above the threshold
    pub async fn get_trusted_peers(&self, shard_id: ShardId) -> Vec<String> {
        let scores = self.scores.read().await;
        let mut trusted_peers = Vec::new();

        for (peer_id, peer_scores) in scores.iter() {
            if let Some(entry) = peer_scores.get(&shard_id) {
                if entry.score >= self.config.min_reputation {
                    trusted_peers.push(peer_id.clone());
                }
            }
        }

        trusted_peers
    }

    /// Get a random trusted peer
    pub async fn get_random_trusted_peer(&self, shard_id: ShardId) -> Option<String> {
        let trusted_peers = self.get_trusted_peers(shard_id).await;

        if trusted_peers.is_empty() {
            return None;
        }

        let index = thread_rng().gen_range(0..trusted_peers.len());
        Some(trusted_peers[index].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reputation_manager() {
        let config = ReputationConfig::default();
        let manager = ReputationManager::new(config.clone());

        // Test initial score
        let peer_id = "peer1";
        let shard_id = 0;

        let score = manager.get_score(peer_id, shard_id).await;
        assert_eq!(score, config.initial_reputation);

        // Test updating score
        manager
            .update_score(
                peer_id,
                shard_id,
                ReputationUpdateReason::SuccessfulContribution,
                0.1,
            )
            .await
            .unwrap();

        let score = manager.get_score(peer_id, shard_id).await;
        assert!(score > config.initial_reputation);

        // Reset the score to initial
        let peer_id = "peer2";

        // Test negative update
        manager
            .update_score(
                peer_id,
                shard_id,
                ReputationUpdateReason::FailedValidation,
                -0.2,
            )
            .await
            .unwrap();

        let score = manager.get_score(peer_id, shard_id).await;
        assert!(score < config.initial_reputation);
    }
}
