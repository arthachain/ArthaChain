use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use chrono::{DateTime, Utc};

/// Reputation score with historical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    /// Current score
    pub score: f64,
    /// Historical performance records
    pub history: VecDeque<PerformanceRecord>,
    /// Total stake
    pub stake: u64,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
    /// Slashing status
    pub slashing_status: SlashingStatus,
}

/// Performance record for a validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    /// Timestamp of the record
    pub timestamp: DateTime<Utc>,
    /// Performance score (-1.0 to 1.0)
    pub score: f64,
    /// Type of performance event
    pub event_type: PerformanceEvent,
}

/// Type of performance event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceEvent {
    /// Block proposal
    BlockProposal,
    /// Block validation
    BlockValidation,
    /// Cross-shard consensus
    CrossShardConsensus,
    /// Slashing event
    Slashing,
}

/// Slashing status for a validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlashingStatus {
    /// No slashing
    None,
    /// Temporarily slashed
    Temporary {
        /// Slash amount
        amount: u64,
        /// End timestamp
        end_time: DateTime<Utc>,
    },
    /// Permanently slashed
    Permanent {
        /// Slash amount
        amount: u64,
    },
}

/// Reputation manager for tracking node performance
pub struct ReputationManager {
    scores: Arc<RwLock<HashMap<u64, ReputationScore>>>,
    _max_score: f64,
    history_size: usize,
    slashing_threshold: f64,
    min_stake: u64,
}

impl ReputationManager {
    /// Create a new reputation manager
    pub fn new(max_score: f64, history_size: usize, slashing_threshold: f64, min_stake: u64) -> Self {
        Self {
            scores: Arc::new(RwLock::new(HashMap::new())),
            _max_score: max_score,
            history_size,
            slashing_threshold,
            min_stake,
        }
    }

    /// Increase reputation for a shard
    pub async fn increase_reputation(&self, shard_id: u64, event_type: PerformanceEvent) -> Result<()> {
        let mut scores = self.scores.write().await;
        let score = scores.entry(shard_id).or_insert(ReputationScore {
            score: 0.0,
            history: VecDeque::with_capacity(self.history_size),
            stake: 0,
            last_update: Utc::now(),
            slashing_status: SlashingStatus::None,
        });

        // Add performance record
        score.history.push_back(PerformanceRecord {
            timestamp: Utc::now(),
            score: 0.1,
            event_type: event_type.clone(),
        });

        // Trim history if needed
        if score.history.len() > self.history_size {
            score.history.pop_front();
        }

        // Update score with weighted average
        score.score = self.calculate_weighted_score(&score.history);
        score.last_update = Utc::now();

        Ok(())
    }

    /// Decrease reputation for a shard
    pub async fn decrease_reputation(&self, shard_id: u64, event_type: PerformanceEvent) -> Result<()> {
        let mut scores = self.scores.write().await;
        let score = scores.entry(shard_id).or_insert(ReputationScore {
            score: 0.0,
            history: VecDeque::with_capacity(self.history_size),
            stake: 0,
            last_update: Utc::now(),
            slashing_status: SlashingStatus::None,
        });

        // Add performance record
        score.history.push_back(PerformanceRecord {
            timestamp: Utc::now(),
            score: -0.1,
            event_type: event_type.clone(),
        });

        // Trim history if needed
        if score.history.len() > self.history_size {
            score.history.pop_front();
        }

        // Update score with weighted average
        score.score = self.calculate_weighted_score(&score.history);
        score.last_update = Utc::now();

        // Check for slashing conditions
        if score.score < -self.slashing_threshold {
            self.apply_slashing(shard_id, score).await?;
        }

        Ok(())
    }

    /// Calculate weighted score from history
    fn calculate_weighted_score(&self, history: &VecDeque<PerformanceRecord>) -> f64 {
        if history.is_empty() {
            return 0.0;
        }

        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for (i, record) in history.iter().enumerate() {
            let weight = 1.0 / (i + 1) as f64; // More recent records have higher weight
            total_weight += weight;
            weighted_sum += record.score * weight;
        }

        weighted_sum / total_weight
    }

    /// Apply slashing conditions
    async fn apply_slashing(&self, _shard_id: u64, score: &mut ReputationScore) -> Result<()> {
        match score.slashing_status {
            SlashingStatus::None => {
                // First offense: temporary slashing
                let slash_amount = (score.stake as f64 * 0.1) as u64; // 10% of stake
                score.slashing_status = SlashingStatus::Temporary {
                    amount: slash_amount,
                    end_time: Utc::now() + chrono::Duration::hours(24),
                };
                score.stake -= slash_amount;
            }
            SlashingStatus::Temporary { .. } => {
                // Second offense: permanent slashing
                let slash_amount = (score.stake as f64 * 0.5) as u64; // 50% of stake
                score.slashing_status = SlashingStatus::Permanent {
                    amount: slash_amount,
                };
                score.stake -= slash_amount;
            }
            SlashingStatus::Permanent { .. } => {
                // Already permanently slashed
            }
        }
        Ok(())
    }

    /// Get the current reputation score for a shard
    pub async fn get_reputation(&self, shard_id: u64) -> f64 {
        let scores = self.scores.read().await;
        scores.get(&shard_id).map(|s| s.score).unwrap_or(0.0)
    }

    /// Get the current stake for a shard
    pub async fn get_stake(&self, shard_id: u64) -> u64 {
        let scores = self.scores.read().await;
        scores.get(&shard_id).map(|s| s.stake).unwrap_or(0)
    }

    /// Update stake for a shard
    pub async fn update_stake(&self, shard_id: u64, new_stake: u64) -> Result<()> {
        let mut scores = self.scores.write().await;
        if let Some(score) = scores.get_mut(&shard_id) {
            score.stake = new_stake;
        }
        Ok(())
    }

    /// Check if a shard is eligible for validation
    pub async fn is_eligible(&self, _shard_id: u64) -> bool {
        let scores = self.scores.read().await;
        if let Some(score) = scores.get(&_shard_id) {
            score.score >= 0.0 && score.stake >= self.min_stake
        } else {
            false
        }
    }
} 