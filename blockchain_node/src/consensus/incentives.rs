use crate::ledger::block::Block;
use crate::ledger::transaction::Transaction;
use crate::network::types::NodeId;
use anyhow::Result;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for the incentive system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncentiveConfig {
    /// Base block reward amount
    pub base_block_reward: u64,
    /// Percentage of transaction fees to allocate to validators
    pub fee_percentage: u8,
    /// Reward adjustment interval in blocks
    pub reward_adjustment_interval: u64,
    /// Use dynamic reward scaling
    pub use_dynamic_rewards: bool,
    /// Minimum stake to receive rewards
    pub min_stake_for_rewards: u64,
    /// Reward distribution method
    pub distribution_method: RewardDistributionMethod,
    /// Reputation bonus percentage
    pub reputation_bonus_percentage: u8,
    /// Enable penalties
    pub enable_penalties: bool,
    /// Penalty for missed blocks
    pub missed_block_penalty: u64,
    /// Penalty for invalid blocks
    pub invalid_block_penalty: u64,
    /// Rewards for special contributions
    pub special_contribution_rewards: HashMap<String, u64>,
}

impl Default for IncentiveConfig {
    fn default() -> Self {
        Self {
            base_block_reward: 1000,
            fee_percentage: 80,
            reward_adjustment_interval: 10000,
            use_dynamic_rewards: true,
            min_stake_for_rewards: 1000,
            distribution_method: RewardDistributionMethod::ProportionalStake,
            reputation_bonus_percentage: 10,
            enable_penalties: true,
            missed_block_penalty: 100,
            invalid_block_penalty: 500,
            special_contribution_rewards: {
                let mut rewards = HashMap::new();
                rewards.insert("checkpoint_creation".to_string(), 50);
                rewards.insert("security_report".to_string(), 200);
                rewards.insert("network_support".to_string(), 100);
                rewards
            },
        }
    }
}

/// Methods for distributing rewards
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewardDistributionMethod {
    /// Equal distribution to all validators
    Equal,
    /// Proportional to stake
    ProportionalStake,
    /// Proportional to work done
    ProportionalWork,
    /// Rank-based distribution
    RankBased,
    /// Reputation-weighted distribution
    ReputationWeighted,
    /// Social-graph weighted (for SVBFT)
    SocialGraphWeighted,
}

/// Reward event for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardEvent {
    /// Validator that received the reward
    pub validator_id: NodeId,
    /// Amount of the reward
    pub amount: u64,
    /// Block height
    pub block_height: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Reason for the reward
    pub reason: String,
    /// Related block hash if applicable
    pub block_hash: Option<Vec<u8>>,
    /// Transaction hash if applicable
    pub tx_hash: Option<Vec<u8>>,
}

/// Penalty event for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyEvent {
    /// Validator that received the penalty
    pub validator_id: NodeId,
    /// Amount of the penalty
    pub amount: u64,
    /// Block height
    pub block_height: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Reason for the penalty
    pub reason: String,
    /// Related incident ID if applicable
    pub incident_id: Option<String>,
}

/// Reward calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardCalculation {
    /// Total block reward
    pub total_block_reward: u64,
    /// Total transaction fees
    pub total_tx_fees: u64,
    /// Rewards by validator
    pub rewards_by_validator: HashMap<NodeId, u64>,
    /// Total distributed rewards
    pub total_distributed: u64,
    /// Block height
    pub block_height: u64,
    /// Distribution method used
    pub distribution_method: RewardDistributionMethod,
}

/// Manager for handling blockchain incentives
pub struct IncentiveManager {
    /// Configuration
    config: RwLock<IncentiveConfig>,
    /// Validator stakes
    validator_stakes: RwLock<HashMap<NodeId, u64>>,
    /// Validator reputations
    validator_reputations: RwLock<HashMap<NodeId, f64>>,
    /// Active validators
    validators: Arc<RwLock<HashSet<NodeId>>>,
    /// Last reward calculation
    last_reward_calculation: RwLock<Option<RewardCalculation>>,
    /// Reward history
    reward_history: RwLock<Vec<RewardEvent>>,
    /// Penalty history
    penalty_history: RwLock<Vec<PenaltyEvent>>,
    /// Last reward time
    last_reward_time: RwLock<Instant>,
    /// Running flag
    running: RwLock<bool>,
    /// Total rewards distributed
    total_rewards_distributed: RwLock<u64>,
    /// Current block height
    current_block_height: RwLock<u64>,
}

impl IncentiveManager {
    /// Create a new incentive manager
    pub fn new(config: IncentiveConfig, validators: Arc<RwLock<HashSet<NodeId>>>) -> Self {
        Self {
            config: RwLock::new(config),
            validator_stakes: RwLock::new(HashMap::new()),
            validator_reputations: RwLock::new(HashMap::new()),
            validators,
            last_reward_calculation: RwLock::new(None),
            reward_history: RwLock::new(Vec::new()),
            penalty_history: RwLock::new(Vec::new()),
            last_reward_time: RwLock::new(Instant::now()),
            running: RwLock::new(false),
            total_rewards_distributed: RwLock::new(0),
            current_block_height: RwLock::new(0),
        }
    }

    /// Start the incentive manager
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(anyhow::anyhow!("Incentive manager already running"));
        }

        *running = true;
        info!("Incentive manager started");
        Ok(())
    }

    /// Stop the incentive manager
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(anyhow::anyhow!("Incentive manager not running"));
        }

        *running = false;
        info!("Incentive manager stopped");
        Ok(())
    }

    /// Update validator stake
    pub async fn update_stake(&self, validator_id: NodeId, stake: u64) -> Result<()> {
        let mut stakes = self.validator_stakes.write().await;
        stakes.insert(validator_id, stake);
        Ok(())
    }

    /// Update validator reputation
    pub async fn update_reputation(&self, validator_id: NodeId, reputation: f64) -> Result<()> {
        let mut reputations = self.validator_reputations.write().await;
        reputations.insert(validator_id, reputation);
        Ok(())
    }

    /// Calculate rewards for a block
    pub async fn calculate_block_rewards(&self, block: &Block) -> Result<RewardCalculation> {
        let config = self.config.read().await;
        let validator_stakes = self.validator_stakes.read().await;
        let validator_reputations = self.validator_reputations.read().await;
        let validators = self.validators.read().await;

        // Calculate total transaction fees
        let total_tx_fees = block.txs.iter().map(|tx| tx.fee.unwrap_or(0)).sum::<u64>();

        // Calculate total block reward
        let total_block_reward = if config.use_dynamic_rewards {
            // Adjust based on block height
            let base_reward = config.base_block_reward;
            let adjustment_factor =
                1.0 - (block.height as f64 / (block.height + 2_000_000) as f64) * 0.5;
            (base_reward as f64 * adjustment_factor) as u64
        } else {
            config.base_block_reward
        };

        // Calculate validator portion of fees
        let validator_fee_reward = total_tx_fees * config.fee_percentage as u64 / 100;

        // Calculate total reward pool
        let reward_pool = total_block_reward + validator_fee_reward;

        // Distribute rewards based on configured method
        let mut rewards_by_validator = HashMap::new();
        let mut total_distributed = 0;

        match config.distribution_method {
            RewardDistributionMethod::Equal => {
                // Everyone gets an equal share
                let validator_count = validators.len() as u64;
                if validator_count > 0 {
                    let reward_per_validator = reward_pool / validator_count;

                    for validator in validators.iter() {
                        if validator_stakes.get(validator).copied().unwrap_or(0)
                            >= config.min_stake_for_rewards
                        {
                            rewards_by_validator.insert(validator.clone(), reward_per_validator);
                            total_distributed += reward_per_validator;
                        }
                    }
                }
            }
            RewardDistributionMethod::ProportionalStake => {
                // Rewards proportional to stake
                let total_stake: u64 = validator_stakes
                    .iter()
                    .filter(|(id, stake)| {
                        validators.contains(*id) && **stake >= config.min_stake_for_rewards
                    })
                    .map(|(_, stake)| *stake)
                    .sum();

                if total_stake > 0 {
                    for (validator, stake) in validator_stakes.iter() {
                        if validators.contains(validator) && *stake >= config.min_stake_for_rewards
                        {
                            let reward =
                                (reward_pool as u128 * *stake as u128 / total_stake as u128) as u64;
                            rewards_by_validator.insert(validator.clone(), reward);
                            total_distributed += reward;
                        }
                    }
                }
            }
            RewardDistributionMethod::ReputationWeighted => {
                // Rewards weighted by reputation
                let mut total_reputation = 0.0;

                for validator in validators.iter() {
                    if validator_stakes.get(validator).copied().unwrap_or(0)
                        >= config.min_stake_for_rewards
                    {
                        total_reputation +=
                            validator_reputations.get(validator).copied().unwrap_or(0.5);
                    }
                }

                if total_reputation > 0.0 {
                    for validator in validators.iter() {
                        if validator_stakes.get(validator).copied().unwrap_or(0)
                            >= config.min_stake_for_rewards
                        {
                            let reputation =
                                validator_reputations.get(validator).copied().unwrap_or(0.5);
                            let reward =
                                (reward_pool as f64 * reputation / total_reputation) as u64;
                            rewards_by_validator.insert(validator.clone(), reward);
                            total_distributed += reward;
                        }
                    }
                }
            }
            RewardDistributionMethod::SocialGraphWeighted => {
                // For SVBFT: implement social graph weighting in the future
                // For now, fallback to reputation-weighted
                let mut total_reputation = 0.0;

                for validator in validators.iter() {
                    if validator_stakes.get(validator).copied().unwrap_or(0)
                        >= config.min_stake_for_rewards
                    {
                        total_reputation +=
                            validator_reputations.get(validator).copied().unwrap_or(0.5);
                    }
                }

                if total_reputation > 0.0 {
                    for validator in validators.iter() {
                        if validator_stakes.get(validator).copied().unwrap_or(0)
                            >= config.min_stake_for_rewards
                        {
                            let reputation =
                                validator_reputations.get(validator).copied().unwrap_or(0.5);
                            let reward =
                                (reward_pool as f64 * reputation / total_reputation) as u64;
                            rewards_by_validator.insert(validator.clone(), reward);
                            total_distributed += reward;
                        }
                    }
                }
            }
            _ => {
                // Other methods would be implemented here
                // For now, default to equal distribution
                let validator_count = validators.len() as u64;
                if validator_count > 0 {
                    let reward_per_validator = reward_pool / validator_count;

                    for validator in validators.iter() {
                        if validator_stakes.get(validator).copied().unwrap_or(0)
                            >= config.min_stake_for_rewards
                        {
                            rewards_by_validator.insert(validator.clone(), reward_per_validator);
                            total_distributed += reward_per_validator;
                        }
                    }
                }
            }
        }

        // Record this calculation
        let calculation = RewardCalculation {
            total_block_reward,
            total_tx_fees,
            rewards_by_validator: rewards_by_validator.clone(),
            total_distributed,
            block_height: block.height,
            distribution_method: config.distribution_method,
        };

        *self.last_reward_calculation.write().await = Some(calculation.clone());
        *self.last_reward_time.write().await = Instant::now();
        *self.current_block_height.write().await = block.height;

        // Update total rewards distributed
        let mut total = self.total_rewards_distributed.write().await;
        *total += total_distributed;

        // Record reward events
        let mut reward_history = self.reward_history.write().await;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for (validator, amount) in rewards_by_validator.iter() {
            reward_history.push(RewardEvent {
                validator_id: validator.clone(),
                amount: *amount,
                block_height: block.height,
                timestamp,
                reason: "Block reward".to_string(),
                block_hash: Some(block.hash.clone()),
                tx_hash: None,
            });
        }

        Ok(calculation)
    }

    /// Apply a penalty to a validator
    pub async fn apply_penalty(
        &self,
        validator_id: NodeId,
        amount: u64,
        reason: String,
        block_height: u64,
        incident_id: Option<String>,
    ) -> Result<()> {
        let config = self.config.read().await;
        if !config.enable_penalties {
            return Ok(());
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let penalty = PenaltyEvent {
            validator_id: validator_id.clone(),
            amount,
            block_height,
            timestamp,
            reason,
            incident_id,
        };

        // Record the penalty
        let mut penalty_history = self.penalty_history.write().await;
        penalty_history.push(penalty);

        // In a real system, we'd actually deduct from the validator's balance here

        Ok(())
    }

    /// Apply a special reward
    pub async fn apply_special_reward(
        &self,
        validator_id: NodeId,
        reward_type: String,
        block_height: u64,
        tx_hash: Option<Vec<u8>>,
    ) -> Result<u64> {
        let config = self.config.read().await;
        let amount = config
            .special_contribution_rewards
            .get(&reward_type)
            .copied()
            .unwrap_or(0);

        if amount == 0 {
            return Ok(0);
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let reward = RewardEvent {
            validator_id: validator_id.clone(),
            amount,
            block_height,
            timestamp,
            reason: format!("Special reward: {}", reward_type),
            block_hash: None,
            tx_hash,
        };

        // Record the reward
        let mut reward_history = self.reward_history.write().await;
        reward_history.push(reward);

        // Update total rewards distributed
        let mut total = self.total_rewards_distributed.write().await;
        *total += amount;

        // In a real system, we'd actually credit the validator's balance here

        Ok(amount)
    }

    /// Get the last reward calculation
    pub async fn get_last_reward_calculation(&self) -> Option<RewardCalculation> {
        self.last_reward_calculation.read().await.clone()
    }

    /// Get total rewards for a validator
    pub async fn get_validator_total_rewards(&self, validator_id: &NodeId) -> u64 {
        let reward_history = self.reward_history.read().await;

        reward_history
            .iter()
            .filter(|event| &event.validator_id == validator_id)
            .map(|event| event.amount)
            .sum()
    }

    /// Get total penalties for a validator
    pub async fn get_validator_total_penalties(&self, validator_id: &NodeId) -> u64 {
        let penalty_history = self.penalty_history.read().await;

        penalty_history
            .iter()
            .filter(|event| &event.validator_id == validator_id)
            .map(|event| event.amount)
            .sum()
    }

    /// Get reward history for a validator
    pub async fn get_validator_reward_history(&self, validator_id: &NodeId) -> Vec<RewardEvent> {
        let reward_history = self.reward_history.read().await;

        reward_history
            .iter()
            .filter(|event| &event.validator_id == validator_id)
            .cloned()
            .collect()
    }

    /// Get penalty history for a validator
    pub async fn get_validator_penalty_history(&self, validator_id: &NodeId) -> Vec<PenaltyEvent> {
        let penalty_history = self.penalty_history.read().await;

        penalty_history
            .iter()
            .filter(|event| &event.validator_id == validator_id)
            .cloned()
            .collect()
    }

    /// Get all reward events
    pub async fn get_all_reward_events(&self) -> Vec<RewardEvent> {
        self.reward_history.read().await.clone()
    }

    /// Get all penalty events
    pub async fn get_all_penalty_events(&self) -> Vec<PenaltyEvent> {
        self.penalty_history.read().await.clone()
    }

    /// Get the total rewards distributed
    pub async fn get_total_rewards_distributed(&self) -> u64 {
        *self.total_rewards_distributed.read().await
    }

    /// Update the configuration
    pub async fn update_config(&self, config: IncentiveConfig) -> Result<()> {
        let mut cfg = self.config.write().await;
        *cfg = config;
        Ok(())
    }

    /// Get incentive statistics
    pub async fn get_statistics(&self) -> IncentiveStatistics {
        let reward_history = self.reward_history.read().await;
        let penalty_history = self.penalty_history.read().await;
        let validator_stakes = self.validator_stakes.read().await;
        let validator_reputations = self.validator_reputations.read().await;
        let validators = self.validators.read().await;
        let total_distributed = *self.total_rewards_distributed.read().await;

        // Calculate rewards by validator
        let mut rewards_by_validator = HashMap::new();
        for event in reward_history.iter() {
            *rewards_by_validator
                .entry(event.validator_id.clone())
                .or_insert(0) += event.amount;
        }

        // Calculate penalties by validator
        let mut penalties_by_validator = HashMap::new();
        for event in penalty_history.iter() {
            *penalties_by_validator
                .entry(event.validator_id.clone())
                .or_insert(0) += event.amount;
        }

        // Find validator with highest rewards
        let validator_with_highest_rewards = rewards_by_validator
            .iter()
            .max_by_key(|(_, amount)| *amount)
            .map(|(validator, amount)| (validator.clone(), *amount));

        // Find validator with highest penalties
        let validator_with_highest_penalties = penalties_by_validator
            .iter()
            .max_by_key(|(_, amount)| *amount)
            .map(|(validator, amount)| (validator.clone(), *amount));

        // Calculate average rewards
        let average_rewards = if !rewards_by_validator.is_empty() {
            total_distributed / rewards_by_validator.len() as u64
        } else {
            0
        };

        IncentiveStatistics {
            total_rewards_distributed,
            total_penalties_applied: penalty_history.iter().map(|e| e.amount).sum(),
            reward_events_count: reward_history.len(),
            penalty_events_count: penalty_history.len(),
            rewards_by_validator,
            penalties_by_validator,
            validator_with_highest_rewards,
            validator_with_highest_penalties,
            average_rewards,
            active_validators_count: validators.len(),
            validators_with_rewards_count: rewards_by_validator.len(),
            validators_with_penalties_count: penalties_by_validator.len(),
        }
    }
}

impl Clone for IncentiveManager {
    fn clone(&self) -> Self {
        // This is a partial clone for use in async tasks
        Self {
            config: RwLock::new(self.config.try_read().unwrap_or_default().clone()),
            validator_stakes: RwLock::new(HashMap::new()),
            validator_reputations: RwLock::new(HashMap::new()),
            validators: self.validators.clone(),
            last_reward_calculation: RwLock::new(None),
            reward_history: RwLock::new(Vec::new()),
            penalty_history: RwLock::new(Vec::new()),
            last_reward_time: RwLock::new(Instant::now()),
            running: RwLock::new(false),
            total_rewards_distributed: RwLock::new(0),
            current_block_height: RwLock::new(0),
        }
    }
}

/// Statistics about the incentive system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncentiveStatistics {
    /// Total rewards distributed
    pub total_rewards_distributed: u64,
    /// Total penalties applied
    pub total_penalties_applied: u64,
    /// Number of reward events
    pub reward_events_count: usize,
    /// Number of penalty events
    pub penalty_events_count: usize,
    /// Rewards by validator
    pub rewards_by_validator: HashMap<NodeId, u64>,
    /// Penalties by validator
    pub penalties_by_validator: HashMap<NodeId, u64>,
    /// Validator with highest rewards
    pub validator_with_highest_rewards: Option<(NodeId, u64)>,
    /// Validator with highest penalties
    pub validator_with_highest_penalties: Option<(NodeId, u64)>,
    /// Average rewards per validator
    pub average_rewards: u64,
    /// Number of active validators
    pub active_validators_count: usize,
    /// Number of validators that have received rewards
    pub validators_with_rewards_count: usize,
    /// Number of validators that have received penalties
    pub validators_with_penalties_count: usize,
}
