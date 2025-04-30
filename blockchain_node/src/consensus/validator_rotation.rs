// Standard library imports
use std::sync::Arc;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{SystemTime, Duration};

// External crate imports
use log::{info, debug};
use rand::{thread_rng, Rng};
use rand::rngs::ThreadRng;
use serde::{Serialize, Deserialize};

// Internal crate imports
use crate::types::Address;
use crate::consensus::reputation::ReputationManager;

/// Validator set configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSetConfig {
    /// Minimum number of validators
    pub min_validators: usize,
    /// Maximum number of validators
    pub max_validators: usize,
    /// Rotation period in blocks
    pub rotation_period: u64,
    /// Sliding window size
    pub window_size: usize,
    /// Minimum stake required
    pub min_stake: u64,
    /// Minimum reputation score required
    pub min_reputation: f64,
    /// Handoff period in blocks
    pub handoff_period: u64,
}

/// Validator information
#[derive(Debug, Clone)]
pub struct ValidatorInfo {
    /// Validator address
    pub address: Address,
    /// Stake amount
    pub stake: u64,
    /// Performance metrics
    pub performance: ValidatorPerformance,
    /// Last active block
    pub last_active: Option<SystemTime>,
    /// Is active
    pub is_active: bool,
}

/// Validator performance metrics
#[derive(Clone, Debug)]
pub struct ValidatorPerformance {
    pub proposal_success_rate: f64,
    pub vote_success_rate: f64,
    pub avg_block_time: Duration,
    pub network_latency: Duration,
    pub resource_utilization: f64,
    pub uptime: Duration,
    pub last_active: Option<SystemTime>,
    pub total_blocks_proposed: u64,
    pub total_votes: u64,
    pub missed_proposals: u64,
    pub missed_votes: u64,
}

impl ValidatorPerformance {
    pub fn new() -> Self {
        Self {
            proposal_success_rate: 0.0,
            vote_success_rate: 0.0,
            avg_block_time: Duration::from_secs(0),
            network_latency: Duration::from_secs(0),
            resource_utilization: 0.0,
            uptime: Duration::from_secs(0),
            last_active: None,
            total_blocks_proposed: 0,
            total_votes: 0,
            missed_proposals: 0,
            missed_votes: 0,
        }
    }

    pub fn update_metrics(&mut self, 
        proposal_success: bool,
        vote_success: bool, 
        block_time: Duration,
        latency: Duration,
        utilization: f64
    ) {
        if proposal_success {
            self.total_blocks_proposed += 1;
        } else {
            self.missed_proposals += 1;
        }
        
        if vote_success {
            self.total_votes += 1;
        } else {
            self.missed_votes += 1;
        }

        self.proposal_success_rate = self.total_blocks_proposed as f64 / 
            (self.total_blocks_proposed + self.missed_proposals) as f64;
        
        self.vote_success_rate = self.total_votes as f64 /
            (self.total_votes + self.missed_votes) as f64;

        self.avg_block_time = block_time;
        self.network_latency = latency;
        self.resource_utilization = utilization;
        self.last_active = Some(SystemTime::now());
    }

    pub fn calculate_score(&self) -> f64 {
        let proposal_weight = 0.4;
        let vote_weight = 0.3;
        let latency_weight = 0.2;
        let utilization_weight = 0.1;

        let proposal_score = self.proposal_success_rate;
        let vote_score = self.vote_success_rate;
        
        // Convert latency to a score (lower is better)
        let latency_score = 1.0 - (self.network_latency.as_secs_f64() / 1.0).min(1.0);
        
        // Resource utilization score (closer to optimal is better)
        let utilization_score = 1.0 - (self.resource_utilization - 0.7).abs();

        proposal_weight * proposal_score +
        vote_weight * vote_score +
        latency_weight * latency_score +
        utilization_weight * utilization_score
    }
}

/// Validator set rotation manager
pub struct ValidatorRotationManager {
    /// Current validator set
    current_validators: HashMap<Address, ValidatorInfo>,
    /// Next validator set
    next_validators: HashMap<Address, ValidatorInfo>,
    /// Sliding window of recent validators
    validator_window: VecDeque<Address>,
    /// Configuration
    config: ValidatorSetConfig,
    /// Reputation manager
    _reputation_manager: Arc<ReputationManager>,
    /// Random number generator
    _rng: ThreadRng,
    /// Current block height
    current_height: u64,
    /// Handoff in progress
    handoff_in_progress: bool,
}

/// Rotation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationEvent {
    /// Block height
    pub height: u64,
    /// Validator address
    pub validator: Address,
    /// Previous shard ID
    pub previous_shard: u64,
    /// New shard ID
    pub new_shard: u64,
    /// Reason for rotation
    pub reason: RotationReason,
}

/// Reason for validator rotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationReason {
    /// Scheduled rotation
    Scheduled,
    /// Performance issues
    Performance,
    /// Stake changes
    Stake,
    /// Reputation changes
    Reputation,
    /// Network conditions
    Network,
    /// Manual rotation
    Manual,
}

#[derive(Debug, thiserror::Error)]
pub enum ValidatorRotationError {
    #[error("Insufficient validators: required {required}, found {found}")]
    InsufficientValidators {
        required: usize,
        found: usize,
    },
    
    #[error("Invalid stake amount: {0}")]
    InvalidStake(u64),
    
    #[error("Invalid reputation score: {0}")]
    InvalidReputation(f64),
    
    #[error("Validator not found: {0}")]
    ValidatorNotFound(Address),
    
    #[error("Handoff already in progress")]
    HandoffInProgress,
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl ValidatorRotationManager {
    /// Create a new validator rotation manager
    pub fn new(
        config: ValidatorSetConfig,
        reputation_manager: Arc<ReputationManager>,
    ) -> Self {
        Self {
            current_validators: HashMap::new(),
            next_validators: HashMap::new(),
            validator_window: VecDeque::with_capacity(config.window_size),
            config,
            _reputation_manager: reputation_manager,
            _rng: thread_rng(),
            current_height: 0,
            handoff_in_progress: false,
        }
    }

    /// Update validator set based on current block height
    pub async fn update_validator_set(&mut self, height: u64) -> Result<(), ValidatorRotationError> {
        self.current_height = height;
        
        // Check if it's time for rotation
        if height % self.config.rotation_period == 0 {
            self.rotate_validators().await?;
        }
        
        // Check if handoff is needed
        if self.handoff_in_progress && height % self.config.handoff_period == 0 {
            self.complete_handoff().await?;
        }
        
        Ok(())
    }

    /// Rotate validators
    async fn rotate_validators(&mut self) -> Result<(), ValidatorRotationError> {
        debug!("Rotating validator set at height {}", self.current_height);
        
        // Select new validators
        self.select_new_validators().await?;
        
        // Start handoff process
        self.handoff_in_progress = true;
        
        info!("Validator rotation initiated at height {}", self.current_height);
        
        Ok(())
    }

    /// Select new validators using weighted random selection based on stake and reputation
    async fn select_new_validators(&mut self) -> Result<(), ValidatorRotationError> {
        let mut rng = ThreadRng::default();
        let mut selected = HashSet::new();
        let mut total_weight = 0.0;
        
        // Calculate weights for each validator
        let mut weights: Vec<(Address, f64)> = self.validator_window
            .iter()
            .filter_map(|addr| {
                if self.is_validator_eligible(addr) {
                    let weight = self.calculate_validator_weight(addr);
                    total_weight += weight;
                    Some((addr.clone(), weight))
                } else {
                    None
                }
            })
            .collect();
            
        if weights.len() < self.config.min_validators {
            return Err(ValidatorRotationError::InsufficientValidators {
                required: self.config.min_validators,
                found: weights.len(),
            });
        }
        
        // Sort by weight descending for deterministic selection when weights are equal
        weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Always select top performers up to min_validators
        for (addr, _) in weights.iter().take(self.config.min_validators) {
            selected.insert(addr.clone());
        }
        
        // Randomly select remaining validators weighted by stake and reputation
        while selected.len() < self.config.max_validators && !weights.is_empty() {
            let threshold = rng.gen::<f64>() * total_weight;
            let mut cumulative = 0.0;
            
            for (i, (addr, weight)) in weights.iter().enumerate() {
                cumulative += weight;
                if cumulative >= threshold && !selected.contains(addr) {
                    selected.insert(addr.clone());
                    total_weight -= weight;
                    weights.remove(i);
                    break;
                }
            }
        }
        
        // Update next validator set with validator info
        self.next_validators = selected
            .into_iter()
            .filter_map(|addr| {
                self.current_validators.get(&addr).map(|info| {
                    (addr.clone(), info.clone())
                })
            })
            .collect();
            
        info!("Selected {} new validators", self.next_validators.len());
        Ok(())
    }
    
    /// Calculate weight for validator selection based on stake and performance
    fn calculate_validator_weight(&self, addr: &Address) -> f64 {
        let info = self.current_validators.get(addr).unwrap();
        let stake_weight = (info.stake as f64) / (self.config.min_stake as f64);
        let perf_weight = info.performance.calculate_score();
        
        // Combine stake and performance with configurable weights
        0.7 * stake_weight + 0.3 * perf_weight
    }
    
    /// Check if validator meets minimum requirements
    fn is_validator_eligible(&self, addr: &Address) -> bool {
        let info = self.current_validators.get(addr).unwrap();
        // Check minimum stake requirement
        if info.stake < self.config.min_stake {
            return false;
        }
        
        // Check if validator has been active recently
        if let Some(last_active) = info.last_active {
            if last_active.elapsed().unwrap_or_default() > Duration::from_secs(3600) {
                return false;
            }
        }
        
        // Check minimum performance requirements
        let score = info.performance.calculate_score();
        score >= 0.5
    }

    /// Complete the handoff to the next validator set
    async fn complete_handoff(&mut self) -> Result<(), ValidatorRotationError> {
        debug!("Completing validator handoff at height {}", self.current_height);
        
        // Verify next validator set is ready
        let found = self.next_validators.len();
        if found < self.config.min_validators {
            return Err(ValidatorRotationError::InsufficientValidators {
                required: self.config.min_validators,
                found,
            });
        }
        
        // Update current validators
        self.current_validators = self.next_validators.clone();
        self.next_validators.clear();
        self.handoff_in_progress = false;
        
        info!("Validator handoff completed at height {}", self.current_height);
        
        Ok(())
    }

    /// Update validator performance metrics
    pub async fn update_validator_performance(
        &mut self,
        address: &Address,
        performance: ValidatorPerformance,
    ) -> Result<(), ValidatorRotationError> {
        if let Some(info) = self.current_validators.get_mut(address) {
            info.performance = performance.clone();
            info.last_active = Some(SystemTime::now());
        } else if let Some(info) = self.next_validators.get_mut(address) {
            info.performance = performance;
            info.last_active = Some(SystemTime::now());
        } else {
            return Err(ValidatorRotationError::ValidatorNotFound(address.clone()));
        }
        
        Ok(())
    }

    /// Get the current validator set
    pub fn get_current_validators(&self) -> &HashMap<Address, ValidatorInfo> {
        &self.current_validators
    }

    /// Get the next validator set
    pub fn get_next_validators(&self) -> &HashMap<Address, ValidatorInfo> {
        &self.next_validators
    }

    /// Check if a validator is in the current set
    pub fn is_current_validator(&self, address: &Address) -> bool {
        self.current_validators.contains_key(address)
    }

    /// Check if a validator is in the next set
    pub fn is_next_validator(&self, address: &Address) -> bool {
        self.next_validators.contains_key(address)
    }

    pub fn update_validators(&mut self, address: &Address, performance: ValidatorPerformance) -> Result<(), ValidatorRotationError> {
        if let Some(info) = self.current_validators.get_mut(address) {
            info.performance = performance.clone();
        }
        
        if let Some(info) = self.next_validators.get_mut(address) {
            info.performance = performance;
        }
        
        Ok(())
    }
}

pub struct ValidatorRotation {
    validators: HashMap<Address, ValidatorPerformance>,
    rng: ThreadRng,
}

impl ValidatorRotation {
    pub fn new() -> Self {
        Self {
            validators: HashMap::new(),
            rng: thread_rng(),
        }
    }

    pub fn update_validator_info(&mut self, address: &Address, performance: ValidatorPerformance) {
        self.validators.insert(address.clone(), performance);
    }

    pub fn get_performance(&self, address: &Address) -> Option<&ValidatorPerformance> {
        self.validators.get(address)
    }

    pub fn select_random_validator(&mut self) -> Option<Address> {
        if self.validators.is_empty() {
            return None;
        }
        let validators: Vec<_> = self.validators.keys().cloned().collect();
        let idx = self.rng.gen_range(0..validators.len());
        Some(validators[idx].clone())
    }
}

impl ValidatorInfo {
    pub fn new() -> Self {
        Self {
            address: Address::default(),
            stake: 0,
            performance: ValidatorPerformance::new(),
            last_active: None,
            is_active: false
        }
    }

    pub fn update_performance(&mut self, performance: ValidatorPerformance) {
        self.performance = performance;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_rotation() {
        let mut rotation = ValidatorRotation::new();
        let address = Address::default(); // Use default instead of random
        let performance = ValidatorPerformance::new();

        rotation.update_validator_info(&address, performance.clone());
        assert!(rotation.get_performance(&address).is_some());
    }
} 