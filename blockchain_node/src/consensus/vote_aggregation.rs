use crate::ledger::block::Block;
use crate::network::types::NodeId;
use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::timeout;

/// Vote types supported by the aggregator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoteType {
    /// Vote to prepare a block
    Prepare,
    /// Vote to commit a block
    Commit,
    /// Vote for view change
    ViewChange,
    /// Vote for checkpoint
    Checkpoint,
    /// Vote for a proposed leader
    LeaderElection,
    /// Vote for a new view
    NewView,
}

/// Vote from a validator for a specific message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Message hash being voted on
    pub message_hash: Vec<u8>,
    /// Type of vote
    pub vote_type: VoteType,
    /// Height of the blockchain when vote was cast
    pub height: u64,
    /// Round or view number
    pub round: u64,
    /// ID of the validator
    pub validator_id: NodeId,
    /// Timestamp
    pub timestamp: u64,
    /// Signature of the voter
    pub signature: Vec<u8>,
    /// Additional vote metadata
    pub metadata: HashMap<String, String>,
}

impl Vote {
    /// Create a new vote
    pub fn new(
        message_hash: Vec<u8>,
        vote_type: VoteType,
        height: u64,
        round: u64,
        validator_id: NodeId,
        signature: Vec<u8>,
    ) -> Self {
        Self {
            message_hash,
            vote_type,
            height,
            round,
            validator_id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature,
            metadata: HashMap::new(),
        }
    }

    /// Verify the vote signature
    pub fn verify(&self) -> bool {
        // In a real implementation, this would verify the signature
        // against the validator's public key
        !self.signature.is_empty()
    }
}

/// Set of votes for a specific message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteSet {
    /// Message hash
    pub message_hash: Vec<u8>,
    /// Type of votes
    pub vote_type: VoteType,
    /// Block height
    pub height: u64,
    /// Round or view number
    pub round: u64,
    /// Votes by validator ID
    pub votes: HashMap<NodeId, Vote>,
    /// Timestamp when the vote set was created
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
}

impl VoteSet {
    /// Create a new vote set
    pub fn new(message_hash: Vec<u8>, vote_type: VoteType, height: u64, round: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            message_hash,
            vote_type,
            height,
            round,
            votes: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a vote to the set
    pub fn add_vote(&mut self, vote: Vote) -> Result<bool> {
        // Check that the vote matches this set
        if vote.message_hash != self.message_hash
            || vote.vote_type != self.vote_type
            || vote.height != self.height
            || vote.round != self.round
        {
            return Err(anyhow!("Vote does not match this set"));
        }

        // Check if we already have a vote from this validator
        if self.votes.contains_key(&vote.validator_id) {
            // Already voted
            return Ok(false);
        }

        // Add the vote
        self.votes.insert(vote.validator_id.clone(), vote);
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Ok(true)
    }

    /// Get the total number of votes
    pub fn vote_count(&self) -> usize {
        self.votes.len()
    }

    /// Check if the vote set has reached quorum
    pub fn has_quorum(&self, total_validators: usize, threshold_percentage: u8) -> bool {
        if total_validators == 0 {
            return false;
        }

        let threshold = (total_validators * threshold_percentage as usize) / 100;
        self.vote_count() >= threshold
    }

    /// Get all validators who have voted
    pub fn get_voters(&self) -> HashSet<NodeId> {
        self.votes.keys().cloned().collect()
    }
}

/// Configuration for vote aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteAggregatorConfig {
    /// Quorum threshold as percentage (0-100)
    pub quorum_percentage: u8,
    /// Vote timeout in milliseconds
    pub vote_timeout_ms: u64,
    /// Maximum number of pending vote sets
    pub max_pending_vote_sets: usize,
    /// Cleanup interval in milliseconds
    pub cleanup_interval_ms: u64,
    /// Maximum vote age in milliseconds
    pub max_vote_age_ms: u64,
    /// Verify vote signatures
    pub verify_signatures: bool,
}

impl Default for VoteAggregatorConfig {
    fn default() -> Self {
        Self {
            quorum_percentage: 67, // 2/3 majority
            vote_timeout_ms: 5000,
            max_pending_vote_sets: 1000,
            cleanup_interval_ms: 60000, // 1 minute
            max_vote_age_ms: 3600000,   // 1 hour
            verify_signatures: true,
        }
    }
}

/// Key for identifying a vote set
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VoteSetKey {
    /// Message hash
    message_hash: Vec<u8>,
    /// Vote type
    vote_type: VoteType,
    /// Height
    height: u64,
    /// Round
    round: u64,
}

impl VoteSetKey {
    fn new(message_hash: Vec<u8>, vote_type: VoteType, height: u64, round: u64) -> Self {
        Self {
            message_hash,
            vote_type,
            height,
            round,
        }
    }

    fn from_vote(vote: &Vote) -> Self {
        Self {
            message_hash: vote.message_hash.clone(),
            vote_type: vote.vote_type,
            height: vote.height,
            round: vote.round,
        }
    }
}

/// Result of vote aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    /// The vote set
    pub vote_set: VoteSet,
    /// Whether quorum was reached
    pub quorum_reached: bool,
    /// Total validators
    pub total_validators: usize,
    /// Required votes for quorum
    pub required_votes: usize,
}

/// Aggregator for consensus votes
pub struct VoteAggregator {
    /// Configuration
    config: RwLock<VoteAggregatorConfig>,
    /// Vote sets by key
    vote_sets: RwLock<HashMap<VoteSetKey, VoteSet>>,
    /// Validators
    validators: Arc<RwLock<HashSet<NodeId>>>,
    /// Vote receiver
    vote_receiver: Option<mpsc::Receiver<Vote>>,
    /// Result sender
    result_sender: Option<mpsc::Sender<AggregationResult>>,
    /// Running flag
    running: RwLock<bool>,
    /// Last cleanup time
    last_cleanup: RwLock<Instant>,
}

impl VoteAggregator {
    /// Create a new vote aggregator
    pub fn new(
        config: VoteAggregatorConfig,
        validators: Arc<RwLock<HashSet<NodeId>>>,
        vote_receiver: Option<mpsc::Receiver<Vote>>,
        result_sender: Option<mpsc::Sender<AggregationResult>>,
    ) -> Self {
        Self {
            config: RwLock::new(config),
            vote_sets: RwLock::new(HashMap::new()),
            validators,
            vote_receiver,
            result_sender,
            running: RwLock::new(false),
            last_cleanup: RwLock::new(Instant::now()),
        }
    }

    /// Start the vote aggregator
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(anyhow!("Vote aggregator already running"));
        }

        *running = true;

        // Start vote processing if we have a receiver
        if let Some(receiver) = self.vote_receiver.take() {
            self.start_vote_processing(receiver);
        }

        // Start cleanup task
        self.start_cleanup_task();

        info!("Vote aggregator started");
        Ok(())
    }

    /// Stop the vote aggregator
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(anyhow!("Vote aggregator not running"));
        }

        *running = false;
        info!("Vote aggregator stopped");
        Ok(())
    }

    /// Start vote processing
    fn start_vote_processing(&self, mut receiver: mpsc::Receiver<Vote>) {
        let self_clone = Arc::new(self.clone());

        tokio::spawn(async move {
            while let Some(vote) = receiver.recv().await {
                let is_running = *self_clone.running.read().await;
                if !is_running {
                    break;
                }

                match self_clone.process_vote(vote).await {
                    Ok(Some(result)) => {
                        // If we have a result and a sender, send the result
                        if let Some(sender) = &self_clone.result_sender {
                            let _ = sender.send(result).await;
                        }
                    }
                    Ok(None) => {
                        // Vote processed but no quorum yet
                    }
                    Err(e) => {
                        warn!("Error processing vote: {}", e);
                    }
                }
            }
        });
    }

    /// Start cleanup task
    fn start_cleanup_task(&self) {
        let self_clone = Arc::new(self.clone());

        tokio::spawn(async move {
            let mut interval = {
                let config = self_clone.config.read().await;
                tokio::time::interval(Duration::from_millis(config.cleanup_interval_ms))
            };

            loop {
                interval.tick().await;

                let is_running = *self_clone.running.read().await;
                if !is_running {
                    break;
                }

                if let Err(e) = self_clone.cleanup_old_vote_sets().await {
                    warn!("Error during vote set cleanup: {}", e);
                }
            }
        });
    }

    /// Process a vote
    pub async fn process_vote(&self, vote: Vote) -> Result<Option<AggregationResult>> {
        // Verify the vote if configured
        let config = self.config.read().await;
        if config.verify_signatures && !vote.verify() {
            return Err(anyhow!("Invalid vote signature"));
        }

        // Check if the validator is in our validator set
        let validators = self.validators.read().await;
        if !validators.contains(&vote.validator_id) {
            return Err(anyhow!(
                "Vote from unknown validator: {}",
                vote.validator_id
            ));
        }

        // Get or create the vote set
        let key = VoteSetKey::from_vote(&vote);
        let mut vote_sets = self.vote_sets.write().await;

        // Check if we've reached the maximum number of vote sets
        if vote_sets.len() >= config.max_pending_vote_sets && !vote_sets.contains_key(&key) {
            return Err(anyhow!("Too many pending vote sets"));
        }

        // Get or create the vote set
        let vote_set = vote_sets.entry(key.clone()).or_insert_with(|| {
            VoteSet::new(
                vote.message_hash.clone(),
                vote.vote_type,
                vote.height,
                vote.round,
            )
        });

        // Add the vote
        match vote_set.add_vote(vote) {
            Ok(true) => {
                // Vote was added
                debug!(
                    "Added vote for [{:?}] height={} round={}, total={}",
                    vote_set.vote_type,
                    vote_set.height,
                    vote_set.round,
                    vote_set.vote_count()
                );
            }
            Ok(false) => {
                // Duplicate vote, ignore
                return Ok(None);
            }
            Err(e) => {
                return Err(e);
            }
        }

        // Check if we've reached quorum
        let total_validators = validators.len();
        let quorum_reached = vote_set.has_quorum(total_validators, config.quorum_percentage);

        // Create the result
        let result = if quorum_reached {
            debug!(
                "Quorum reached for [{:?}] height={} round={}, votes={}/{}",
                vote_set.vote_type,
                vote_set.height,
                vote_set.round,
                vote_set.vote_count(),
                total_validators
            );

            let required_votes = (total_validators * config.quorum_percentage as usize) / 100;
            Some(AggregationResult {
                vote_set: vote_set.clone(),
                quorum_reached,
                total_validators,
                required_votes,
            })
        } else {
            None
        };

        Ok(result)
    }

    /// Clean up old vote sets
    async fn cleanup_old_vote_sets(&self) -> Result<usize> {
        let config = self.config.read().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let max_age_secs = config.max_vote_age_ms / 1000;

        let mut to_remove = Vec::new();
        {
            let vote_sets = self.vote_sets.read().await;
            for (key, vote_set) in vote_sets.iter() {
                if now - vote_set.updated_at > max_age_secs {
                    to_remove.push(key.clone());
                }
            }
        }

        if !to_remove.is_empty() {
            let mut vote_sets = self.vote_sets.write().await;
            for key in &to_remove {
                vote_sets.remove(key);
            }
        }

        // Update last cleanup time
        let mut last_cleanup = self.last_cleanup.write().await;
        *last_cleanup = Instant::now();

        debug!("Cleaned up {} old vote sets", to_remove.len());
        Ok(to_remove.len())
    }

    /// Get vote set by key parameters
    pub async fn get_vote_set(
        &self,
        message_hash: &[u8],
        vote_type: VoteType,
        height: u64,
        round: u64,
    ) -> Option<VoteSet> {
        let key = VoteSetKey::new(message_hash.to_vec(), vote_type, height, round);
        let vote_sets = self.vote_sets.read().await;
        vote_sets.get(&key).cloned()
    }

    /// Get all vote sets for a specific height
    pub async fn get_vote_sets_for_height(&self, height: u64) -> Vec<VoteSet> {
        let vote_sets = self.vote_sets.read().await;
        vote_sets
            .iter()
            .filter(|(key, _)| key.height == height)
            .map(|(_, vs)| vs.clone())
            .collect()
    }

    /// Get all vote sets for a specific type
    pub async fn get_vote_sets_for_type(&self, vote_type: VoteType) -> Vec<VoteSet> {
        let vote_sets = self.vote_sets.read().await;
        vote_sets
            .iter()
            .filter(|(key, _)| key.vote_type == vote_type)
            .map(|(_, vs)| vs.clone())
            .collect()
    }

    /// Create a vote
    pub fn create_vote(
        &self,
        message_hash: Vec<u8>,
        vote_type: VoteType,
        height: u64,
        round: u64,
        validator_id: NodeId,
        private_key: &[u8],
    ) -> Result<Vote> {
        // In a real implementation, this would sign the vote with the private key
        // For now, just create a dummy signature
        let signature = self.sign_vote(
            &message_hash,
            vote_type,
            height,
            round,
            &validator_id,
            private_key,
        )?;

        Ok(Vote::new(
            message_hash,
            vote_type,
            height,
            round,
            validator_id,
            signature,
        ))
    }

    /// Sign a vote with the private key
    fn sign_vote(
        &self,
        message_hash: &[u8],
        vote_type: VoteType,
        height: u64,
        round: u64,
        validator_id: &str,
        private_key: &[u8],
    ) -> Result<Vec<u8>> {
        // In a real implementation, this would sign the vote data with the private key
        // For now, just create a dummy signature
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(message_hash);
        hasher.update(&[vote_type as u8]);
        hasher.update(&height.to_be_bytes());
        hasher.update(&round.to_be_bytes());
        hasher.update(validator_id.as_bytes());
        let signature = hasher.finalize().to_vec();

        Ok(signature)
    }

    /// Wait for quorum on a specific vote set
    pub async fn wait_for_quorum(
        &self,
        message_hash: &[u8],
        vote_type: VoteType,
        height: u64,
        round: u64,
    ) -> Result<AggregationResult> {
        let config = self.config.read().await;
        let timeout_duration = Duration::from_millis(config.vote_timeout_ms);

        // Create a future that resolves when quorum is reached
        let self_clone = Arc::new(self.clone());
        let message_hash = message_hash.to_vec();

        let wait_future = async move {
            loop {
                // Check if we already have quorum
                if let Some(vote_set) = self_clone
                    .get_vote_set(&message_hash, vote_type, height, round)
                    .await
                {
                    let validators = self_clone.validators.read().await;
                    let config = self_clone.config.read().await;
                    if vote_set.has_quorum(validators.len(), config.quorum_percentage) {
                        let required_votes =
                            (validators.len() * config.quorum_percentage as usize) / 100;
                        return Ok(AggregationResult {
                            vote_set,
                            quorum_reached: true,
                            total_validators: validators.len(),
                            required_votes,
                        });
                    }
                }

                // Wait a bit before checking again
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        };

        // Wait with timeout
        match timeout(timeout_duration, wait_future).await {
            Ok(result) => result,
            Err(_) => Err(anyhow!("Timeout waiting for quorum")),
        }
    }

    /// Update configuration
    pub async fn update_config(&self, config: VoteAggregatorConfig) -> Result<()> {
        let mut cfg = self.config.write().await;
        *cfg = config;
        Ok(())
    }

    /// Count votes for a specific message
    pub async fn count_votes(
        &self,
        message_hash: &[u8],
        vote_type: VoteType,
        height: u64,
        round: u64,
    ) -> usize {
        let key = VoteSetKey::new(message_hash.to_vec(), vote_type, height, round);
        let vote_sets = self.vote_sets.read().await;
        vote_sets.get(&key).map(|vs| vs.vote_count()).unwrap_or(0)
    }

    /// Get all pending vote sets
    pub async fn get_all_vote_sets(&self) -> Vec<VoteSet> {
        let vote_sets = self.vote_sets.read().await;
        vote_sets.values().cloned().collect()
    }

    /// Clear all vote sets
    pub async fn clear_vote_sets(&self) -> Result<usize> {
        let mut vote_sets = self.vote_sets.write().await;
        let count = vote_sets.len();
        vote_sets.clear();
        Ok(count)
    }
}

impl Clone for VoteAggregator {
    fn clone(&self) -> Self {
        // This is a partial clone for internal use
        Self {
            config: RwLock::new(self.config.try_read().unwrap_or_default().clone()),
            vote_sets: RwLock::new(HashMap::new()),
            validators: self.validators.clone(),
            vote_receiver: None,
            result_sender: None,
            running: RwLock::new(false),
            last_cleanup: RwLock::new(Instant::now()),
        }
    }
}
