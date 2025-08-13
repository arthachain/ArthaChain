use anyhow::{anyhow, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use crate::consensus::reputation::ReputationManager;
use crate::ledger::block::Block;
use crate::network::types::{NetworkMessage, NodeId};

/// Configuration for the Byzantine Fault Tolerance module
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ByzantineConfig {
    /// Minimum number of confirmations needed for consensus
    pub min_confirmations: usize,
    /// Timeout for waiting for confirmations
    pub confirmation_timeout_ms: u64,
    /// Maximum tolerated Byzantine nodes (f in 3f+1)
    pub max_byzantine_nodes: usize,
    /// Block proposal timeout
    pub block_proposal_timeout_ms: u64,
    /// View change timeout
    pub view_change_timeout_ms: u64,
    /// Batch size for processing transactions
    pub batch_size: usize,
    /// Heartbeat interval
    pub heartbeat_interval_ms: u64,
}

impl Default for ByzantineConfig {
    fn default() -> Self {
        Self {
            min_confirmations: 2,
            confirmation_timeout_ms: 5000,
            max_byzantine_nodes: 1,
            block_proposal_timeout_ms: 10000,
            view_change_timeout_ms: 15000,
            batch_size: 100,
            heartbeat_interval_ms: 1000,
        }
    }
}

/// Status of a consensus round
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsensusStatus {
    /// Initial state
    Initial,
    /// Block proposed, waiting for votes
    Proposed,
    /// Pre-committed by this node
    PreCommitted,
    /// Committed by this node
    Committed,
    /// Finalized (reached consensus)
    Finalized,
    /// Failed to reach consensus
    Failed,
}

/// Type of consensus message
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum ConsensusMessageType {
    /// Propose a new block
    Propose {
        /// Block data
        block_data: Vec<u8>,
        /// Height of the block
        height: u64,
        /// Hash of the block
        block_hash: Vec<u8>,
    },
    /// Pre-vote for a block
    PreVote {
        /// Hash of the block
        block_hash: Vec<u8>,
        /// Height of the block
        height: u64,
        /// Validator signature
        signature: Vec<u8>,
    },
    /// Pre-commit for a block
    PreCommit {
        /// Hash of the block
        block_hash: Vec<u8>,
        /// Height of the block
        height: u64,
        /// Validator signature
        signature: Vec<u8>,
    },
    /// Commit for a block
    Commit {
        /// Hash of the block
        block_hash: Vec<u8>,
        /// Height of the block
        height: u64,
        /// Validator signature
        signature: Vec<u8>,
    },
    /// View change request
    ViewChange {
        /// New view number
        new_view: u64,
        /// Reason for view change
        reason: String,
        /// Validator signature
        signature: Vec<u8>,
    },
    /// Heartbeat to detect node failures
    Heartbeat {
        /// Current view
        view: u64,
        /// Current height
        height: u64,
        /// Timestamp
        timestamp: u64,
    },
}

/// Types of Byzantine faults
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ByzantineFaultType {
    /// Double signing (equivocation)
    DoubleSigning,
    /// Vote withholding
    VoteWithholding,
    /// Block withholding
    BlockWithholding,
    /// Invalid block proposal
    InvalidBlockProposal,
    /// Delayed message delivery
    DelayedMessages,
    /// Inconsistent votes
    InconsistentVotes,
    /// Malformed messages
    MalformedMessages,
    /// Spurious view changes
    SpuriousViewChanges,
    /// Invalid transaction inclusion
    InvalidTransactions,
    /// Selective message transmission
    SelectiveTransmission,
    /// Sybil attack attempt
    SybilAttempt,
    /// Eclipse attack attempt
    EclipseAttempt,
    /// Long-range attack
    LongRangeAttack,
    /// Replay attack
    ReplayAttack,
}

/// Evidence of Byzantine behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineEvidence {
    /// Type of fault
    pub fault_type: ByzantineFaultType,
    /// Node ID of the Byzantine node
    pub node_id: NodeId,
    /// Timestamp when the fault was detected
    pub timestamp: u64,
    /// Related block(s) if applicable
    pub related_blocks: Vec<Vec<u8>>,
    /// Evidence data (specific to the fault type)
    pub data: Vec<u8>,
    /// Description of the fault
    pub description: String,
    /// Reporting nodes
    pub reporters: Vec<NodeId>,
    /// Evidence hash for verification
    pub evidence_hash: Vec<u8>,
}

/// Byzantine fault detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineDetectionConfig {
    /// Maximum acceptable message delay (ms)
    pub max_message_delay_ms: u64,
    /// Minimum number of reporters to consider evidence valid
    pub min_reporters: usize,
    /// Time window for collecting evidence (ms)
    pub evidence_window_ms: u64,
    /// Number of faults before blacklisting
    pub fault_threshold: usize,
    /// Duration of blacklisting (ms)
    pub blacklist_duration_ms: u64,
    /// Enable AI-based detection
    pub enable_ai_detection: bool,
    /// Penalty for Byzantine behavior
    pub penalty_amount: u64,
    /// Enable automatic slashing
    pub enable_slashing: bool,
    /// Required confidence level for AI detection
    pub ai_confidence_threshold: f64,
}

impl Default for ByzantineDetectionConfig {
    fn default() -> Self {
        Self {
            max_message_delay_ms: 5000,
            min_reporters: 3,
            evidence_window_ms: 60000, // 1 minute
            fault_threshold: 5,
            blacklist_duration_ms: 3600000, // 1 hour
            enable_ai_detection: true,
            penalty_amount: 1000,
            enable_slashing: true,
            ai_confidence_threshold: 0.85,
        }
    }
}

/// Byzantine consensus manager
pub struct ByzantineManager {
    /// Node ID of this validator
    node_id: NodeId,
    /// Total number of validators
    total_validators: usize,
    /// Current view number
    view: RwLock<u64>,
    /// Current consensus height
    height: RwLock<u64>,
    /// Configuration
    config: Arc<RwLock<ByzantineConfig>>,
    /// Message channel for sending consensus messages
    tx_sender: mpsc::Sender<(ConsensusMessageType, NodeId)>,
    /// Message channel for receiving consensus messages
    rx_receiver: RwLock<mpsc::Receiver<ConsensusMessageType>>,
    /// Reputation manager
    reputation_manager: Arc<ReputationManager>,
    /// Active consensus rounds
    active_rounds: RwLock<HashMap<Vec<u8>, ConsensusRound>>,
    /// Known validators
    validators: RwLock<HashSet<NodeId>>,
    /// Last time we received heartbeats from validators
    last_heartbeats: RwLock<HashMap<NodeId, Instant>>,
}

/// Consensus round data
struct ConsensusRound {
    /// Block hash
    block_hash: Vec<u8>,
    /// Block height
    height: u64,
    /// Status of the round
    status: ConsensusStatus,
    /// When the round started
    start_time: Instant,
    /// Pre-votes received from validators
    pre_votes: HashMap<NodeId, Vec<u8>>,
    /// Pre-commits received from validators
    pre_commits: HashMap<NodeId, Vec<u8>>,
    /// Commits received from validators
    commits: HashMap<NodeId, Vec<u8>>,
}

/// Byzantine fault detector
pub struct ByzantineDetector {
    /// Configuration
    config: ByzantineDetectionConfig,
    /// Detected faults by node
    faults: Arc<RwLock<HashMap<NodeId, Vec<ByzantineEvidence>>>>,
    /// Blacklisted nodes
    blacklist: Arc<RwLock<HashMap<NodeId, Instant>>>,
    /// Valid message history for equivocation detection
    message_history: Arc<RwLock<HashMap<NodeId, HashMap<u64, Vec<u8>>>>>,
    /// Pending evidence (not yet fully verified)
    pending_evidence: Arc<RwLock<HashMap<Vec<u8>, (ByzantineEvidence, HashSet<NodeId>)>>>,
    /// Current validators
    validators: Arc<RwLock<HashSet<NodeId>>>,
    /// AI detection model
    #[cfg(feature = "ai_detection")]
    ai_model: Option<Arc<crate::ai_engine::AnomalyDetector>>,
}

impl ByzantineManager {
    /// Create a new ByzantineManager
    pub fn new(
        node_id: NodeId,
        total_validators: usize,
        config: ByzantineConfig,
        tx_sender: mpsc::Sender<(ConsensusMessageType, NodeId)>,
        rx_receiver: mpsc::Receiver<ConsensusMessageType>,
        reputation_manager: Arc<ReputationManager>,
    ) -> Self {
        Self {
            node_id,
            total_validators,
            view: RwLock::new(0),
            height: RwLock::new(0),
            config: Arc::new(RwLock::new(config)),
            tx_sender,
            rx_receiver: RwLock::new(rx_receiver),
            reputation_manager,
            active_rounds: RwLock::new(HashMap::new()),
            validators: RwLock::new(HashSet::new()),
            last_heartbeats: RwLock::new(HashMap::new()),
        }
    }

    /// Start the Byzantine consensus manager
    pub async fn start(&self) -> Result<()> {
        info!(
            "Starting Byzantine consensus manager for node {}",
            self.node_id
        );

        // Start background tasks
        self.start_message_handler().await?;
        self.start_heartbeat_monitor().await?;
        self.start_round_timeout_checker().await?;

        Ok(())
    }

    /// Start the message handler task
    async fn start_message_handler(&self) -> Result<()> {
        let mut rx = self.rx_receiver.write().await;
        let tx_sender = self.tx_sender.clone();
        let node_id = self.node_id;
        let active_rounds = self.active_rounds.clone();
        let view = self.view.clone();
        let height = self.height.clone();
        let validators = self.validators.clone();
        let reputation_manager = self.reputation_manager.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            info!("Byzantine consensus message handler started");

            while let Some(message) = rx.recv().await {
                debug!("Received consensus message: {:?}", message);

                match message {
                    ConsensusMessageType::Propose {
                        block_data,
                        height: msg_height,
                        block_hash,
                    } => {
                        // Handle propose message
                        let current_height = *height.read().await;
                        if msg_height < current_height {
                            warn!(
                                "Received proposal for old height {}, current height {}",
                                msg_height, current_height
                            );
                            continue;
                        }

                        // Create or update round
                        let mut rounds = active_rounds.write().await;
                        if !rounds.contains_key(&block_hash) {
                            rounds.insert(
                                block_hash.clone(),
                                ConsensusRound {
                                    block_hash: block_hash.clone(),
                                    height: msg_height,
                                    status: ConsensusStatus::Proposed,
                                    start_time: Instant::now(),
                                    pre_votes: HashMap::new(),
                                    pre_commits: HashMap::new(),
                                    commits: HashMap::new(),
                                },
                            );
                        }

                        // Send pre-vote for this block
                        // In a real implementation, we would validate the block before voting
                        let signature = vec![1, 2, 3, 4]; // Placeholder
                        let pre_vote = ConsensusMessageType::PreVote {
                            block_hash: block_hash.clone(),
                            height: msg_height,
                            signature,
                        };

                        // Broadcast pre-vote to all validators
                        for &validator in validators.read().await.iter() {
                            if let Err(e) = tx_sender.send((pre_vote.clone(), validator)).await {
                                error!("Failed to send pre-vote: {}", e);
                            }
                        }
                    }

                    ConsensusMessageType::PreVote {
                        block_hash,
                        height: msg_height,
                        signature,
                    } => {
                        // Handle pre-vote message
                        let mut rounds = active_rounds.write().await;
                        if let Some(round) = rounds.get_mut(&block_hash) {
                            // In a real implementation, verify the signature

                            // Record the pre-vote
                            round.pre_votes.insert(node_id, signature);

                            // Check if we have enough pre-votes
                            let config = config.read().await;
                            let min_votes = 2 * config.max_byzantine_nodes + 1;

                            if round.pre_votes.len() >= min_votes {
                                // Send pre-commit
                                let signature = vec![5, 6, 7, 8]; // Placeholder
                                let pre_commit = ConsensusMessageType::PreCommit {
                                    block_hash: block_hash.clone(),
                                    height: msg_height,
                                    signature,
                                };

                                round.status = ConsensusStatus::PreCommitted;

                                // Broadcast pre-commit to all validators
                                for &validator in validators.read().await.iter() {
                                    if let Err(e) =
                                        tx_sender.send((pre_commit.clone(), validator)).await
                                    {
                                        error!("Failed to send pre-commit: {}", e);
                                    }
                                }
                            }
                        }
                    }

                    ConsensusMessageType::PreCommit {
                        block_hash,
                        height: msg_height,
                        signature,
                    } => {
                        // Handle pre-commit message
                        let mut rounds = active_rounds.write().await;
                        if let Some(round) = rounds.get_mut(&block_hash) {
                            // In a real implementation, verify the signature

                            // Record the pre-commit
                            round.pre_commits.insert(node_id, signature);

                            // Check if we have enough pre-commits
                            let config = config.read().await;
                            let min_commits = 2 * config.max_byzantine_nodes + 1;

                            if round.pre_commits.len() >= min_commits {
                                // Send commit
                                let signature = vec![9, 10, 11, 12]; // Placeholder
                                let commit = ConsensusMessageType::Commit {
                                    block_hash: block_hash.clone(),
                                    height: msg_height,
                                    signature,
                                };

                                round.status = ConsensusStatus::Committed;

                                // Broadcast commit to all validators
                                for &validator in validators.read().await.iter() {
                                    if let Err(e) =
                                        tx_sender.send((commit.clone(), validator)).await
                                    {
                                        error!("Failed to send commit: {}", e);
                                    }
                                }
                            }
                        }
                    }

                    ConsensusMessageType::Commit {
                        block_hash,
                        height: msg_height,
                        signature,
                    } => {
                        // Handle commit message
                        let mut rounds = active_rounds.write().await;
                        if let Some(round) = rounds.get_mut(&block_hash) {
                            // In a real implementation, verify the signature

                            // Record the commit
                            round.commits.insert(node_id, signature);

                            // Check if we have enough commits
                            let config = config.read().await;
                            let min_commits = 2 * config.max_byzantine_nodes + 1;

                            if round.commits.len() >= min_commits {
                                // We have consensus!
                                round.status = ConsensusStatus::Finalized;

                                // Update height
                                let mut current_height = height.write().await;
                                if msg_height > *current_height {
                                    *current_height = msg_height;
                                }

                                info!("Consensus reached for block at height {}", msg_height);

                                // In a real implementation, commit the block to the chain
                            }
                        }
                    }

                    ConsensusMessageType::ViewChange {
                        new_view,
                        reason,
                        signature,
                    } => {
                        // Handle view change
                        // In a real implementation, verify the signature and check if view change is justified

                        info!("View change requested to {} because: {}", new_view, reason);

                        let mut current_view = view.write().await;
                        if new_view > *current_view {
                            *current_view = new_view;

                            // In a real implementation, reset the round state and start a new round
                        }
                    }

                    ConsensusMessageType::Heartbeat {
                        view: msg_view,
                        height: msg_height,
                        timestamp,
                    } => {
                        // Update last heartbeat time
                        last_heartbeats
                            .write()
                            .await
                            .insert(node_id, Instant::now());

                        // Check if we need to catch up
                        let current_height = *height.read().await;
                        if msg_height > current_height {
                            // In a real implementation, request missing blocks
                            warn!(
                                "Node is behind: current height {}, network height {}",
                                current_height, msg_height
                            );
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Start the heartbeat monitor
    async fn start_heartbeat_monitor(&self) -> Result<()> {
        let tx_sender = self.tx_sender.clone();
        let node_id = self.node_id;
        let validators = self.validators.clone();
        let view = self.view.clone();
        let height = self.height.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let heartbeat_interval = {
                let config = config.read().await;
                Duration::from_millis(config.heartbeat_interval_ms)
            };

            let mut interval = tokio::time::interval(heartbeat_interval);

            loop {
                interval.tick().await;

                // Send heartbeat to all validators
                let current_view = *view.read().await;
                let current_height = *height.read().await;
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                let heartbeat = ConsensusMessageType::Heartbeat {
                    view: current_view,
                    height: current_height,
                    timestamp,
                };

                for &validator in validators.read().await.iter() {
                    if validator != node_id {
                        if let Err(e) = tx_sender.send((heartbeat.clone(), validator)).await {
                            error!("Failed to send heartbeat: {}", e);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Start the round timeout checker
    async fn start_round_timeout_checker(&self) -> Result<()> {
        let active_rounds = self.active_rounds.clone();
        let tx_sender = self.tx_sender.clone();
        let node_id = self.node_id;
        let validators = self.validators.clone();
        let view = self.view.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            // Check for timed out rounds every second
            let mut interval = tokio::time::interval(Duration::from_secs(1));

            loop {
                interval.tick().await;

                let config = config.read().await;
                let proposal_timeout = Duration::from_millis(config.block_proposal_timeout_ms);
                let view_change_timeout = Duration::from_millis(config.view_change_timeout_ms);

                let mut rounds = active_rounds.write().await;
                let now = Instant::now();

                let mut timed_out_rounds = Vec::new();

                for (block_hash, round) in rounds.iter() {
                    let elapsed = now.duration_since(round.start_time);

                    match round.status {
                        ConsensusStatus::Initial | ConsensusStatus::Proposed => {
                            if elapsed > proposal_timeout {
                                timed_out_rounds.push(block_hash.clone());
                            }
                        }
                        ConsensusStatus::PreCommitted | ConsensusStatus::Committed => {
                            if elapsed > view_change_timeout {
                                timed_out_rounds.push(block_hash.clone());
                            }
                        }
                        _ => {}
                    }
                }

                // Handle timed out rounds
                for block_hash in timed_out_rounds {
                    if let Some(round) = rounds.get(&block_hash) {
                        info!(
                            "Round for block at height {} timed out with status {:?}",
                            round.height, round.status
                        );

                        // Initiate view change
                        let current_view = *view.read().await;
                        let new_view = current_view + 1;

                        let reason = format!("Round timeout at height {}", round.height);
                        let signature = vec![13, 14, 15, 16]; // Placeholder

                        let view_change = ConsensusMessageType::ViewChange {
                            new_view,
                            reason,
                            signature,
                        };

                        // Broadcast view change to all validators
                        for &validator in validators.read().await.iter() {
                            if let Err(e) = tx_sender.send((view_change.clone(), validator)).await {
                                error!("Failed to send view change: {}", e);
                            }
                        }

                        // Remove the timed out round
                        rounds.remove(&block_hash);
                    }
                }
            }
        });

        Ok(())
    }

    /// Propose a new block
    pub async fn propose_block(&self, block_data: Vec<u8>, height: u64) -> Result<Vec<u8>> {
        // Generate a placeholder block hash
        let mut rng = rand::thread_rng();
        let mut block_hash = Vec::with_capacity(32);

        // Fill with 32 random bytes using u8 range instead of gen::<u8>()
        for _ in 0..32 {
            // Generate a random u8 (0-255)
            let random_byte = rng.gen_range(0..=255);
            block_hash.push(random_byte);
        }

        // Log the proposal
        info!(
            "Proposing block at height {} with hash {:?}",
            height, block_hash
        );

        // Create propose message
        let propose = ConsensusMessageType::Propose {
            block_data,
            height,
            block_hash: block_hash.clone(),
        };

        // Create new round
        let mut rounds = self.active_rounds.write().await;
        rounds.insert(
            block_hash.clone(),
            ConsensusRound {
                block_hash: block_hash.clone(),
                height,
                status: ConsensusStatus::Initial,
                start_time: Instant::now(),
                pre_votes: HashMap::new(),
                pre_commits: HashMap::new(),
                commits: HashMap::new(),
            },
        );

        // Broadcast proposal to all validators
        for &validator in self.validators.read().await.iter() {
            if let Err(e) = self.tx_sender.send((propose.clone(), validator)).await {
                error!("Failed to send proposal: {}", e);
            }
        }

        Ok(block_hash)
    }

    /// Register a validator
    pub async fn register_validator(&self, validator_id: NodeId) {
        self.validators.write().await.insert(validator_id);
    }

    /// Get the current consensus height
    pub async fn get_height(&self) -> u64 {
        *self.height.read().await
    }

    /// Get the current view
    pub async fn get_view(&self) -> u64 {
        *self.view.read().await
    }

    /// Check if a block has been finalized
    pub async fn is_finalized(&self, block_hash: &[u8]) -> bool {
        let rounds = self.active_rounds.read().await;
        if let Some(round) = rounds.get(block_hash) {
            round.status == ConsensusStatus::Finalized
        } else {
            false
        }
    }

    /// Get consensus status for a block
    pub async fn get_consensus_status(&self, block_hash: &[u8]) -> Option<ConsensusStatus> {
        let rounds = self.active_rounds.read().await;
        rounds.get(block_hash).map(|round| round.status)
    }

    /// Update the Byzantine configuration
    pub async fn update_config(&self, config: ByzantineConfig) {
        *self.config.write().await = config;
    }
}

impl ByzantineDetector {
    /// Create a new Byzantine detector
    pub fn new(config: ByzantineDetectionConfig, validators: Arc<RwLock<HashSet<NodeId>>>) -> Self {
        Self {
            config,
            faults: Arc::new(RwLock::new(HashMap::new())),
            blacklist: Arc::new(RwLock::new(HashMap::new())),
            message_history: Arc::new(RwLock::new(HashMap::new())),
            pending_evidence: Arc::new(RwLock::new(HashMap::new())),
            validators,
            #[cfg(feature = "ai_detection")]
            ai_model: None,
        }
    }

    /// Initialize the Byzantine detector
    pub async fn initialize(&mut self) -> Result<()> {
        // Initialize AI model if enabled
        #[cfg(feature = "ai_detection")]
        if self.config.enable_ai_detection {
            self.ai_model = Some(Arc::new(crate::ai_engine::AnomalyDetector::new().await?));
        }

        // Start background tasks for cleaning up old data
        self.start_cleanup_tasks();

        info!("Byzantine fault detector initialized");
        Ok(())
    }

    /// Start background cleanup tasks
    fn start_cleanup_tasks(&self) {
        let blacklist = self.blacklist.clone();
        let blacklist_duration = Duration::from_millis(self.config.blacklist_duration_ms);

        // Cleanup task for blacklist
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;

                let mut bl = blacklist.write().await;
                let now = Instant::now();
                bl.retain(|_, timestamp| now.duration_since(*timestamp) < blacklist_duration);
            }
        });

        // Cleanup task for message history
        let message_history = self.message_history.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(300)).await;

                let mut history = message_history.write().await;
                // Keep only the last 1000 messages per node to prevent memory growth
                for (_, node_history) in history.iter_mut() {
                    if node_history.len() > 1000 {
                        let keys: Vec<u64> = node_history.keys().cloned().collect();
                        let mut sorted_keys = keys;
                        sorted_keys.sort();

                        // Remove oldest entries
                        let to_remove = sorted_keys.len() - 1000;
                        for key in &sorted_keys[0..to_remove] {
                            node_history.remove(key);
                        }
                    }
                }
            }
        });
    }

    /// Check if a node is blacklisted
    pub async fn is_blacklisted(&self, node_id: &NodeId) -> bool {
        let blacklist = self.blacklist.read().await;
        if let Some(timestamp) = blacklist.get(node_id) {
            let now = Instant::now();
            let blacklist_duration = Duration::from_millis(self.config.blacklist_duration_ms);
            return now.duration_since(*timestamp) < blacklist_duration;
        }
        false
    }

    /// Get the number of recorded faults for a node
    pub async fn get_fault_count(&self, node_id: &NodeId) -> usize {
        let faults = self.faults.read().await;
        faults.get(node_id).map_or(0, |f| f.len())
    }

    /// Get all Byzantine faults for a node
    pub async fn get_node_faults(&self, node_id: &NodeId) -> Vec<ByzantineEvidence> {
        let faults = self.faults.read().await;
        faults.get(node_id).cloned().unwrap_or_default()
    }

    /// Report Byzantine behavior
    pub async fn report_fault(
        &self,
        fault_type: ByzantineFaultType,
        node_id: NodeId,
        reporter: NodeId,
        related_blocks: Vec<Vec<u8>>,
        data: Vec<u8>,
        description: String,
    ) -> Result<()> {
        // Check if the reported node is a validator
        let validators = self.validators.read().await;
        if !validators.contains(&node_id) {
            debug!("Ignoring fault report for non-validator node {}", node_id);
            return Ok(());
        }

        // Create evidence
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let evidence = ByzantineEvidence {
            fault_type: fault_type.clone(),
            node_id: node_id.clone(),
            timestamp,
            related_blocks,
            data: data.clone(),
            description,
            reporters: vec![reporter.clone()],
            evidence_hash: self.compute_evidence_hash(&fault_type, &node_id, &data),
        };

        // Process the evidence
        self.process_evidence(evidence).await
    }

    /// Process reported evidence
    async fn process_evidence(&self, evidence: ByzantineEvidence) -> Result<()> {
        let evidence_hash = evidence.evidence_hash.clone();
        let node_id = evidence.node_id.clone();
        let fault_type = evidence.fault_type.clone();

        // Check if this is a duplicate report
        let mut pending = self.pending_evidence.write().await;

        if let Some((existing_evidence, reporters)) = pending.get_mut(&evidence_hash) {
            // Add this reporter if not already reported
            if !reporters.contains(&evidence.reporters[0]) {
                reporters.insert(evidence.reporters[0].clone());
                existing_evidence.reporters = reporters.iter().cloned().collect();

                // If we have enough reports, verify and record the fault
                if reporters.len() >= self.config.min_reporters {
                    let evidence_to_commit = existing_evidence.clone();
                    drop(pending); // Release the lock before verification

                    // Verify using AI if enabled
                    let is_valid = if self.config.enable_ai_detection {
                        self.verify_with_ai(&evidence_to_commit).await?
                    } else {
                        true
                    };

                    if is_valid {
                        // Record the verified fault
                        self.record_verified_fault(evidence_to_commit).await?;

                        // Remove from pending after processing
                        let mut pending = self.pending_evidence.write().await;
                        pending.remove(&evidence_hash);
                    }
                }
            }
        } else {
            // First report of this evidence
            let mut reporters = HashSet::new();
            reporters.insert(evidence.reporters[0].clone());

            pending.insert(evidence_hash.clone(), (evidence.clone(), reporters));

            // If only one reporter is required, process immediately
            if self.config.min_reporters <= 1 {
                drop(pending); // Release the lock before verification

                // Verify using AI if enabled
                let is_valid = if self.config.enable_ai_detection {
                    self.verify_with_ai(&evidence).await?
                } else {
                    true
                };

                if is_valid {
                    // Record the verified fault
                    self.record_verified_fault(evidence).await?;

                    // Remove from pending after processing
                    let mut pending = self.pending_evidence.write().await;
                    pending.remove(&evidence_hash);
                }
            }
        }

        info!(
            "Processed Byzantine fault report for node {}: {:?}",
            node_id, fault_type
        );
        Ok(())
    }

    /// Verify evidence using AI models
    #[cfg(feature = "ai_detection")]
    async fn verify_with_ai(&self, evidence: &ByzantineEvidence) -> Result<bool> {
        if let Some(ai_model) = &self.ai_model {
            // Prepare evidence for AI verification
            let features = self.prepare_evidence_features(evidence).await?;

            // Run AI verification
            let (is_valid, confidence) = ai_model.verify_byzantine_behavior(features).await?;

            if confidence >= self.config.ai_confidence_threshold {
                debug!(
                    "AI verified Byzantine behavior for node {} with confidence {:.2}",
                    evidence.node_id, confidence
                );
                return Ok(is_valid);
            } else {
                debug!(
                    "AI verification confidence too low ({:.2}) for node {}, treating as valid",
                    confidence, evidence.node_id
                );
                // Default to accepting the evidence if confidence is low
                return Ok(true);
            }
        }

        // If AI detection is not available, default to accepting the evidence
        Ok(true)
    }

    // Non-AI version of verify_with_ai for when the feature is disabled
    #[cfg(not(feature = "ai_detection"))]
    async fn verify_with_ai(&self, _evidence: &ByzantineEvidence) -> Result<bool> {
        Ok(true)
    }

    /// Prepare evidence features for AI verification
    #[cfg(feature = "ai_detection")]
    async fn prepare_evidence_features(&self, evidence: &ByzantineEvidence) -> Result<Vec<f32>> {
        // Extract relevant features from the evidence based on fault type
        let mut features = Vec::new();

        // Add basic features
        features.push(evidence.reporters.len() as f32);
        features.push(evidence.related_blocks.len() as f32);
        features.push(evidence.timestamp as f32 / 1_000_000.0); // Normalize timestamp

        // Add fault-type specific features
        match evidence.fault_type {
            ByzantineFaultType::DoubleSigning => {
                // Extract signatures from evidence data
                if evidence.data.len() >= 128 {
                    let sig1_bytes = &evidence.data[0..64];
                    let sig2_bytes = &evidence.data[64..128];

                    // Compare similarity of signatures
                    let similarity = self.compute_similarity(sig1_bytes, sig2_bytes);
                    features.push(similarity);
                }
            }
            ByzantineFaultType::DelayedMessages => {
                // Extract delay time from evidence data
                if evidence.data.len() >= 8 {
                    let delay_bytes = &evidence.data[0..8];
                    if let Ok(delay) = bincode::deserialize::<u64>(delay_bytes) {
                        features.push(delay as f32 / 1000.0); // Convert to seconds
                    }
                }
            }
            _ => {
                // Generic features for other fault types
                features.push(self.get_fault_count(&evidence.node_id).await as f32);

                // Use data size as a feature
                features.push(evidence.data.len() as f32 / 1024.0); // Normalize by KB
            }
        }

        // Pad to ensure fixed length
        while features.len() < 10 {
            features.push(0.0);
        }

        Ok(features)
    }

    /// Compute similarity between two byte slices (simple implementation)
    fn compute_similarity(&self, a: &[u8], b: &[u8]) -> f32 {
        let mut same_bytes = 0;
        let len = a.len().min(b.len());

        for i in 0..len {
            if a[i] == b[i] {
                same_bytes += 1;
            }
        }

        same_bytes as f32 / len as f32
    }

    /// Compute a hash for the evidence
    fn compute_evidence_hash(
        &self,
        fault_type: &ByzantineFaultType,
        node_id: &NodeId,
        data: &[u8],
    ) -> Vec<u8> {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(format!("{:?}", fault_type).as_ref());
        hasher.update(node_id.as_ref());
        hasher.update(data);

        hasher.finalize().to_vec()
    }

    /// Record a verified Byzantine fault
    async fn record_verified_fault(&self, evidence: ByzantineEvidence) -> Result<()> {
        let node_id = evidence.node_id.clone();
        let fault_type = evidence.fault_type.clone();

        // Add to fault history
        let mut faults = self.faults.write().await;
        let node_faults = faults.entry(node_id.clone()).or_insert_with(Vec::new);
        node_faults.push(evidence.clone());

        // Check if we need to blacklist the node
        if node_faults.len() >= self.config.fault_threshold {
            let mut blacklist = self.blacklist.write().await;
            blacklist.insert(node_id.clone(), Instant::now());

            info!(
                "Node {} has been blacklisted due to Byzantine behavior",
                node_id
            );

            // Apply penalties if enabled
            if self.config.enable_slashing {
                self.apply_penalty(&node_id, &fault_type).await?;
            }
        }

        info!(
            "Recorded verified Byzantine fault for node {}: {:?}",
            node_id, fault_type
        );
        Ok(())
    }

    /// Apply penalty for Byzantine behavior
    async fn apply_penalty(&self, node_id: &NodeId, fault_type: &ByzantineFaultType) -> Result<()> {
        // In a real implementation, this would integrate with the staking system
        // to slash the validator's stake

        let penalty = match fault_type {
            ByzantineFaultType::DoubleSigning => self.config.penalty_amount * 2,
            ByzantineFaultType::InvalidBlockProposal => self.config.penalty_amount * 3,
            _ => self.config.penalty_amount,
        };

        info!(
            "Applying penalty of {} to node {} for {:?}",
            penalty, node_id, fault_type
        );

        // In a real implementation:
        // 1. Update the staking contract
        // 2. Record the slash event
        // 3. Potentially trigger validator removal

        Ok(())
    }

    /// Check a block for potential Byzantine behavior
    pub async fn check_block(&self, block: &Block, proposer: &NodeId) -> Result<bool> {
        // If the proposer is blacklisted, reject the block
        if self.is_blacklisted(proposer).await {
            warn!("Rejected block from blacklisted proposer {}", proposer);
            return Ok(false);
        }

        // Check for invalid block structure
        if !self.validate_block_structure(block).await? {
            self.report_fault(
                ByzantineFaultType::InvalidBlockProposal,
                proposer.clone(),
                "system".to_string(),
                vec![block.hash.clone()],
                block.hash.clone(),
                "Invalid block structure".to_string(),
            )
            .await?;

            return Ok(false);
        }

        // Check for invalid transactions
        if !self.validate_block_transactions(block).await? {
            self.report_fault(
                ByzantineFaultType::InvalidTransactions,
                proposer.clone(),
                "system".to_string(),
                vec![block.hash.clone()],
                block.hash.clone(),
                "Block contains invalid transactions".to_string(),
            )
            .await?;

            return Ok(false);
        }

        // Block appears valid from Byzantine perspective
        Ok(true)
    }

    /// Validate block structure
    async fn validate_block_structure(&self, block: &Block) -> Result<bool> {
        // In a real implementation, this would perform comprehensive checks:
        // - Verify block hash is correct
        // - Verify structure and fields
        // - Check timestamps and sequence validity

        // Simple check for demonstration
        if block.hash.is_empty() || block.prev_hash.is_empty() {
            return Ok(false);
        }

        Ok(true)
    }

    /// Validate block transactions
    async fn validate_block_transactions(&self, block: &Block) -> Result<bool> {
        // In a real implementation, this would:
        // - Check for double-spends
        // - Verify all transaction signatures
        // - Check for other transaction-level issues

        // Simple check for demonstration
        if block.txs.is_empty() && !block.is_empty_block {
            return Ok(false);
        }

        Ok(true)
    }

    /// Check for equivocation (double signing)
    pub async fn check_equivocation(
        &self,
        node_id: &NodeId,
        view: u64,
        signature: &[u8],
        block_hash: &[u8],
    ) -> Result<bool> {
        let mut history = self.message_history.write().await;
        let node_history = history.entry(node_id.clone()).or_insert_with(HashMap::new);

        if let Some(existing_sig) = node_history.get(&view) {
            // Check if signatures are for different blocks
            if existing_sig != block_hash {
                // Construct evidence data
                let mut evidence_data = Vec::new();
                evidence_data.extend_from_slice(existing_sig);
                evidence_data.extend_from_slice(block_hash);

                // Report equivocation
                self.report_fault(
                    ByzantineFaultType::DoubleSigning,
                    node_id.clone(),
                    "system".to_string(),
                    vec![block_hash.to_vec()],
                    evidence_data,
                    format!("Equivocation detected for view {}", view),
                )
                .await?;

                return Ok(false);
            }
        } else {
            // Record the signature for this view
            node_history.insert(view, block_hash.to_vec());
        }

        Ok(true)
    }

    /// Get statistics about Byzantine faults
    pub async fn get_statistics(&self) -> HashMap<ByzantineFaultType, usize> {
        let mut stats = HashMap::new();
        let faults = self.faults.read().await;

        for fault_list in faults.values() {
            for evidence in fault_list {
                *stats.entry(evidence.fault_type.clone()).or_insert(0) += 1;
            }
        }

        stats
    }
}
