use crate::config::Config;
use crate::consensus::view_change::{ViewChangeConfig, ViewChangeManager, ViewChangeMessage};
use crate::ledger::block::Block;
use crate::ledger::state::State;
use crate::types::Address;
use anyhow::{anyhow, Result};
use hex;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;

/// Phase of the HotStuff consensus protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsensusPhase {
    /// Initial state
    New,
    /// Prepare phase
    Prepare,
    /// Pre-commit phase
    PreCommit,
    /// Commit phase
    Commit,
    /// Decide phase
    Decide,
}

/// Configuration for SVBFT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVBFTConfig {
    /// Minimum number of validators required for consensus
    pub min_validators: usize,
    /// Base timeout for consensus phases (ms)
    pub base_timeout_ms: u64,
    /// Maximum timeout for consensus phases (ms)
    pub max_timeout_ms: u64,
    /// Timeout multiplier for each retry
    pub timeout_multiplier: f64,
    /// View change timeout (ms)
    pub view_change_timeout_ms: u64,
    /// Maximum batch size for transactions
    pub max_batch_size: usize,
    /// Minimum quorum size (if None, calculated as 2f+1)
    pub min_quorum_size: Option<usize>,
    /// Enable adaptive quorum sizing
    pub adaptive_quorum: bool,
}

impl Default for SVBFTConfig {
    fn default() -> Self {
        Self {
            min_validators: 4,
            base_timeout_ms: 1000,
            max_timeout_ms: 10000,
            timeout_multiplier: 1.5,
            view_change_timeout_ms: 5000,
            max_batch_size: 500,
            min_quorum_size: None,
            adaptive_quorum: true,
        }
    }
}

/// Message types for SVBFT consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    /// Prepare message
    Prepare {
        /// View number
        view: u64,
        /// Block hash
        block_hash: Vec<u8>,
        /// Node ID
        node_id: String,
        /// Signature
        signature: Vec<u8>,
    },
    /// Pre-commit message
    PreCommit {
        /// View number
        view: u64,
        /// Block hash
        block_hash: Vec<u8>,
        /// Node ID
        node_id: String,
        /// Signature
        signature: Vec<u8>,
    },
    /// Commit message
    Commit {
        /// View number
        view: u64,
        /// Block hash
        block_hash: Vec<u8>,
        /// Node ID
        node_id: String,
        /// Signature
        signature: Vec<u8>,
    },
    /// Decide message
    Decide {
        /// View number
        view: u64,
        /// Block hash
        block_hash: Vec<u8>,
        /// Node ID
        node_id: String,
        /// Signature
        signature: Vec<u8>,
    },
    /// New view message (for view change)
    NewView {
        /// New view number
        new_view: u64,
        /// Node ID
        node_id: String,
        /// Signatures from other nodes
        signatures: Vec<Vec<u8>>,
        /// New proposed block (optional)
        new_block: Option<Block>,
    },
    /// Proposal message
    Proposal {
        /// View number
        view: u64,
        /// Block
        block: Block,
        /// Node ID
        node_id: String,
        /// Signature
        signature: Vec<u8>,
    },
}

/// Node capabilities for adaptive quorum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Network latency (ms)
    pub latency_ms: u32,
    /// Hardware tier (0-3, higher is better)
    pub hardware_tier: u8,
    /// Bandwidth (mbps)
    pub bandwidth_mbps: u32,
    /// Mobile device flag
    pub is_mobile: bool,
    /// Reliability score (0-1)
    pub reliability: f32,
}

/// SVBFT consensus state for a specific view/round
#[derive(Debug)]
struct ConsensusRound {
    /// View number
    view: u64,
    /// Current phase
    phase: ConsensusPhase,
    /// Proposed block
    proposed_block: Option<Block>,
    /// Prepare votes
    prepare_votes: HashMap<String, Vec<u8>>,
    /// Pre-commit votes
    precommit_votes: HashMap<String, Vec<u8>>,
    /// Commit votes
    commit_votes: HashMap<String, Vec<u8>>,
    /// Decide votes
    decide_votes: HashMap<String, Vec<u8>>,
    /// Round start time
    start_time: Instant,
    /// Timeout for current phase
    current_timeout: Duration,
    /// Leader for this view
    leader: String,
    /// Quorum size for this round
    quorum_size: usize,
}

impl ConsensusRound {
    /// Create a new consensus round
    fn new(view: u64, leader: String, quorum_size: usize, base_timeout: Duration) -> Self {
        Self {
            view,
            phase: ConsensusPhase::New,
            proposed_block: None,
            prepare_votes: HashMap::new(),
            precommit_votes: HashMap::new(),
            commit_votes: HashMap::new(),
            decide_votes: HashMap::new(),
            start_time: Instant::now(),
            current_timeout: base_timeout,
            leader,
            quorum_size,
        }
    }

    /// Check if we have a quorum of votes for a phase
    fn has_quorum_for_phase(&self, phase: ConsensusPhase) -> bool {
        let votes = match phase {
            ConsensusPhase::Prepare => &self.prepare_votes,
            ConsensusPhase::PreCommit => &self.precommit_votes,
            ConsensusPhase::Commit => &self.commit_votes,
            ConsensusPhase::Decide => &self.decide_votes,
            _ => return false,
        };

        votes.len() >= self.quorum_size
    }
}

/// SVBFTConsensus implements the Social Verified Byzantine Fault Tolerance consensus
pub struct SVBFTConsensus {
    /// Configuration
    #[allow(dead_code)]
    config: Config,
    /// SVBFT specific configuration
    svbft_config: SVBFTConfig,
    /// Blockchain state
    state: Arc<RwLock<State>>,
    /// Channel for receiving consensus messages
    message_receiver: mpsc::Receiver<ConsensusMessage>,
    /// Channel for sending consensus messages
    message_sender: mpsc::Sender<ConsensusMessage>,
    /// Channel for receiving shutdown signal
    shutdown_receiver: broadcast::Receiver<()>,
    /// Channel for sending newly mined blocks
    block_receiver: mpsc::Receiver<Block>,
    /// Current view number
    current_view: Arc<Mutex<u64>>,
    /// Current consensus round
    current_round: Arc<Mutex<Option<ConsensusRound>>>,
    /// Node capabilities by node_id
    node_capabilities: Arc<RwLock<HashMap<String, NodeCapabilities>>>,
    /// All validators
    validators: Arc<RwLock<HashSet<String>>>,
    /// Running flag
    running: Arc<Mutex<bool>>,
    /// This node's ID
    node_id: String,
    /// Finalized blocks
    finalized_blocks: Arc<Mutex<HashMap<Vec<u8>, Block>>>,
    /// Enhanced view change manager with Byzantine fault tolerance
    view_change_manager: Arc<Mutex<ViewChangeManager>>,
}

impl SVBFTConsensus {
    /// Create a new SVBFT consensus instance
    pub async fn new(
        config: Config,
        state: Arc<RwLock<State>>,
        message_sender: mpsc::Sender<ConsensusMessage>,
        message_receiver: mpsc::Receiver<ConsensusMessage>,
        block_receiver: mpsc::Receiver<Block>,
        shutdown_receiver: broadcast::Receiver<()>,
        svbft_config: Option<SVBFTConfig>,
    ) -> Result<Self> {
        let node_id = config
            .node_id
            .clone()
            .ok_or_else(|| anyhow!("Node ID not set in config"))?;

        let svbft_cfg = svbft_config.unwrap_or_default();

        // Initialize view change manager with Byzantine fault tolerance
        let view_change_config = ViewChangeConfig {
            view_timeout: Duration::from_millis(svbft_cfg.view_change_timeout_ms),
            max_view_changes: 10,
            min_validators: svbft_cfg.min_validators,
            leader_election_interval: Duration::from_millis(svbft_cfg.base_timeout_ms),
        };

        let view_change_manager = ViewChangeManager::new(
            svbft_cfg
                .min_quorum_size
                .unwrap_or(2 * svbft_cfg.min_validators / 3 + 1),
            view_change_config,
        );

        Ok(Self {
            config: config.clone(),
            svbft_config: svbft_cfg,
            state,
            message_receiver,
            message_sender,
            shutdown_receiver,
            block_receiver,
            current_view: Arc::new(Mutex::new(0)),
            current_round: Arc::new(Mutex::new(None)),
            node_capabilities: Arc::new(RwLock::new(HashMap::new())),
            validators: Arc::new(RwLock::new(HashSet::new())),
            running: Arc::new(Mutex::new(false)),
            node_id,
            finalized_blocks: Arc::new(Mutex::new(HashMap::new())),
            view_change_manager: Arc::new(Mutex::new(view_change_manager)),
        })
    }

    /// Start the SVBFT consensus engine
    pub async fn start(&mut self) -> Result<JoinHandle<()>> {
        // Set running flag
        {
            let mut running = self.running.lock().await;
            *running = true;
        }

        // Clone shared data for the task
        let running = self.running.clone();
        let state = self.state.clone();
        let current_view = self.current_view.clone();
        let current_round = self.current_round.clone();
        let node_capabilities = self.node_capabilities.clone();
        let validators = self.validators.clone();
        // Move receivers instead of cloning them
        let mut message_receiver = std::mem::replace(
            &mut self.message_receiver,
            mpsc::channel::<ConsensusMessage>(100).1,
        );
        let message_sender = self.message_sender.clone();
        let mut block_receiver =
            std::mem::replace(&mut self.block_receiver, mpsc::channel::<Block>(100).1);
        let mut shutdown_receiver =
            std::mem::replace(&mut self.shutdown_receiver, broadcast::channel::<()>(1).1);
        let node_id = self.node_id.clone();
        let svbft_config = self.svbft_config.clone();
        let finalized_blocks = self.finalized_blocks.clone();
        let view_change_manager = self.view_change_manager.clone();

        let handle = tokio::spawn(async move {
            info!("SVBFT consensus started");

            // Initialize validators and capabilities
            if let Err(e) = Self::initialize_validators(&validators, &node_capabilities).await {
                error!("Failed to initialize validators: {}", e);
            }

            // Initialize the first view
            let quorum_size =
                Self::calculate_quorum_size(&validators, &node_capabilities, &svbft_config).await;
            // Fix RwLockReadGuard issue
            let validators_set = validators.read().await;
            let validators_copy = validators_set.clone();
            drop(validators_set); // Release the lock before calling select_leader_for_view

            let leader = Self::select_leader_for_view(0, &validators_copy)
                .unwrap_or_else(|| node_id.clone());
            let base_timeout = Duration::from_millis(svbft_config.base_timeout_ms);

            {
                let mut round = current_round.lock().await;
                *round = Some(ConsensusRound::new(0, leader, quorum_size, base_timeout));
            }

            info!(
                "SVBFT consensus initialized with quorum size: {}",
                quorum_size
            );

            // Main consensus loop
            loop {
                // Check for shutdown signal
                if let Ok(()) = shutdown_receiver.try_recv() {
                    info!("SVBFT consensus shutting down");
                    break;
                }

                // Process incoming blocks from miners
                while let Ok(block) = block_receiver.try_recv() {
                    if let Err(e) = Self::handle_new_block(
                        block,
                        &current_view,
                        &current_round,
                        &node_id,
                        &message_sender,
                    )
                    .await
                    {
                        warn!("Error handling new block: {}", e);
                    }
                }

                // Process incoming consensus messages
                while let Ok(message) = message_receiver.try_recv() {
                    if let Err(e) = Self::handle_consensus_message(
                        message,
                        &current_view,
                        &current_round,
                        &validators,
                        &state,
                        &node_id,
                        &message_sender,
                        &finalized_blocks,
                        &svbft_config,
                        &node_capabilities,
                    )
                    .await
                    {
                        warn!("Error handling consensus message: {}", e);
                    }
                }

                // Check for timeouts and trigger view changes if needed
                if let Err(e) = Self::check_timeouts(
                    &current_view,
                    &current_round,
                    &validators,
                    &node_id,
                    &message_sender,
                    &svbft_config,
                    &node_capabilities,
                    &view_change_manager,
                )
                .await
                {
                    warn!("Error checking timeouts: {}", e);
                }

                // Sleep a bit to avoid busy waiting
                tokio::time::sleep(Duration::from_millis(10)).await;
            }

            // Set running flag to false
            {
                let mut running = running.lock().await;
                *running = false;
            }

            info!("SVBFT consensus stopped");
        });

        Ok(handle)
    }

    /// Initialize validators
    async fn initialize_validators(
        validators: &Arc<RwLock<HashSet<String>>>,
        node_capabilities: &Arc<RwLock<HashMap<String, NodeCapabilities>>>,
    ) -> Result<()> {
        // In a real implementation, we would read validator list from state
        // For now, we'll just populate with some dummy data

        let mut validators_set = validators.write().await;
        let mut capabilities_map = node_capabilities.write().await;

        // Add some fake validators
        for i in 1..=10 {
            let node_id = format!("validator{}", i);
            validators_set.insert(node_id.clone());

            // Simulate different capabilities
            let is_mobile = i % 3 == 0;
            let hardware_tier = if is_mobile { 1 } else { 3 };
            let latency_ms = if is_mobile {
                100 + (i * 10)
            } else {
                50 + (i * 5)
            };
            let bandwidth_mbps = if is_mobile { 10 } else { 100 };

            capabilities_map.insert(
                node_id,
                NodeCapabilities {
                    latency_ms: latency_ms as u32,
                    hardware_tier,
                    bandwidth_mbps,
                    is_mobile,
                    reliability: 0.9 - (i as f32 * 0.01),
                },
            );
        }

        info!("Initialized {} validators", validators_set.len());
        Ok(())
    }

    /// Calculate quorum size based on validator set and capabilities
    async fn calculate_quorum_size(
        validators: &Arc<RwLock<HashSet<String>>>,
        node_capabilities: &Arc<RwLock<HashMap<String, NodeCapabilities>>>,
        svbft_config: &SVBFTConfig,
    ) -> usize {
        // If a fixed quorum size is configured, use that
        if let Some(size) = svbft_config.min_quorum_size {
            return size;
        }

        let validators_set = validators.read().await;
        let f = validators_set.len() / 3; // Maximum number of Byzantine nodes we can tolerate

        // Standard BFT requires 2f+1 nodes
        let standard_quorum = 2 * f + 1;

        // If adaptive quorum is disabled, use standard quorum
        if !svbft_config.adaptive_quorum {
            return standard_quorum;
        }

        // For adaptive quorum, adjust based on device capabilities
        let capabilities = node_capabilities.read().await;

        let mobile_count = capabilities.values().filter(|cap| cap.is_mobile).count();

        let low_bandwidth_count = capabilities
            .values()
            .filter(|cap| cap.bandwidth_mbps < 20)
            .count();

        // If more than half of nodes are mobile or have low bandwidth,
        // add an extra node to the quorum for higher reliability
        if mobile_count > validators_set.len() / 2 || low_bandwidth_count > validators_set.len() / 2
        {
            return (2 * f + 1) + 1;
        }

        standard_quorum
    }

    /// Select leader for a view
    fn select_leader_for_view(view: u64, validators: &HashSet<String>) -> Option<String> {
        if validators.is_empty() {
            return None;
        }

        // Simple round-robin leader selection based on view number
        let validators_vec: Vec<_> = validators.iter().cloned().collect();
        let leader_idx = (view as usize) % validators_vec.len();
        Some(validators_vec[leader_idx].clone())
    }

    /// Handle a new block from miners
    async fn handle_new_block(
        block: Block,
        current_view: &Arc<Mutex<u64>>,
        current_round: &Arc<Mutex<Option<ConsensusRound>>>,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
    ) -> Result<()> {
        let view = *current_view.lock().await;
        let mut round_guard = current_round.lock().await;

        let round = match &mut *round_guard {
            Some(r) => r,
            None => return Err(anyhow!("No active consensus round")),
        };

        // If we're the leader for this view, propose the block
        if round.leader == node_id {
            // Create a proposal message - Clone the block before moving it
            let proposal = ConsensusMessage::Proposal {
                view,
                block: block.clone(),
                node_id: node_id.to_string(),
                signature: vec![0; 64], // In real implementation, sign the block hash
            };

            // Send the proposal
            message_sender.send(proposal).await?;

            // Set the proposed block - Clone the block again
            round.proposed_block = Some(block.clone());
            round.phase = ConsensusPhase::Prepare;

            info!("Proposed block {} in view {}", block.hash()?.to_hex(), view);
        } else {
            debug!(
                "Received block but not the leader for view {}, ignoring",
                view
            );
        }

        Ok(())
    }

    /// Handle consensus message - Proposal variant
    async fn handle_proposal(
        view: u64,
        block: Block,
        proposer: String,
        _signature: Vec<u8>,
        round: &mut ConsensusRound,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
    ) -> Result<()> {
        // Verify the proposer is the leader
        if proposer != round.leader {
            warn!("Received proposal from non-leader: {}", proposer);
            return Ok(());
        }

        // In real implementation, verify the signature

        // Set the proposed block
        round.proposed_block = Some(block.clone());
        round.phase = ConsensusPhase::Prepare;

        // Vote for prepare
        let prepare = ConsensusMessage::Prepare {
            view,
            block_hash: block.hash_bytes(),
            node_id: node_id.to_string(),
            signature: vec![0; 64], // In real implementation, sign the block hash
        };

        // Send prepare vote
        message_sender.send(prepare.clone()).await?;

        // Add own vote
        if let ConsensusMessage::Prepare {
            node_id, signature, ..
        } = prepare
        {
            round.prepare_votes.insert(node_id, signature);
        }

        info!(
            "Received proposal for block {} in view {}, sent prepare vote",
            block.hash()?,
            view
        );

        Ok(())
    }

    /// Handle consensus message - Prepare variant
    async fn handle_prepare(
        view: u64,
        block_hash: Vec<u8>,
        voter: String,
        signature: Vec<u8>,
        round: &mut ConsensusRound,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
        validators: &HashSet<String>,
    ) -> Result<()> {
        // Verify the voter is a validator
        if !validators.contains(&voter) {
            warn!("Received prepare from non-validator: {}", voter);
            return Ok(());
        }

        // In real implementation, verify the signature

        // Add the vote
        round.prepare_votes.insert(voter, signature);

        // Check if we have a quorum
        if round.has_quorum_for_phase(ConsensusPhase::Prepare) {
            // Move to pre-commit phase
            round.phase = ConsensusPhase::PreCommit;

            // Create pre-commit message
            let precommit = ConsensusMessage::PreCommit {
                view,
                block_hash: block_hash.clone(),
                node_id: node_id.to_string(),
                signature: vec![0; 64], // In real implementation, sign the block hash
            };

            // Send pre-commit vote
            message_sender.send(precommit.clone()).await?;

            // Add own vote
            if let ConsensusMessage::PreCommit {
                node_id, signature, ..
            } = precommit
            {
                round.precommit_votes.insert(node_id, signature);
            }

            info!(
                "Prepare quorum reached for block {} in view {}, sent pre-commit vote",
                hex::encode(&block_hash),
                view
            );
        }

        Ok(())
    }

    /// Handle consensus message - PreCommit variant
    async fn handle_precommit(
        view: u64,
        block_hash: Vec<u8>,
        voter: String,
        signature: Vec<u8>,
        round: &mut ConsensusRound,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
        validators: &HashSet<String>,
    ) -> Result<()> {
        // Verify the voter is a validator
        if !validators.contains(&voter) {
            warn!("Received pre-commit from non-validator: {}", voter);
            return Ok(());
        }

        // In real implementation, verify the signature

        // Add the vote
        round.precommit_votes.insert(voter, signature);

        // Check if we have a quorum
        if round.has_quorum_for_phase(ConsensusPhase::PreCommit) {
            // Move to commit phase
            round.phase = ConsensusPhase::Commit;

            // Create commit message
            let commit = ConsensusMessage::Commit {
                view,
                block_hash: block_hash.clone(),
                node_id: node_id.to_string(),
                signature: vec![0; 64], // In real implementation, sign the block hash
            };

            // Send commit vote
            message_sender.send(commit.clone()).await?;

            // Add own vote
            if let ConsensusMessage::Commit {
                node_id, signature, ..
            } = commit
            {
                round.commit_votes.insert(node_id, signature);
            }

            info!(
                "Pre-commit quorum reached for block {} in view {}, sent commit vote",
                hex::encode(&block_hash),
                view
            );
        }

        Ok(())
    }

    /// Handle consensus message - Commit variant
    async fn handle_commit(
        view: u64,
        block_hash: Vec<u8>,
        voter: String,
        signature: Vec<u8>,
        round: &mut ConsensusRound,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
        validators: &HashSet<String>,
    ) -> Result<()> {
        // Verify the voter is a validator
        if !validators.contains(&voter) {
            warn!("Received commit from non-validator: {}", voter);
            return Ok(());
        }

        // In real implementation, verify the signature

        // Add the vote
        round.commit_votes.insert(voter, signature);

        // Check if we have a quorum
        if round.has_quorum_for_phase(ConsensusPhase::Commit) {
            // Move to decide phase
            round.phase = ConsensusPhase::Decide;

            // Create decide message
            let decide = ConsensusMessage::Decide {
                view,
                block_hash: block_hash.clone(),
                node_id: node_id.to_string(),
                signature: vec![0; 64], // In real implementation, sign the block hash
            };

            // Send decide vote
            message_sender.send(decide.clone()).await?;

            // Add own vote
            if let ConsensusMessage::Decide {
                node_id, signature, ..
            } = decide
            {
                round.decide_votes.insert(node_id, signature);
            }

            info!(
                "Commit quorum reached for block {} in view {}, sent decide vote",
                hex::encode(&block_hash),
                view
            );
        }

        Ok(())
    }

    /// Handle consensus message
    #[allow(clippy::too_many_arguments)]
    async fn handle_consensus_message(
        message: ConsensusMessage,
        current_view: &Arc<Mutex<u64>>,
        current_round: &Arc<Mutex<Option<ConsensusRound>>>,
        validators: &Arc<RwLock<HashSet<String>>>,
        _state: &Arc<RwLock<State>>,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
        finalized_blocks: &Arc<Mutex<HashMap<Vec<u8>, Block>>>,
        svbft_config: &SVBFTConfig,
        node_capabilities: &Arc<RwLock<HashMap<String, NodeCapabilities>>>,
    ) -> Result<()> {
        let view = *current_view.lock().await;
        let mut round_guard = current_round.lock().await;

        let round = match &mut *round_guard {
            Some(r) => r,
            None => return Err(anyhow!("No active consensus round")),
        };

        let validators_set = validators.read().await;

        match message {
            ConsensusMessage::Proposal {
                view: msg_view,
                block,
                node_id: proposer,
                signature,
            } => {
                // Ignore messages for different views
                if msg_view != view {
                    debug!(
                        "Ignoring proposal for different view: {} (current: {})",
                        msg_view, view
                    );
                    return Ok(());
                }

                Self::handle_proposal(
                    view,
                    block,
                    proposer,
                    signature,
                    round,
                    node_id,
                    message_sender,
                )
                .await?;
            }

            ConsensusMessage::Prepare {
                view: msg_view,
                block_hash,
                node_id: voter,
                signature,
            } => {
                // Ignore messages for different views
                if msg_view != view {
                    debug!(
                        "Ignoring prepare for different view: {} (current: {})",
                        msg_view, view
                    );
                    return Ok(());
                }

                Self::handle_prepare(
                    view,
                    block_hash,
                    voter,
                    signature,
                    round,
                    node_id,
                    message_sender,
                    &validators_set,
                )
                .await?;
            }

            ConsensusMessage::PreCommit {
                view: msg_view,
                block_hash,
                node_id: voter,
                signature,
            } => {
                // Ignore messages for different views
                if msg_view != view {
                    debug!(
                        "Ignoring pre-commit for different view: {} (current: {})",
                        msg_view, view
                    );
                    return Ok(());
                }

                Self::handle_precommit(
                    view,
                    block_hash,
                    voter,
                    signature,
                    round,
                    node_id,
                    message_sender,
                    &validators_set,
                )
                .await?;
            }

            ConsensusMessage::Commit {
                view: msg_view,
                block_hash,
                node_id: voter,
                signature,
            } => {
                // Ignore messages for different views
                if msg_view != view {
                    debug!(
                        "Ignoring commit for different view: {} (current: {})",
                        msg_view, view
                    );
                    return Ok(());
                }

                Self::handle_commit(
                    view,
                    block_hash,
                    voter,
                    signature,
                    round,
                    node_id,
                    message_sender,
                    &validators_set,
                )
                .await?;
            }

            ConsensusMessage::Decide {
                view: msg_view,
                block_hash,
                node_id: voter,
                signature,
            } => {
                // Ignore messages for different views
                if msg_view != view {
                    debug!(
                        "Ignoring decide for different view: {} (current: {})",
                        msg_view, view
                    );
                    return Ok(());
                }

                // Verify the voter is a validator
                if !validators_set.contains(&voter) {
                    warn!("Received decide from non-validator: {}", voter);
                    return Ok(());
                }

                // In real implementation, verify the signature

                // Add the vote
                round.decide_votes.insert(voter, signature);

                // Check if we have a quorum
                if round.has_quorum_for_phase(ConsensusPhase::Decide) {
                    // Finalize the block
                    if let Some(block) = &round.proposed_block {
                        // In real implementation, apply the block to state

                        // Store finalized block
                        let mut finalized = finalized_blocks.lock().await;
                        finalized.insert(block_hash.clone(), block.clone());

                        info!(
                            "Decide quorum reached for block {} in view {}, block finalized",
                            hex::encode(&block_hash),
                            view
                        );

                        // Start a new view/round
                        drop(round_guard); // Drop the lock before starting new view

                        Self::advance_to_next_view(
                            current_view,
                            current_round,
                            validators,
                            node_id,
                            svbft_config,
                            node_capabilities,
                        )
                        .await?;
                    } else {
                        error!("Block finalized but not available in round state");
                    }
                }
            }

            ConsensusMessage::NewView {
                new_view,
                node_id: sender,
                signatures,
                new_block,
            } => {
                // Verify the sender is a validator
                if !validators_set.contains(&sender) {
                    warn!("Received new view from non-validator: {}", sender);
                    return Ok(());
                }

                // Verify the new view is higher than current view
                if new_view <= view {
                    debug!(
                        "Ignoring new view {}, not higher than current view {}",
                        new_view, view
                    );
                    return Ok(());
                }

                // Silence the unused variable warning
                let _unused_signatures = signatures;

                // Accept the new view
                info!("Accepting new view {}", new_view);

                drop(round_guard); // Drop the lock before updating views

                // Update view
                {
                    let mut view_guard = current_view.lock().await;
                    *view_guard = new_view;
                }

                // Start a new round
                let quorum_size =
                    Self::calculate_quorum_size(validators, node_capabilities, svbft_config).await;
                let leader = Self::select_leader_for_view(new_view, &validators_set)
                    .unwrap_or_else(|| node_id.to_string());
                let base_timeout = Duration::from_millis(svbft_config.base_timeout_ms);

                {
                    let mut round_guard = current_round.lock().await;
                    *round_guard = Some(ConsensusRound::new(
                        new_view,
                        leader.clone(),
                        quorum_size,
                        base_timeout,
                    ));

                    // If there's a new block in the view change and we're the leader, propose it
                    if let Some(block) = new_block {
                        if leader == node_id {
                            // Create a proposal message
                            let proposal = ConsensusMessage::Proposal {
                                view: new_view,
                                block: block.clone(),
                                node_id: node_id.to_string(),
                                signature: vec![0; 64], // In real implementation, sign the block hash
                            };

                            // Send the proposal
                            message_sender.send(proposal).await?;

                            // Set the proposed block
                            if let Some(r) = &mut *round_guard {
                                r.proposed_block = Some(block.clone());
                                r.phase = ConsensusPhase::Prepare;
                            }

                            info!(
                                "Proposed block {} in new view {}",
                                block.hash()?.to_hex(),
                                new_view
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check for timeouts and trigger view changes if needed
    async fn check_timeouts(
        current_view: &Arc<Mutex<u64>>,
        current_round: &Arc<Mutex<Option<ConsensusRound>>>,
        validators: &Arc<RwLock<HashSet<String>>>,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
        svbft_config: &SVBFTConfig,
        node_capabilities: &Arc<RwLock<HashMap<String, NodeCapabilities>>>,
        view_change_manager: &Arc<Mutex<ViewChangeManager>>,
    ) -> Result<()> {
        let mut round_guard = current_round.lock().await;

        let round = match &mut *round_guard {
            Some(r) => r,
            None => return Ok(()),
        };

        // Check if we've timed out
        if round.start_time.elapsed() > round.current_timeout {
            info!("Timeout in view {} phase {:?}", round.view, round.phase);

            // Drop the round lock before view change operations
            let view = round.view;
            drop(round_guard);

            // Use enhanced view change manager with Byzantine fault tolerance
            let new_view = view + 1;
            let validator_bytes = node_id.as_bytes().to_vec();

            // Create view change message
            let validator_addr = Address::from_bytes(&validator_bytes)
                .map_err(|_| anyhow!("Invalid validator address"))?;

            let view_change_msg = ViewChangeMessage::new(
                new_view,
                validator_addr.clone(),
                vec![1, 2, 3, 4], // Mock signature - in production, use real crypto
            );

            // Process view change through enhanced manager
            {
                let mut manager = view_change_manager.lock().await;

                // Initialize manager with current validators if not done
                let validators_set = validators.read().await;
                let validator_hashes: HashSet<Vec<u8>> = validators_set
                    .iter()
                    .map(|v| v.as_bytes().to_vec())
                    .collect();
                drop(validators_set);

                if let Err(e) = manager.initialize(validator_hashes).await {
                    warn!("Failed to initialize view change manager: {}", e);
                    return Ok(());
                }

                // Process the view change message
                match manager
                    .process_view_change_message(view_change_msg, validator_addr)
                    .await
                {
                    Ok(view_changed) => {
                        if view_changed {
                            info!("View change executed successfully to view {}", new_view);

                            // Update current view
                            {
                                let mut view_guard = current_view.lock().await;
                                *view_guard = new_view;
                            }

                            // Advance to next view
                            Self::advance_to_next_view(
                                current_view,
                                current_round,
                                validators,
                                node_id,
                                svbft_config,
                                node_capabilities,
                            )
                            .await?;
                        }
                    }
                    Err(e) => {
                        warn!("View change failed: {}", e);

                        // Fallback to traditional view change
                        let new_view_msg = ConsensusMessage::NewView {
                            new_view,
                            node_id: node_id.to_string(),
                            signatures: vec![],
                            new_block: None,
                        };

                        message_sender.send(new_view_msg).await?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Advance to the next view
    async fn advance_to_next_view(
        current_view: &Arc<Mutex<u64>>,
        current_round: &Arc<Mutex<Option<ConsensusRound>>>,
        validators: &Arc<RwLock<HashSet<String>>>,
        node_id: &str,
        svbft_config: &SVBFTConfig,
        node_capabilities: &Arc<RwLock<HashMap<String, NodeCapabilities>>>,
    ) -> Result<()> {
        // Get current view and increment
        let new_view = {
            let mut view_guard = current_view.lock().await;
            *view_guard += 1;
            *view_guard
        };

        // Calculate new quorum size
        let quorum_size =
            Self::calculate_quorum_size(validators, node_capabilities, svbft_config).await;

        // Select leader for new view
        let leader = {
            let validators_set = validators.read().await;
            Self::select_leader_for_view(new_view, &validators_set)
                .unwrap_or_else(|| node_id.to_string())
        };

        // Calculate timeout for new round
        let base_timeout = Duration::from_millis(svbft_config.base_timeout_ms);

        // Create new round
        {
            let mut round_guard = current_round.lock().await;
            *round_guard = Some(ConsensusRound::new(
                new_view,
                leader.clone(),
                quorum_size,
                base_timeout,
            ));
        }

        info!("Advanced to view {} with leader {}", new_view, leader);

        Ok(())
    }

    /// Get the current view number
    pub async fn get_current_view(&self) -> u64 {
        *self.current_view.lock().await
    }

    /// Get the current leader node ID
    pub async fn get_current_leader(&self) -> Option<String> {
        let round_guard = self.current_round.lock().await;
        match &*round_guard {
            Some(round) => Some(round.leader.clone()),
            None => None,
        }
    }

    /// Get the current quorum size
    pub async fn get_quorum_size(&self) -> Option<usize> {
        let round_guard = self.current_round.lock().await;
        match &*round_guard {
            Some(round) => Some(round.quorum_size),
            None => None,
        }
    }

    /// Get the current phase
    pub async fn get_current_phase(&self) -> Option<ConsensusPhase> {
        let round_guard = self.current_round.lock().await;
        match &*round_guard {
            Some(round) => Some(round.phase),
            None => None,
        }
    }

    /// Get all finalized blocks
    pub async fn get_finalized_blocks(&self) -> HashMap<Vec<u8>, Block> {
        self.finalized_blocks.lock().await.clone()
    }

    /// Get all validators in the network
    pub async fn get_validators(&self) -> HashSet<String> {
        self.validators.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quorum_size_calculation() {
        // Create test data
        let validators = Arc::new(RwLock::new(HashSet::new()));
        let node_capabilities = Arc::new(RwLock::new(HashMap::new()));

        // Add validators
        {
            let mut validators_set = validators.write().await;
            let mut capabilities_map = node_capabilities.write().await;

            // Add 10 validators, 4 of which are mobile
            for i in 1..=10 {
                let node_id = format!("validator{}", i);
                validators_set.insert(node_id.clone());

                let is_mobile = i % 3 == 0;
                capabilities_map.insert(
                    node_id,
                    NodeCapabilities {
                        latency_ms: 100,
                        hardware_tier: if is_mobile { 1 } else { 3 },
                        bandwidth_mbps: if is_mobile { 10 } else { 100 },
                        is_mobile,
                        reliability: 0.9,
                    },
                );
            }
        }

        // Test with adaptive quorum disabled
        let config_no_adaptive = SVBFTConfig {
            adaptive_quorum: false,
            min_quorum_size: None,
            ..Default::default()
        };

        let quorum_size_no_adaptive = SVBFTConsensus::calculate_quorum_size(
            &validators,
            &node_capabilities,
            &config_no_adaptive,
        )
        .await;

        // With 10 validators, f=3, so quorum should be 2f+1=7
        assert_eq!(quorum_size_no_adaptive, 7);

        // Test with adaptive quorum enabled
        let config_adaptive = SVBFTConfig {
            adaptive_quorum: true,
            min_quorum_size: None,
            ..Default::default()
        };

        let quorum_size_adaptive = SVBFTConsensus::calculate_quorum_size(
            &validators,
            &node_capabilities,
            &config_adaptive,
        )
        .await;

        // Since 4 out of 10 nodes are mobile (not majority),
        // adaptive quorum should not add extra nodes
        assert_eq!(quorum_size_adaptive, 7);

        // Test with majority of nodes being mobile
        {
            let mut capabilities_map = node_capabilities.write().await;

            // Change capabilities so that 6 out of 10 are mobile
            for i in 1..=10 {
                let node_id = format!("validator{}", i);
                let is_mobile = i <= 6; // First 6 are mobile
                capabilities_map.insert(
                    node_id,
                    NodeCapabilities {
                        latency_ms: 100,
                        hardware_tier: if is_mobile { 1 } else { 3 },
                        bandwidth_mbps: if is_mobile { 10 } else { 100 },
                        is_mobile,
                        reliability: 0.9,
                    },
                );
            }
        }

        let quorum_size_adaptive_majority_mobile = SVBFTConsensus::calculate_quorum_size(
            &validators,
            &node_capabilities,
            &config_adaptive,
        )
        .await;

        // Since majority are mobile, adaptive quorum should add an extra node
        assert_eq!(quorum_size_adaptive_majority_mobile, 8);

        // Test with fixed quorum size
        let config_fixed = SVBFTConfig {
            adaptive_quorum: true,
            min_quorum_size: Some(5),
            ..Default::default()
        };

        let quorum_size_fixed =
            SVBFTConsensus::calculate_quorum_size(&validators, &node_capabilities, &config_fixed)
                .await;

        // Should use the fixed size regardless of other factors
        assert_eq!(quorum_size_fixed, 5);
    }
}
