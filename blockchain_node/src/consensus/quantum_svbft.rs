use crate::config::Config;
use crate::consensus::view_change::{ViewChangeConfig, ViewChangeManager, ViewState};
use crate::ledger::block::Block;
use crate::ledger::state::State;
use crate::utils::crypto::{dilithium_sign, dilithium_verify};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

// Define NodeId locally as a string type alias
pub type NodeId = String;

/// Phase of the QuantumSVBFT consensus protocol
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
    /// View change phase
    ViewChange,
}

/// Configuration for Quantum SVBFT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSVBFTConfig {
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
    /// Quantum resistance level (0-3, higher is more secure but slower)
    pub quantum_resistance_level: u8,
    /// View change config
    pub view_change_config: ViewChangeConfig,
    /// Max consecutive leader terms
    pub max_consecutive_terms: usize,
    /// Performance monitoring window (number of blocks)
    pub performance_window: usize,
    /// Enable parallel vote validation
    pub parallel_validation: bool,
    /// Checkpoint interval (number of blocks)
    pub checkpoint_interval: u64,
}

impl Default for QuantumSVBFTConfig {
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
            quantum_resistance_level: 2,
            view_change_config: ViewChangeConfig {
                view_timeout: Duration::from_secs(10),
                max_view_changes: 5,
                min_validators: 4,
                leader_election_interval: Duration::from_secs(300),
            },
            max_consecutive_terms: 2,
            performance_window: 100,
            parallel_validation: true,
            checkpoint_interval: 100,
        }
    }
}

/// Message types for QuantumSVBFT consensus
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
        /// Quantum-resistant signature
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
        /// Quantum-resistant signature
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
        /// Quantum-resistant signature
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
        /// Quantum-resistant signature
        signature: Vec<u8>,
    },
    /// New view message (for view change)
    NewView {
        /// New view number
        new_view: u64,
        /// Node ID
        node_id: String,
        /// Quantum-resistant signatures from other nodes
        signatures: Vec<Vec<u8>>,
        /// New proposed block (optional)
        new_block: Option<Block>,
        /// Justification for view change
        justification: ViewChangeJustification,
    },
    /// Proposal message
    Proposal {
        /// View number
        view: u64,
        /// Block
        block: Block,
        /// Node ID
        node_id: String,
        /// Quantum-resistant signature
        signature: Vec<u8>,
    },
    /// View change request message
    ViewChangeRequest {
        /// Current view
        current_view: u64,
        /// New view
        new_view: u64,
        /// Node ID
        node_id: String,
        /// Quantum-resistant signature
        signature: Vec<u8>,
        /// Reason for view change
        reason: ViewChangeReason,
    },
    /// Heartbeat to detect leader failure
    Heartbeat {
        /// View number
        view: u64,
        /// Node ID (leader)
        node_id: String,
        /// Timestamp
        timestamp: u64,
        /// Quantum-resistant signature
        signature: Vec<u8>,
    },
    /// Checkpoint notification
    Checkpoint {
        /// Block height
        height: u64,
        /// Block hash
        block_hash: Vec<u8>,
        /// Node ID
        node_id: String,
        /// Quantum-resistant signature
        signature: Vec<u8>,
    },
}

/// Node performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePerformance {
    /// Block proposal success rate
    pub proposal_success_rate: f64,
    /// Average block production time (ms)
    pub avg_block_time_ms: u64,
    /// Response latency (ms)
    pub response_latency_ms: u64,
    /// Participation rate in consensus
    pub participation_rate: f64,
    /// Last update timestamp
    pub last_update: u64,
}

/// View change reason
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViewChangeReason {
    /// Leader timeout (no heartbeat or proposal)
    LeaderTimeout,
    /// Invalid proposal from leader
    InvalidProposal,
    /// Network partition detected
    NetworkPartition,
    /// Leader performance degradation
    PerformanceDegradation,
    /// Byzantine behavior detected
    ByzantineBehavior,
    /// Regular leader rotation
    ScheduledRotation,
}

/// View change justification (evidence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewChangeJustification {
    /// Reason for view change
    pub reason: ViewChangeReason,
    /// Evidence (depends on reason type)
    pub evidence: Option<Vec<u8>>,
    /// Timestamps for timeouts
    pub timestamps: Option<Vec<u64>>,
    /// Conflicting messages for Byzantine behavior
    pub conflicting_messages: Option<Vec<ConsensusMessage>>,
    /// Performance metrics for degradation
    pub performance_metrics: Option<NodePerformance>,
}

/// QuantumSVBFTConsensus implements a quantum-resistant version of SVBFT
pub struct QuantumSVBFTConsensus {
    /// Configuration
    #[allow(dead_code)]
    config: Config,
    /// QuantumSVBFT specific configuration
    qsvbft_config: QuantumSVBFTConfig,
    /// Blockchain state
    state: Arc<RwLock<State>>,
    /// Channel for receiving consensus messages
    #[allow(dead_code)]
    message_receiver: mpsc::Receiver<ConsensusMessage>,
    /// Channel for sending consensus messages
    message_sender: mpsc::Sender<ConsensusMessage>,
    /// Channel for receiving shutdown signal
    #[allow(dead_code)]
    shutdown_receiver: broadcast::Receiver<()>,
    /// Channel for sending newly mined blocks
    #[allow(dead_code)]
    block_receiver: mpsc::Receiver<Block>,
    /// Current view number
    current_view: Arc<Mutex<u64>>,
    /// Current consensus phase
    current_phase: Arc<Mutex<ConsensusPhase>>,
    /// View change manager
    view_change_manager: Arc<RwLock<ViewChangeManager>>,
    /// Validators
    validators: Arc<RwLock<HashSet<String>>>,
    /// Node performance metrics
    #[allow(dead_code)]
    node_performance: Arc<RwLock<HashMap<String, NodePerformance>>>,
    /// Running flag
    running: Arc<Mutex<bool>>,
    /// This node's ID
    node_id: String,
    /// Finalized blocks
    finalized_blocks: Arc<Mutex<HashMap<Vec<u8>, Block>>>,
    /// View change history
    #[allow(dead_code)]
    view_change_history: Arc<RwLock<VecDeque<ViewState>>>,
    /// Last leader heartbeat timestamps
    #[allow(dead_code)]
    leader_heartbeats: Arc<RwLock<HashMap<String, u64>>>,
    /// Checkpoints (block_height -> (block_hash, signatures))
    #[allow(dead_code)]
    checkpoints: Arc<RwLock<HashMap<u64, (Vec<u8>, HashMap<String, Vec<u8>>)>>>,
}

impl QuantumSVBFTConsensus {
    /// Create a new QuantumSVBFT consensus instance
    pub async fn new(
        config: Config,
        state: Arc<RwLock<State>>,
        message_sender: mpsc::Sender<ConsensusMessage>,
        message_receiver: mpsc::Receiver<ConsensusMessage>,
        block_receiver: mpsc::Receiver<Block>,
        shutdown_receiver: broadcast::Receiver<()>,
        node_id: String,
        qsvbft_config: Option<QuantumSVBFTConfig>,
    ) -> Result<Self> {
        let qsvbft_config = qsvbft_config.unwrap_or_default();

        // Initialize view change manager
        let view_change_manager = Arc::new(RwLock::new(ViewChangeManager::new(
            qsvbft_config.min_quorum_size.unwrap_or_default(),
            qsvbft_config.view_change_config.clone(),
        )));

        let validators = Arc::new(RwLock::new(HashSet::new()));

        // Initialize history with capacity for tracking recent view changes
        let view_change_history = Arc::new(RwLock::new(VecDeque::with_capacity(10)));

        Ok(Self {
            config,
            qsvbft_config,
            state,
            message_receiver,
            message_sender,
            shutdown_receiver,
            block_receiver,
            current_view: Arc::new(Mutex::new(0)),
            current_phase: Arc::new(Mutex::new(ConsensusPhase::New)),
            view_change_manager,
            validators,
            node_performance: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(Mutex::new(false)),
            node_id,
            finalized_blocks: Arc::new(Mutex::new(HashMap::new())),
            view_change_history,
            leader_heartbeats: Arc::new(RwLock::new(HashMap::new())),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start the consensus process
    pub async fn start(&mut self) -> Result<JoinHandle<()>> {
        // Load validators
        self.initialize_validators().await?;

        // Calculate quorum size based on validator count
        let quorum_size = self.calculate_quorum_size().await;

        // Initialize view change manager with validator set
        {
            let validators = self.validators.read().await;
            let validator_addresses: HashSet<Vec<u8>> =
                validators.iter().map(|v| v.as_bytes().to_vec()).collect();

            let view_change_manager = self.view_change_manager.write().await;
            view_change_manager.initialize(validator_addresses).await?;
        }

        info!(
            "Starting QuantumSVBFT consensus with {} validators, quorum size {}",
            self.validators.read().await.len(),
            quorum_size,
        );

        // Set running flag
        *self.running.lock().await = true;

        // Start leader heartbeat monitoring
        self.start_heartbeat_monitoring();

        // Start the main consensus loop
        let handle = self.run_consensus_loop();

        Ok(handle)
    }

    /// Run the main consensus loop
    fn run_consensus_loop(&self) -> JoinHandle<()> {
        // Create a new message receiver channel for this consensus loop
        // This avoids the cloning issue by creating a dedicated channel
        let (consensus_tx, mut consensus_rx) = broadcast::channel(1000);

        let running = self.running.clone();
        let current_view = self.current_view.clone();
        let current_phase = self.current_phase.clone();
        let validators = self.validators.clone();
        let _state = self.state.clone();
        let node_id = self.node_id.clone();
        let message_sender = self.message_sender.clone();
        let _finalized_blocks = self.finalized_blocks.clone();
        let qsvbft_config = self.qsvbft_config.clone();
        let _node_performance = self.node_performance.clone();
        let _view_change_manager = self.view_change_manager.clone();

        // Create and return the main consensus task
        tokio::spawn(async move {
            info!("QuantumSVBFT consensus loop started with dedicated message handling");

            // Setup interval for periodic timeout checks
            let mut timeout_interval = tokio::time::interval(Duration::from_millis(100));
            let mut heartbeat_interval = tokio::time::interval(Duration::from_secs(1));

            while *running.lock().await {
                tokio::select! {
                    // Handle incoming consensus messages
                    msg_result = consensus_rx.recv() => {
                        match msg_result {
                            Ok(message) => {
                                if let Err(e) = Self::handle_consensus_message(
                                    message,
                                    &current_view,
                                    &current_phase,
                                    &validators,
                                    &_state,
                                    &node_id,
                                    &message_sender,
                                    &_finalized_blocks,
                                    &qsvbft_config,
                                    &_view_change_manager,
                                ).await {
                                    error!("Error handling consensus message: {}", e);
                                }
                            }
                            Err(broadcast::error::RecvError::Lagged(_)) => {
                                warn!("Consensus message receiver lagged, continuing...");
                            }
                            Err(broadcast::error::RecvError::Closed) => {
                                warn!("Consensus message channel closed, exiting loop");
                                break;
                            }
                        }
                    }

                    // Check for phase timeouts
                    _ = timeout_interval.tick() => {
                        if let Err(e) = Self::check_timeouts(
                            &current_view,
                            &current_phase,
                            &validators,
                            &node_id,
                            &message_sender,
                            &qsvbft_config,
                        ).await {
                            error!("Error checking timeouts: {}", e);
                        }
                    }

                    // Heartbeat tick
                    _ = heartbeat_interval.tick() => {
                        // Send periodic heartbeat if we're the leader
                        let view = *current_view.lock().await;
                        if let Ok(is_leader) = Self::is_leader_for_view(view, &node_id, &validators).await {
                            if is_leader {
                                if let Err(e) = Self::send_heartbeat(view, &node_id, &message_sender).await {
                                    error!("Failed to send heartbeat: {}", e);
                                }
                            }
                        }
                    }
                }
            }

            info!("QuantumSVBFT consensus loop terminated");
        })
    }

    /// Handle a new block proposed for consensus
    #[allow(dead_code)]
    async fn handle_new_block(
        block: Block,
        current_view: &Arc<Mutex<u64>>,
        current_phase: &Arc<Mutex<ConsensusPhase>>,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
        validators: &Arc<RwLock<HashSet<String>>>,
    ) -> Result<()> {
        let view = *current_view.lock().await;
        let is_leader = Self::is_leader_for_view(view, node_id, validators).await?;

        // Only the leader can propose blocks
        if is_leader {
            // Sign the block with quantum-resistant signature
            let block_bytes = block.encode_for_signing()?;
            let signature = dilithium_sign(node_id.as_ref(), &block_bytes)?;

            // Create a proposal message
            let proposal = ConsensusMessage::Proposal {
                view,
                block,
                node_id: node_id.to_string(),
                signature,
            };

            // Send proposal to all validators
            message_sender.send(proposal).await?;

            // Update phase
            *current_phase.lock().await = ConsensusPhase::Prepare;
        } else {
            warn!("Non-leader node tried to propose a block: {}", node_id);
        }

        Ok(())
    }

    /// Handle a consensus message
    #[allow(dead_code)]
    async fn handle_consensus_message(
        message: ConsensusMessage,
        current_view: &Arc<Mutex<u64>>,
        current_phase: &Arc<Mutex<ConsensusPhase>>,
        validators: &Arc<RwLock<HashSet<String>>>,
        _state: &Arc<RwLock<State>>,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
        finalized_blocks: &Arc<Mutex<HashMap<Vec<u8>, Block>>>,
        qsvbft_config: &QuantumSVBFTConfig,
        view_change_manager: &Arc<RwLock<ViewChangeManager>>,
    ) -> Result<()> {
        match message {
            ConsensusMessage::Proposal {
                view,
                block,
                node_id: proposer,
                signature,
            } => {
                // Verify the view matches current view
                if view != *current_view.lock().await {
                    warn!(
                        "Received proposal for view {} but current view is {}",
                        view,
                        *current_view.lock().await
                    );
                    return Ok(());
                }

                // Verify the proposer is the leader for this view
                let validators_guard = validators.read().await;
                let expected_leader = Self::select_leader_for_view(view, &validators_guard)?;
                if proposer != expected_leader {
                    warn!(
                        "Received proposal from {} but leader is {}",
                        proposer, expected_leader
                    );
                    return Ok(());
                }

                // Verify the signature
                let block_bytes = block.encode_for_signing()?;
                if !dilithium_verify(proposer.as_ref(), &block_bytes, &signature)? {
                    warn!("Invalid signature for proposal");
                    return Ok(());
                }

                // Process the proposal
                info!("Valid proposal received from leader, sending prepare vote");

                // Sign the prepare vote with quantum-resistant signature
                let block_hash = block.hash().unwrap_or_default();
                let msg_bytes =
                    format!("prepare:{}:{}", view, hex::encode(&block_hash.as_ref())).into_bytes();
                let prepare_sig = dilithium_sign(node_id.as_ref(), &msg_bytes)?;

                // Send prepare vote
                let prepare = ConsensusMessage::Prepare {
                    view,
                    block_hash: block_hash.as_ref().to_vec(),
                    node_id: node_id.to_string(),
                    signature: prepare_sig,
                };

                // Store block in finalized blocks (will be actually finalized later)
                let mut finalized = finalized_blocks.lock().await;
                finalized.insert(block_hash.as_ref().to_vec(), block);

                // Send prepare message
                message_sender.send(prepare).await?;

                // Update phase
                *current_phase.lock().await = ConsensusPhase::Prepare;
            }

            ConsensusMessage::Prepare {
                view,
                block_hash: _,
                node_id: voter,
                signature: _,
            } => {
                // Handle prepare message...
                // For brevity, this is simplified
                if view == *current_view.lock().await {
                    info!("Prepare vote received from {}", voter);

                    // In a real implementation, we would:
                    // 1. Verify signature
                    // 2. Collect prepare votes until quorum
                    // 3. Move to pre-commit phase when quorum reached

                    // For now, just update phase
                    *current_phase.lock().await = ConsensusPhase::PreCommit;
                }
            }

            ConsensusMessage::ViewChangeRequest {
                current_view: req_view,
                new_view,
                node_id: requester,
                signature,
                reason,
            } => {
                info!(
                    "View change request received from {}: {} -> {}, reason: {:?}",
                    requester, req_view, new_view, reason
                );

                // Verify the view is current
                if req_view != *current_view.lock().await {
                    warn!("View change request for outdated view");
                    return Ok(());
                }

                // Verify the requester is a valid validator
                let validators_guard = validators.read().await;
                if !validators_guard.contains(&requester) {
                    warn!("View change request from non-validator");
                    return Ok(());
                }

                // Verify signature
                let msg_bytes = format!("viewchange:{}:{}", req_view, new_view).into_bytes();
                if !dilithium_verify(requester.as_ref(), &msg_bytes, &signature)? {
                    warn!("Invalid signature for view change request");
                    return Ok(());
                }

                // Process view change using the view change manager
                let mut vcm = view_change_manager.write().await;

                // Convert quantum_svbft::ViewChangeReason to view_change::ViewChangeReason
                let view_change_reason = match reason {
                    ViewChangeReason::LeaderTimeout => {
                        crate::consensus::view_change::ViewChangeReason::LeaderTimeout
                    }
                    ViewChangeReason::InvalidProposal => {
                        crate::consensus::view_change::ViewChangeReason::LeaderMisbehavior
                    }
                    ViewChangeReason::NetworkPartition => {
                        crate::consensus::view_change::ViewChangeReason::NetworkPartition
                    }
                    ViewChangeReason::PerformanceDegradation => {
                        crate::consensus::view_change::ViewChangeReason::LeaderMisbehavior
                    }
                    ViewChangeReason::ByzantineBehavior => {
                        crate::consensus::view_change::ViewChangeReason::LeaderMisbehavior
                    }
                    ViewChangeReason::ScheduledRotation => {
                        crate::consensus::view_change::ViewChangeReason::ValidatorSetChange
                    }
                };

                vcm.handle_view_change_request(
                    requester.as_bytes().to_vec(),
                    req_view,
                    new_view,
                    view_change_reason,
                )
                .await?;

                // Check if we should also vote for this view change
                if Self::should_support_view_change(&reason) {
                    // Support the view change by sending our own request
                    Self::initiate_view_change(req_view, new_view, reason, node_id, message_sender)
                        .await?;
                }
            }

            ConsensusMessage::NewView {
                new_view,
                node_id: new_leader,
                signatures,
                new_block,
                justification: _,
            } => {
                info!(
                    "New view message received: view {}, new leader {}",
                    new_view, new_leader
                );

                // Verify the new leader is valid for the new view
                let validators_guard = validators.read().await;
                let expected_leader = Self::select_leader_for_view(new_view, &validators_guard)?;
                if new_leader != expected_leader {
                    warn!("New view message with incorrect leader");
                    return Ok(());
                }

                // Verify we have enough signatures (would be implemented in production)
                if signatures.len()
                    < qsvbft_config
                        .min_quorum_size
                        .unwrap_or((2 * validators_guard.len() / 3) + 1)
                {
                    warn!("New view message without enough signatures");
                    return Ok(());
                }

                // Update to new view
                *current_view.lock().await = new_view;
                *current_phase.lock().await = ConsensusPhase::New;

                info!(
                    "View change completed: new view {}, leader {}",
                    new_view, new_leader
                );

                // If there's a new block proposed, process it
                if let Some(_block) = new_block {
                    // Process the first block of the new view
                    if new_leader == node_id {
                        // We're the new leader, let others know
                        Self::send_heartbeat(new_view, node_id, message_sender).await?;
                    }
                }
            }

            ConsensusMessage::Heartbeat {
                view: _,
                node_id: _leader,
                timestamp: _,
                signature: _,
            } => {
                // Update leader heartbeat timestamp
                // Implementation omitted for brevity
            }

            // Other message types would be handled similarly...
            _ => {
                // Default case for other message types
                debug!("Received other consensus message type");
            }
        }

        Ok(())
    }

    /// Check for timeouts in the current consensus phase
    async fn check_timeouts(
        current_view: &Arc<Mutex<u64>>,
        _current_phase: &Arc<Mutex<ConsensusPhase>>,
        validators: &Arc<RwLock<HashSet<String>>>,
        _node_id: &str,
        _message_sender: &mpsc::Sender<ConsensusMessage>,
        _qsvbft_config: &QuantumSVBFTConfig,
    ) -> Result<()> {
        let view = *current_view.lock().await;

        // Check for leader failure
        let validators_guard = validators.read().await;
        if let Ok(_leader) = Self::select_leader_for_view(view, &validators_guard) {
            // In a real implementation, we'd check for leader timeouts
            // and initiate view change if needed
        }

        Ok(())
    }

    /// Initiate a view change
    #[allow(dead_code)]
    async fn initiate_view_change(
        current_view: u64,
        new_view: u64,
        reason: ViewChangeReason,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
    ) -> Result<()> {
        info!(
            "Initiating view change: current={}, new={}, reason={:?}",
            current_view, new_view, reason
        );

        // Sign view change request with quantum-resistant signature
        let msg_bytes = format!("viewchange:{}:{}", current_view, new_view).into_bytes();
        let signature = dilithium_sign(node_id.as_ref(), &msg_bytes)?;

        // Send view change request
        let view_change_msg = ConsensusMessage::ViewChangeRequest {
            current_view,
            new_view,
            node_id: node_id.to_string(),
            signature,
            reason,
        };

        message_sender.send(view_change_msg).await?;

        Ok(())
    }

    /// Send heartbeat message
    async fn send_heartbeat(
        view: u64,
        node_id: &str,
        message_sender: &mpsc::Sender<ConsensusMessage>,
    ) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Sign heartbeat with quantum-resistant signature
        let msg_bytes = format!("heartbeat:{}:{}", view, timestamp).into_bytes();
        let signature = dilithium_sign(node_id.as_ref(), &msg_bytes)?;

        // Create and send heartbeat message
        let heartbeat = ConsensusMessage::Heartbeat {
            view,
            node_id: node_id.to_string(),
            timestamp,
            signature,
        };

        message_sender.send(heartbeat).await?;

        Ok(())
    }

    /// Start monitoring leader heartbeats
    fn start_heartbeat_monitoring(&self) {
        // Implementation omitted for brevity
    }

    /// Determine if we should support a view change request
    #[allow(dead_code)]
    fn should_support_view_change(reason: &ViewChangeReason) -> bool {
        match reason {
            ViewChangeReason::LeaderTimeout => true, // Always support timeout-based changes
            ViewChangeReason::ByzantineBehavior => true, // Always support evidence of Byzantine behavior
            ViewChangeReason::InvalidProposal => true, // Support if leader proposed invalid blocks
            _ => false,                                // Be conservative about other reasons
        }
    }

    /// Initialize validator set
    async fn initialize_validators(&self) -> Result<()> {
        // In a production system, this would load validators from state or config
        let mut validators = self.validators.write().await;

        // Add some example validators for demonstration
        validators.insert("validator1".to_string());
        validators.insert("validator2".to_string());
        validators.insert("validator3".to_string());
        validators.insert("validator4".to_string());
        validators.insert(self.node_id.clone());

        Ok(())
    }

    /// Calculate quorum size based on validator count and configuration
    async fn calculate_quorum_size(&self) -> usize {
        let validators = self.validators.read().await;
        let validator_count = validators.len();

        // If min_quorum_size is specified, use that
        if let Some(min_quorum) = self.qsvbft_config.min_quorum_size {
            return std::cmp::max(min_quorum, (2 * validator_count / 3) + 1);
        }

        // Otherwise use Byzantine fault tolerance formula: 2f+1 where f = (n-1)/3
        let f = (validator_count - 1) / 3;
        2 * f + 1
    }

    /// Select leader for a specific view using deterministic algorithm
    async fn is_leader_for_view(
        view: u64,
        node_id: &str,
        validators: &Arc<RwLock<HashSet<String>>>,
    ) -> Result<bool> {
        let validators = validators.read().await;
        let leader = Self::select_leader_for_view(view, &validators)?;
        Ok(leader == node_id)
    }

    /// Select leader for a view
    fn select_leader_for_view(view: u64, validators: &HashSet<String>) -> Result<String> {
        let validators: Vec<String> = validators.iter().cloned().collect();
        if validators.is_empty() {
            return Err(anyhow!("No validators available"));
        }

        // Deterministic leader selection based on view number
        let idx = (view as usize) % validators.len();
        Ok(validators[idx].clone())
    }

    /// Get current view
    pub async fn get_current_view(&self) -> u64 {
        *self.current_view.lock().await
    }

    /// Get current leader
    pub async fn get_current_leader(&self) -> Option<String> {
        let view = *self.current_view.lock().await;
        let validators = self.validators.read().await;
        Self::select_leader_for_view(view, &validators).ok()
    }

    /// Get current consensus phase
    pub async fn get_current_phase(&self) -> ConsensusPhase {
        *self.current_phase.lock().await
    }
}

/// Module with quantum-resistant cryptographic utilities
mod quantum_crypto {
    use anyhow::Result;

    /// Verify Dilithium signature
    #[allow(dead_code)]
    pub fn verify_dilithium(
        _public_key: &[u8],
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<bool> {
        // In a real implementation, this would use quantum-resistant crypto
        // For now, we'll just simulate it
        Ok(true)
    }

    /// Generate Dilithium signature
    #[allow(dead_code)]
    pub fn sign_dilithium(private_key: &[u8], message: &[u8]) -> Result<Vec<u8>> {
        // In a real implementation, this would use quantum-resistant crypto
        // For now, we'll just simulate it
        let mut signature = Vec::new();
        signature.extend_from_slice(message);
        signature.extend_from_slice(private_key);
        Ok(signature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quorum_size_calculation() {
        let config = Config::default();
        let state = Arc::new(RwLock::new(State::new(&config).unwrap()));
        let (message_sender, _) = mpsc::channel(100);
        let (_, message_receiver) = mpsc::channel(100);
        let (_, block_receiver) = mpsc::channel(100);
        let (_shutdown_sender, shutdown_receiver) = broadcast::channel(1);

        let consensus = QuantumSVBFTConsensus::new(
            config,
            state,
            message_sender,
            message_receiver,
            block_receiver,
            shutdown_receiver,
            "test_node".to_string(),
            None,
        )
        .await
        .unwrap();

        // Initialize validators and calculate quorum size
        consensus.initialize_validators().await.unwrap();
        let quorum_size = consensus.calculate_quorum_size().await;

        // Should be 2f+1 where f = (n-1)/3 and n = 5, so f=1 and quorum=3
        assert_eq!(quorum_size, 3);
    }
}
