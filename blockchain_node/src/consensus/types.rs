use crate::ledger::block::Block;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Types of consensus algorithms supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsensusType {
    /// Proof of Authority
    Poa,
    /// Practical Byzantine Fault Tolerance
    Pbft,
    /// Social Verified BFT
    Svbft,
    /// Delegated Proof of Stake
    Dpos,
    /// Proof of Stake
    Pos,
    /// Paxos
    Paxos,
    /// Raft
    Raft,
    /// Honeybadger BFT
    Honeybadger,
    /// Tendermint
    Tendermint,
    /// Algorand
    Algorand,
    /// Direct Acyclic Graph based consensus
    Dag,
    /// Avalanche consensus
    Avalanche,
    /// Proof of History + Tower BFT
    PohTower,
    /// Dynamic Adaptive consensus
    Adaptive,
    /// Proof of Neural Training
    PrpofOfAI,
}

impl fmt::Display for ConsensusType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConsensusType::Poa => write!(f, "Proof of Authority"),
            ConsensusType::Pbft => write!(f, "Practical Byzantine Fault Tolerance"),
            ConsensusType::Svbft => write!(f, "Social Verified Byzantine Fault Tolerance"),
            ConsensusType::Dpos => write!(f, "Delegated Proof of Stake"),
            ConsensusType::Pos => write!(f, "Proof of Stake"),
            ConsensusType::Paxos => write!(f, "Paxos"),
            ConsensusType::Raft => write!(f, "Raft"),
            ConsensusType::Honeybadger => write!(f, "HoneyBadger BFT"),
            ConsensusType::Tendermint => write!(f, "Tendermint"),
            ConsensusType::Algorand => write!(f, "Algorand"),
            ConsensusType::Dag => write!(f, "DAG-based"),
            ConsensusType::Avalanche => write!(f, "Avalanche"),
            ConsensusType::PohTower => write!(f, "Proof of History + Tower BFT"),
            ConsensusType::Adaptive => write!(f, "Dynamic Adaptive"),
            ConsensusType::PrpofOfAI => write!(f, "Proof of Neural Training"),
        }
    }
}

/// Consensus state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsensusState {
    /// Initial state
    Initializing,
    /// Collecting transactions
    CollectingTxs,
    /// Proposing a block
    Proposing,
    /// Validating a proposed block
    Validating,
    /// Committing a block
    Committing,
    /// Finalizing a block
    Finalizing,
    /// Syncing with network
    Syncing,
    /// View change in progress
    ViewChange,
    /// Checkpoint creation
    Checkpointing,
}

/// Generic consensus message wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    /// SVBFT consensus message
    Svbft(crate::consensus::svbft::ConsensusMessage),
    /// PBFT consensus message
    Pbft(PbftMessage),
    /// Tendermint consensus message
    Tendermint(TendermintMessage),
    /// Paxos consensus message
    Paxos(PaxosMessage),
    /// Raft consensus message
    Raft(RaftMessage),
}

/// PBFT consensus message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PbftMessage {
    /// Pre-prepare message
    PrePrepare {
        /// View number
        view: u64,
        /// Sequence number
        sequence: u64,
        /// Block
        block: Block,
        /// Node ID
        node_id: String,
    },
    /// Prepare message
    Prepare {
        /// View number
        view: u64,
        /// Sequence number
        sequence: u64,
        /// Block hash
        block_hash: Vec<u8>,
        /// Node ID
        node_id: String,
    },
    /// Commit message
    Commit {
        /// View number
        view: u64,
        /// Sequence number
        sequence: u64,
        /// Block hash
        block_hash: Vec<u8>,
        /// Node ID
        node_id: String,
    },
    /// View change message
    ViewChange {
        /// New view
        new_view: u64,
        /// Node ID
        node_id: String,
        /// Last stable checkpoint
        checkpoint: u64,
        /// Prepared messages
        prepared: Vec<PreparedCertificate>,
    },
    /// New view message
    NewView {
        /// New view
        new_view: u64,
        /// Node ID
        node_id: String,
        /// View change messages
        view_changes: Vec<ViewChangeCertificate>,
        /// New sequence number
        new_sequence: u64,
    },
}

/// Tendermint consensus message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TendermintMessage {
    /// Proposal message
    Proposal {
        /// Height
        height: u64,
        /// Round
        round: u32,
        /// Block
        block: Block,
        /// Node ID
        node_id: String,
    },
    /// Prevote message
    Prevote {
        /// Height
        height: u64,
        /// Round
        round: u32,
        /// Block hash
        block_hash: Option<Vec<u8>>,
        /// Node ID
        node_id: String,
    },
    /// Precommit message
    Precommit {
        /// Height
        height: u64,
        /// Round
        round: u32,
        /// Block hash
        block_hash: Option<Vec<u8>>,
        /// Node ID
        node_id: String,
    },
}

/// Paxos consensus message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaxosMessage {
    /// Prepare message
    Prepare {
        /// Proposal number
        proposal_num: u64,
        /// Node ID
        node_id: String,
    },
    /// Promise message
    Promise {
        /// Proposal number
        proposal_num: u64,
        /// Highest accepted proposal number
        accepted_proposal_num: Option<u64>,
        /// Accepted value
        accepted_value: Option<Block>,
        /// Node ID
        node_id: String,
    },
    /// Accept message
    Accept {
        /// Proposal number
        proposal_num: u64,
        /// Value
        value: Block,
        /// Node ID
        node_id: String,
    },
    /// Accepted message
    Accepted {
        /// Proposal number
        proposal_num: u64,
        /// Node ID
        node_id: String,
    },
}

/// Raft consensus message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftMessage {
    /// Request vote message
    RequestVote {
        /// Term
        term: u64,
        /// Candidate ID
        candidate_id: String,
        /// Last log index
        last_log_index: u64,
        /// Last log term
        last_log_term: u64,
    },
    /// Vote response
    VoteResponse {
        /// Term
        term: u64,
        /// Vote granted
        vote_granted: bool,
        /// Node ID
        node_id: String,
    },
    /// Append entries message
    AppendEntries {
        /// Term
        term: u64,
        /// Leader ID
        leader_id: String,
        /// Previous log index
        prev_log_index: u64,
        /// Previous log term
        prev_log_term: u64,
        /// Entries
        entries: Vec<LogEntry>,
        /// Leader commit
        leader_commit: u64,
    },
    /// Append entries response
    AppendEntriesResponse {
        /// Term
        term: u64,
        /// Success flag
        success: bool,
        /// Node ID
        node_id: String,
        /// Match index
        match_index: u64,
    },
}

/// Log entry for Raft
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Term when entry was received by leader
    pub term: u64,
    /// Command
    pub command: Block,
    /// Index in the log
    pub index: u64,
}

/// Certificate of a prepared message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedCertificate {
    /// View number
    pub view: u64,
    /// Sequence number
    pub sequence: u64,
    /// Block hash
    pub block_hash: Vec<u8>,
    /// Prepare messages
    pub prepare_msgs: Vec<String>,
}

/// Certificate of a view change message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewChangeCertificate {
    /// New view
    pub new_view: u64,
    /// Node ID
    pub node_id: String,
    /// Last stable checkpoint
    pub checkpoint: u64,
}

/// Consensus service trait
pub trait ConsensusService: Send + Sync {
    /// Start the consensus service
    fn start(&mut self) -> Result<()>;

    /// Stop the consensus service
    fn stop(&mut self) -> Result<()>;

    /// Get the type of consensus
    fn get_type(&self) -> ConsensusType;

    /// Get the current state of consensus
    fn get_state(&self) -> ConsensusState;

    /// Process a consensus message
    fn process_message(&self, message: ConsensusMessage) -> Result<()>;

    /// Submit a block for consensus
    fn submit_block(&self, block: Block) -> Result<()>;
}

/// Consensus metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsensusMetrics {
    /// Total blocks processed
    pub total_blocks: u64,
    /// Average block time in milliseconds
    pub avg_block_time_ms: u64,
    /// Current transactions per second
    pub current_tps: f64,
    /// Finality time in milliseconds
    pub finality_time_ms: u64,
    /// Number of view changes
    pub view_changes: u64,
    /// Number of timeouts
    pub timeouts: u64,
    /// Average quorum size
    pub avg_quorum_size: f64,
    /// Consensus failures
    pub failures: u64,
    /// Average CPU usage
    pub avg_cpu_usage: f64,
    /// Average memory usage
    pub avg_memory_usage: f64,
}

/// Consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Type of consensus
    pub consensus_type: ConsensusType,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Batch size
    pub batch_size: usize,
    /// Minimum number of validators
    pub min_validators: usize,
    /// Maximum number of validators
    pub max_validators: usize,
    /// Use adaptive quorum sizing
    pub adaptive_quorum: bool,
    /// Social metrics weight
    pub social_weight: f64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            consensus_type: ConsensusType::Svbft,
            timeout_ms: 5000,
            batch_size: 500,
            min_validators: 4,
            max_validators: 100,
            adaptive_quorum: true,
            social_weight: 0.7,
        }
    }
}
