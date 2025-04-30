use serde::{Deserialize, Serialize};
use std::time::SystemTime;
use crate::types::{BlockHash, TransactionHash, ShardId, NodeId};
use super::types::SerializableInstant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage {
    pub id: String,
    pub timestamp: SystemTime,
    pub source: String,
    pub target: Option<String>,
    pub message_type: MessageType,
    pub payload: MessagePayload,
    pub signature: Option<Vec<u8>>,
    pub sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Handshake,
    Ping,
    Pong,
    BlockProposal,
    BlockVote,
    Transaction,
    TransactionBatch,
    StateSync,
    ViewChange,
    PeerDiscovery,
    PeerList,
    CrossShard,
    Diagnostic,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    Handshake {
        version: String,
        node_type: String,
        features: Vec<String>,
        timestamp: SystemTime,
    },
    
    Ping {
        nonce: u64,
    },
    
    Pong {
        nonce: u64,
        latency: u64,
    },
    
    BlockProposal {
        block_hash: String,
        height: u64,
        transactions: Vec<String>,
        timestamp: SystemTime,
        proposer: String,
    },
    
    BlockVote {
        block_hash: String,
        height: u64,
        vote_type: VoteType,
        voter: String,
        signature: Vec<u8>,
    },
    
    Transaction {
        tx_hash: String,
        from: String,
        to: String,
        amount: u64,
        nonce: u64,
        signature: Vec<u8>,
    },
    
    TransactionBatch {
        transactions: Vec<String>,
        batch_id: String,
        shard_id: u64,
    },
    
    StateSync {
        start_block: u64,
        end_block: u64,
        shard_id: u64,
        sync_type: SyncType,
    },
    
    ViewChange {
        new_view: u64,
        reason: ViewChangeReason,
        proposer: String,
        signature: Vec<u8>,
    },
    
    PeerDiscovery {
        node_id: String,
        address: String,
        port: u16,
        features: Vec<String>,
    },
    
    PeerList {
        peers: Vec<PeerInfo>,
        timestamp: SystemTime,
    },
    
    CrossShard {
        source_shard: u64,
        target_shard: u64,
        message_type: CrossShardMessageType,
        payload: Vec<u8>,
    },
    
    Diagnostic {
        node_id: String,
        metrics: DiagnosticMetrics,
        timestamp: SystemTime,
    },
    
    Error {
        code: u32,
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoteType {
    Prepare,
    Commit,
    ViewChange,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncType {
    Full,
    Headers,
    Transactions,
    State,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViewChangeReason {
    Timeout,
    LeaderFault,
    NetworkPartition,
    ConsensusStuck,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossShardMessageType {
    BlockFinalization,
    TransactionForward,
    StateUpdate,
    ShardReconfiguration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: String,
    pub address: String,
    pub port: u16,
    pub reputation: f64,
    pub last_seen: SerializableInstant,
    pub features: Vec<String>,
    pub geographic_region: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticMetrics {
    pub uptime: u64,
    pub connected_peers: u32,
    pub pending_transactions: u32,
    pub processed_transactions: u64,
    pub block_height: u64,
    pub memory_usage: u64,
    pub cpu_usage: f64,
    pub bandwidth_in: u64,
    pub bandwidth_out: u64,
    pub latency_stats: LatencyStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub min_latency: u64,
    pub max_latency: u64,
    pub avg_latency: f64,
    pub p95_latency: u64,
    pub p99_latency: u64,
}

impl NetworkMessage {
    pub fn new(source: NodeId, target: Option<NodeId>, payload: MessagePayload) -> Self {
        let timestamp = SystemTime::now();
        let id = Self::generate_message_id(&source, &timestamp, &payload);
        
        Self {
            id,
            timestamp,
            source: source.to_string(),
            target: target.map(|id| id.to_string()),
            message_type: MessageType::Handshake,
            payload,
            signature: None,
            sequence: 0,
        }
    }

    pub fn generate_message_id(source: &NodeId, timestamp: &SystemTime, payload: &MessagePayload) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        
        // Hash source + timestamp + serialized payload
        hasher.update(source.as_bytes());
        if let Ok(duration) = timestamp.duration_since(SystemTime::UNIX_EPOCH) {
            hasher.update(duration.as_secs().to_be_bytes());
            hasher.update(duration.subsec_nanos().to_be_bytes());
        }
        if let Ok(payload_bytes) = bincode::serialize(payload) {
            hasher.update(payload_bytes);
        }
        
        format!("{:x}", hasher.finalize())
    }

    pub fn sign(&mut self, private_key: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement actual signature generation
        self.signature = Some(vec![]);
        Ok(())
    }

    pub fn verify(&self, public_key: &[u8]) -> Result<bool, Box<dyn std::error::Error>> {
        // TODO: Implement actual signature verification
        Ok(true)
    }
} 