use serde::{Deserialize, Serialize};
use std::time::SystemTime;

use super::types::SerializableInstant;

// Define NodeId locally as a string type alias
pub type NodeId = String;

// 🛡️ SPOF ELIMINATION: Redundant Network Messaging Support

/// Message redundancy level for fault tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedundancyLevel {
    None,          // Single path delivery
    Basic,         // 2 path delivery
    High,          // 3 path delivery
    Maximum,       // 5+ path delivery
}

/// Channel route information for redundant delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelRoute {
    pub route_id: String,
    pub route_type: RouteType,
    pub reliability_score: f64,
    pub latency_ms: u64,
    pub is_active: bool,
}

/// Network route types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RouteType {
    Direct,        // Direct peer connection
    Relay,         // Through relay node
    Mesh,          // Mesh network route
    Backup,        // Emergency backup route
}

/// Delivery confirmation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryConfirmation {
    pub confirmed_routes: Vec<String>,
    pub failed_routes: Vec<String>,
    pub confirmation_timestamp: SystemTime,
    pub total_delivery_time_ms: u64,
}

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
    
    // 🛡️ SPOF ELIMINATION: Redundant Network Messaging (SPOF FIX #6)
    pub redundancy_level: RedundancyLevel,
    pub channel_routes: Vec<ChannelRoute>,
    pub backup_routes: Vec<String>,
    pub delivery_confirmation: Option<DeliveryConfirmation>,
    pub message_hash: Option<String>,
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
    
    // 🛡️ SPOF ELIMINATION: Redundant Network Message Types
    RouteDiscovery,           // Discover alternative routes
    RouteHealth,              // Report route health status  
    ChannelFailover,          // Initiate channel failover
    DeliveryConfirmation,     // Confirm message delivery
    RedundantHeartbeat,       // Multi-path heartbeat
    NetworkRedundancyCheck,   // Check network redundancy status
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: NodeId,
    pub address: String,
    pub port: u16,
    pub version: String,
    pub features: Vec<String>,
}

impl NetworkMessage {
    pub fn new(source: NodeId, target: Option<NodeId>, payload: MessagePayload) -> Self {
        let timestamp = SystemTime::now();
        let id = Self::generate_message_id(&source, &timestamp, &payload);

        Self {
            id: id.clone(),
            timestamp,
            source: source.to_string(),
            target: target.map(|id| id.to_string()),
            message_type: MessageType::Handshake,
            payload,
            signature: None,
            sequence: 0,
            
            // 🛡️ SPOF ELIMINATION: Initialize redundant messaging fields
            redundancy_level: RedundancyLevel::Basic, // Default to basic redundancy
            channel_routes: vec![ChannelRoute {
                route_id: "primary".to_string(),
                route_type: RouteType::Direct,
                reliability_score: 1.0,
                latency_ms: 0,
                is_active: true,
            }],
            backup_routes: Vec::new(), // Initialize empty, will be populated by network layer
            delivery_confirmation: None, // Will be set when delivery is confirmed
            message_hash: Some(id), // Use message ID as hash for now
        }
    }

    pub fn generate_message_id(
        source: &NodeId,
        timestamp: &SystemTime,
        payload: &MessagePayload,
    ) -> String {
        use sha2::{Digest, Sha256};
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

    pub fn sign(&mut self, _private_key: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement actual signature generation
        self.signature = Some(vec![]);
        Ok(())
    }

    pub fn verify(&self, _public_key: &[u8]) -> Result<bool, Box<dyn std::error::Error>> {
        // TODO: Implement actual signature verification
        Ok(true)
    }
}
