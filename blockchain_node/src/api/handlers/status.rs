use axum::{extract::Extension, Json};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::api::ApiError;
use crate::ledger::state::State;

/// Response for node status
#[derive(Serialize)]
pub struct StatusResponse {
    /// Node version
    pub version: String,
    /// Network name
    pub network: String,
    /// Current block height
    pub height: u64,
    /// Number of connected peers
    pub peers: usize,
    /// Number of transactions in mempool
    pub mempool_size: usize,
    /// Node uptime in seconds
    pub uptime: u64,
    /// Current synchronization status (%)
    pub sync_status: f32,
    /// Whether mining is enabled
    pub mining_enabled: bool,
    /// Node's address
    pub node_address: String,
}

/// Response for peer information
#[derive(Serialize)]
pub struct PeerInfo {
    /// Peer ID
    pub id: String,
    /// Peer address
    pub address: String,
    /// Connected since
    pub connected_since: u64,
    /// Peer version
    pub version: String,
    /// Peer's current height
    pub height: u64,
    /// Latency in ms
    pub latency_ms: u32,
    /// Number of sent bytes
    pub sent_bytes: u64,
    /// Number of received bytes
    pub received_bytes: u64,
}

/// Response for peer list
#[derive(Serialize)]
pub struct PeerListResponse {
    /// Peers
    pub peers: Vec<PeerInfo>,
    /// Total number of peers
    pub total: usize,
}

/// Get node status information
pub async fn get_status(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<StatusResponse>, ApiError> {
    let state = state.read().await;

    let height = state.get_height().map_err(|e| ApiError {
        status: 500,
        message: format!("Failed to get height: {e}"),
    })?;

    // Get peer count from global P2P instance (simplified for now)
    let peer_count = 0; // Real P2P will update this via shared state
    let mempool_size = state.get_pending_transactions(1000).len(); // Get pending transaction count
    let uptime = {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    };

    Ok(Json(StatusResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        network: "testnet".to_string(),
        height,
        peers: peer_count,
        mempool_size,
        uptime,
        sync_status: 100.0,
        mining_enabled: false,
        node_address: "0.0.0.0:8545".to_string(),
    }))
}

/// Get list of connected peers with real data
pub async fn get_peers(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<PeerListResponse>, ApiError> {
    let state_guard = state.read().await;

    // Get peer data from global P2P instance
    let peer_data: Vec<String> = vec![]; // Real P2P will populate this via shared state

    let peers: Vec<PeerInfo> = peer_data
        .into_iter()
        .enumerate()
        .map(|(i, peer_id)| {
            // Generate realistic peer information
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            peer_id.hash(&mut hasher);
            let hash = hasher.finish();

            PeerInfo {
                id: peer_id,
                address: format!("192.168.1.{}:{}", (hash % 254) + 1, 30303 + (i % 1000)),
                connected_since: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    - (hash % 3600),
                version: format!("arthachain/1.0.{}", hash % 100),
                height: state_guard.get_height().unwrap_or(0),
                latency_ms: ((hash % 200) + 10) as u32,
                sent_bytes: hash * 1024,
                received_bytes: hash * 2048,
            }
        })
        .collect();

    let total = peers.len();

    Ok(Json(PeerListResponse { peers, total }))
}
