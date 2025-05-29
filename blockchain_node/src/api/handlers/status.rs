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

    // Use placeholder values for now since we don't have monitoring service
    let peer_count = 0;
    let mempool_size = 0;
    let uptime = 0;

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

/// Get list of connected peers (deprecated - use network_monitoring::get_peers instead)
pub async fn get_peers() -> Result<Json<PeerListResponse>, ApiError> {
    // Return empty peer list for now
    Ok(Json(PeerListResponse {
        peers: Vec::new(),
        total: 0,
    }))
}
