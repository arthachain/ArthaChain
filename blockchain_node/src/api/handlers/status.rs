use std::sync::Arc;
use axum::{extract::Extension, Json};
use tokio::sync::RwLock;
use serde::Serialize;

use crate::ledger::state::State;
use crate::api::ApiError;

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
    Extension(state): Extension<Arc<RwLock<State>>>
) -> Result<Json<StatusResponse>, ApiError> {
    let state = state.read().await;
    
    let height = state.get_height().map_err(|e| ApiError {
        status: 500,
        message: format!("Failed to get height: {}", e)
    })?;

    Ok(Json(StatusResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        network: "testnet".to_string(),
        height,
        peers: 0, // TODO: Implement peer count
        mempool_size: 0, // TODO: Implement mempool size
        uptime: 0, // TODO: Implement uptime
        sync_status: 100.0,
        mining_enabled: false,
        node_address: "0.0.0.0:8545".to_string(),
    }))
}

/// Get list of connected peers
pub async fn get_peers(
    Extension(_state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<PeerListResponse>, ApiError> {
    // TODO: Get actual peers from the p2p network
    // This is a placeholder that would be populated with real peer info
    let peers = Vec::new();
    
    Ok(Json(PeerListResponse {
        peers,
        total: 0,
    }))
} 