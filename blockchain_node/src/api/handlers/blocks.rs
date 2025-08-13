use axum::{
    extract::{Extension, Path, Query},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::api::ApiError;
use crate::ledger::block::Block;
use crate::ledger::state::State;
use crate::types::Hash;

/// Response for a block
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockResponse {
    /// Block hash
    pub hash: String,
    /// Block height
    pub height: u64,
    /// Previous block hash
    pub prev_hash: String,
    /// Block timestamp
    pub timestamp: u64,
    /// Number of transactions
    pub tx_count: usize,
    /// Merkle root
    pub merkle_root: String,
    /// Block proposer
    pub proposer: String,
    /// Block size in bytes (approximate)
    pub size: usize,
}

impl From<Block> for BlockResponse {
    fn from(block: Block) -> Self {
        Self::from(&block)
    }
}

impl From<&Block> for BlockResponse {
    fn from(block: &Block) -> Self {
        let block_hash = block.hash().unwrap_or_default();
        Self {
            hash: block_hash.to_hex(),
            height: block.header.height,
            prev_hash: hex::encode(block.header.previous_hash.to_bytes()),
            timestamp: block.header.timestamp,
            tx_count: block.transactions.len(),
            merkle_root: hex::encode(block.header.merkle_root.as_bytes()),
            proposer: hex::encode(block.header.producer.as_bytes()),
            // Approximate size based on transactions
            size: block.transactions.len() * 256 + 1024, // Base header size + approx tx size
        }
    }
}

/// Query parameters for block list
#[derive(Debug, Deserialize)]
pub struct BlockQueryParams {
    /// Starting block height
    #[serde(default)]
    pub start: u64,
    /// Maximum number of blocks to return
    #[serde(default = "default_block_limit")]
    pub limit: u64,
}

fn default_block_limit() -> u64 {
    20
}

/// Get the latest block from the chain
pub async fn get_latest_block(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<BlockResponse>, ApiError> {
    let state = state.read().await;

    state
        .latest_block()
        .map(|block| Json(BlockResponse::from(block)))
        .ok_or_else(|| ApiError {
            status: 404,
            message: "No blocks in the chain".to_string(),
        })
}

/// Get a block by its hash
pub async fn get_block_by_hash(
    Path(hash_str): Path<String>,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<BlockResponse>, ApiError> {
    // Convert hash from hex string
    let hash = Hash::from_hex(&hash_str).map_err(|_| ApiError {
        status: 400,
        message: "Invalid block hash format".to_string(),
    })?;

    let state = state.read().await;

    state
        .get_block_by_hash(&hash)
        .map(|block| Json(BlockResponse::from(block)))
        .ok_or_else(|| ApiError {
            status: 404,
            message: "Block not found".to_string(),
        })
}

/// Get a block by its height
pub async fn get_block_by_height(
    Path(height): Path<u64>,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<BlockResponse>, ApiError> {
    let state = state.read().await;

    state
        .get_block_by_height(height)
        .map(|block| Json(BlockResponse::from(block)))
        .ok_or_else(|| ApiError {
            status: 404,
            message: format!("Block at height {height} not found"),
        })
}

/// Get blocks in a range
pub async fn get_blocks(
    Extension(state): Extension<Arc<RwLock<State>>>,
    Query(params): Query<BlockQueryParams>,
) -> Result<Json<Vec<BlockResponse>>, ApiError> {
    let state = state.read().await;

    state
        .get_blocks(params.start, params.limit)
        .map_err(|e| ApiError {
            status: 500,
            message: format!("Failed to get blocks: {e}"),
        })
        .map(|blocks| {
            let responses: Vec<BlockResponse> = blocks.iter().map(BlockResponse::from).collect();
            Json(responses)
        })
}
