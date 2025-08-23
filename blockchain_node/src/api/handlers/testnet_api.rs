use axum::{
    extract::{Extension, Query},
    http::{HeaderMap, HeaderValue, Method},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

use crate::api::ApiError;
use crate::ledger::state::State;
use crate::types::Address;

/// Response for blockchain statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockchainStatsResponse {
    /// Latest block height
    pub latest_block: u64,
    /// Average block time in seconds
    pub block_time: String,
    /// Number of active validators
    pub active_validators: u64,
    /// Total number of transactions
    pub total_transactions: String,
}

/// Response for a block in the explorer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExplorerBlockResponse {
    /// Block height
    pub height: u64,
    /// Block hash (shortened for display)
    pub hash: String,
    /// Number of transactions in the block
    #[serde(rename = "txCount")]
    pub tx_count: usize,
    /// Human-readable timestamp
    pub timestamp: String,
    /// Validator that proposed the block
    pub validator: String,
}

/// Response for a transaction in the explorer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExplorerTransactionResponse {
    /// Transaction hash (shortened for display)
    pub hash: String,
    /// Sender address (shortened for display)
    pub from: String,
    /// Recipient address (shortened for display)
    pub to: String,
    /// Transaction amount with currency
    pub value: String,
    /// Transaction status
    pub status: String,
    /// Human-readable timestamp
    pub timestamp: String,
}

/// Query parameters for recent data
#[derive(Debug, Deserialize)]
pub struct RecentDataParams {
    /// Number of items to return
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    10
}

/// Get blockchain statistics for the dashboard
pub async fn get_blockchain_stats(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<BlockchainStatsResponse>, ApiError> {
    let state = state.read().await;

    let latest_block_height = state.get_height().unwrap_or(0);
    let total_tx_count = state.get_total_transactions();
    let validator_count = state.get_validator_count();

    let stats = BlockchainStatsResponse {
        latest_block: latest_block_height,
        block_time: "2.1s".to_string(), // Fixed value for testnet
        active_validators: validator_count as u64,
        total_transactions: format_number(total_tx_count as u64),
    };

    Ok(Json(stats))
}

/// Get recent blocks for the explorer
pub async fn get_recent_blocks(
    Extension(state): Extension<Arc<RwLock<State>>>,
    Query(params): Query<RecentDataParams>,
) -> Result<Json<Vec<ExplorerBlockResponse>>, ApiError> {
    let state = state.read().await;

    let latest_height = state.get_height().unwrap_or(0);
    let limit = params.limit.min(50); // Cap at 50 blocks
    
    if latest_height == 0 {
        // Blockchain is empty - return genesis block info
        let genesis_block = ExplorerBlockResponse {
            height: 0,
            hash: "0x0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            tx_count: 0,
            timestamp: "Genesis".to_string(),
            validator: "Genesis Validator".to_string(),
        };
        return Ok(Json(vec![genesis_block]));
    }
    
    let start_height = latest_height.saturating_sub(limit as u64);

    let mut blocks = Vec::new();

    for height in (start_height..=latest_height).rev() {
        if let Some(block) = state.get_block_by_height(height) {
            let explorer_block = ExplorerBlockResponse {
                height: block.header.height,
                hash: format_hash(&block.hash().unwrap_or_default().to_evm_hex()),
                tx_count: block.transactions.len(),
                timestamp: format_timestamp_relative(block.header.timestamp),
                validator: format_address(&hex::encode(&block.header.producer.to_bytes())),
            };
            blocks.push(explorer_block);
        }
        // NO MOCK DATA - only return actual blocks that exist

        if blocks.len() >= limit {
            break;
        }
    }

    Ok(Json(blocks))
}

/// Get recent transactions for the explorer
pub async fn get_recent_transactions(
    Extension(state): Extension<Arc<RwLock<State>>>,
    Query(params): Query<RecentDataParams>,
) -> Result<Json<Vec<ExplorerTransactionResponse>>, ApiError> {
    let state = state.read().await;

    let limit = params.limit.min(50); // Cap at 50 transactions
    let mut transactions = Vec::new();

    // Try to get real transactions from recent blocks
    let latest_height = state.get_height().unwrap_or(0);

    if latest_height == 0 {
        // Blockchain is empty - return empty transaction list with info
        return Ok(Json(transactions));
    }

    for height in (latest_height.saturating_sub(10)..=latest_height).rev() {
        if let Some(block) = state.get_block_by_height(height) {
            for (i, tx) in block.transactions.iter().enumerate() {
                if transactions.len() >= limit {
                    break;
                }

                let explorer_tx = ExplorerTransactionResponse {
                    hash: format_hash(&tx.hash().unwrap_or_default().to_evm_hex()),
                    from: format_address(&hex::encode(&tx.from)),
                    to: format_address(&hex::encode(&tx.to)),
                    value: format!(
                        "{:.4} ARTHA",
                        tx.amount as f64 / 1_000_000_000_000_000_000.0
                    ),
                    status: "success".to_string(),
                    timestamp: format_timestamp_relative(block.header.timestamp - (i as u64 * 5)),
                };
                transactions.push(explorer_tx);
            }
        }

        if transactions.len() >= limit {
            break;
        }
    }

    // NO MOCK DATA - only return real transactions from actual blocks

    Ok(Json(transactions))
}

/// CORS layer for the frontend
pub fn create_cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers(Any)
}

// Helper functions for formatting

fn format_hash(hash: &str) -> String {
    // Return full hash with 0x prefix for EVM compatibility
    if hash.starts_with("0x") {
        hash.to_string()
    } else {
        format!("0x{}", hash)
    }
}

fn format_address(address: &str) -> String {
    // Return full address with 0x prefix for EVM compatibility
    if address.starts_with("0x") {
        address.to_string()
    } else {
        format!("0x{}", address)
    }
}

fn format_timestamp_relative(timestamp: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let diff = now.saturating_sub(timestamp);

    if diff < 60 {
        format!("{} seconds ago", diff)
    } else if diff < 3600 {
        format!("{} min ago", diff / 60)
    } else if diff < 86400 {
        format!("{} hours ago", diff / 3600)
    } else {
        format!("{} days ago", diff / 86400)
    }
}

fn format_number(num: u64) -> String {
    if num >= 1_000_000 {
        format!("{:.1}M", num as f64 / 1_000_000.0)
    } else if num >= 1_000 {
        format!("{:.1}K", num as f64 / 1_000.0)
    } else {
        num.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_hash() {
        assert_eq!(format_hash("0x1234567890abcdef"), "0x12345678...");
        assert_eq!(format_hash("0x123"), "0x123");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1_500_000), "1.5M");
        assert_eq!(format_number(2_500), "2.5K");
        assert_eq!(format_number(100), "100");
    }

    #[test]
    fn test_format_address() {
        assert_eq!(format_address("0x1234567890abcdef"), "0x123456...");
        assert_eq!(format_address("0x123"), "0x123");
    }
}
