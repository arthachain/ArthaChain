use anyhow::Result;
use axum::{
    extract::{Extension, Query},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

use arthachain_node::config::Config;
use arthachain_node::ledger::state::State;

/// REAL-TIME ONLY API SERVER - Zero Mock Data
/// This server only returns actual blockchain state data

#[derive(Serialize)]
pub struct RealBlockchainStats {
    pub current_height: u64,
    pub real_block_time_seconds: f64,
    pub actual_transactions_processed: u64,
    pub real_validator_count: u32,
    pub last_block_timestamp: u64,
    pub real_tps: f64,
}

#[derive(Serialize)]
pub struct RealBlockData {
    pub height: u64,
    pub hash: String,
    pub actual_tx_count: usize,
    pub timestamp: u64,
    pub real_proposer: String,
    pub gas_used: u64,
    pub size_bytes: usize,
}

#[derive(Serialize)]
pub struct RealTransactionData {
    pub hash: String,
    pub from: String,
    pub to: String,
    pub amount: u64,
    pub nonce: u64,
    pub block_height: u64,
    pub position_in_block: usize,
}

#[derive(Debug, Deserialize)]
pub struct RecentDataParams {
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    10
}

/// Get REAL blockchain statistics - no fake data
pub async fn get_real_stats(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<RealBlockchainStats>, axum::http::StatusCode> {
    let state = state.read().await;

    // Get actual current height
    let current_height = state.get_height().unwrap_or(0);

    // Calculate real block time from last few blocks
    let real_block_time = if current_height >= 2 {
        let latest_block = state.get_block_by_height(current_height);
        let prev_block = state.get_block_by_height(current_height - 1);

        match (latest_block, prev_block) {
            (Some(latest), Some(prev)) => (latest.header.timestamp - prev.header.timestamp) as f64,
            _ => 0.0,
        }
    } else {
        0.0
    };

    // Count actual transactions from real blocks
    let mut actual_tx_count = 0u64;
    for height in 0..=current_height {
        if let Some(block) = state.get_block_by_height(height) {
            actual_tx_count += block.transactions.len() as u64;
        }
    }

    // Calculate real TPS (transactions per second)
    let real_tps = if current_height >= 10 && real_block_time > 0.0 {
        let mut recent_tx_count = 0u64;
        let start_height = current_height.saturating_sub(10);

        for height in start_height..=current_height {
            if let Some(block) = state.get_block_by_height(height) {
                recent_tx_count += block.transactions.len() as u64;
            }
        }

        recent_tx_count as f64 / (10.0 * real_block_time)
    } else {
        0.0
    };

    // Get real validator count (not mock)
    let real_validator_count = state.get_validator_count() as u32;

    // Get last block timestamp
    let last_timestamp = state
        .get_block_by_height(current_height)
        .map(|b| b.header.timestamp)
        .unwrap_or(0);

    let stats = RealBlockchainStats {
        current_height,
        real_block_time_seconds: real_block_time,
        actual_transactions_processed: actual_tx_count,
        real_validator_count,
        last_block_timestamp: last_timestamp,
        real_tps,
    };

    Ok(Json(stats))
}

/// Get REAL latest block - no mock data
pub async fn get_real_latest_block(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<RealBlockData>, axum::http::StatusCode> {
    let state = state.read().await;

    let current_height = state.get_height().unwrap_or(0);

    if let Some(block) = state.get_block_by_height(current_height) {
        let block_hash = block.hash().unwrap_or_default();

        let real_block = RealBlockData {
            height: block.header.height,
            hash: hex::encode(block_hash.as_bytes()),
            actual_tx_count: block.transactions.len(),
            timestamp: block.header.timestamp,
            real_proposer: hex::encode(&block.header.producer.to_bytes()), // Real producer, not "Validator-01"
            gas_used: 0, // Gas not tracked in this block header
            size_bytes: bincode::serialize(&block).unwrap_or_default().len(),
        };

        Ok(Json(real_block))
    } else {
        Err(axum::http::StatusCode::NOT_FOUND)
    }
}

/// Get REAL recent blocks - only actual blocks, no mock data
pub async fn get_real_recent_blocks(
    Extension(state): Extension<Arc<RwLock<State>>>,
    Query(params): Query<RecentDataParams>,
) -> Result<Json<Vec<RealBlockData>>, axum::http::StatusCode> {
    let state = state.read().await;

    let current_height = state.get_height().unwrap_or(0);
    let mut real_blocks = Vec::new();

    // Only return actual blocks that exist
    let limit = params.limit.min(50);
    let start_height = current_height.saturating_sub(limit as u64);

    for height in (start_height..=current_height).rev() {
        if let Some(block) = state.get_block_by_height(height) {
            let block_hash = block.hash().unwrap_or_default();

            let real_block = RealBlockData {
                height: block.header.height,
                hash: format!("{}...", &hex::encode(block_hash.as_bytes())[0..10]),
                actual_tx_count: block.transactions.len(),
                timestamp: block.header.timestamp,
                real_proposer: format!(
                    "0x{}...",
                    &hex::encode(&block.header.producer.to_bytes())[0..8]
                ),
                gas_used: 0, // Gas not tracked in this block header
                size_bytes: bincode::serialize(&block).unwrap_or_default().len(),
            };
            real_blocks.push(real_block);
        }
        // NO MOCK DATA - if block doesn't exist, don't add anything

        if real_blocks.len() >= limit {
            break;
        }
    }

    Ok(Json(real_blocks))
}

/// Get REAL recent transactions - only actual transactions
pub async fn get_real_recent_transactions(
    Extension(state): Extension<Arc<RwLock<State>>>,
    Query(params): Query<RecentDataParams>,
) -> Result<Json<Vec<RealTransactionData>>, axum::http::StatusCode> {
    let state = state.read().await;

    let current_height = state.get_height().unwrap_or(0);
    let mut real_transactions = Vec::new();
    let limit = params.limit.min(50);

    // Only return actual transactions from real blocks
    let start_height = current_height.saturating_sub(5);

    for height in (start_height..=current_height).rev() {
        if let Some(block) = state.get_block_by_height(height) {
            for (position, tx) in block.transactions.iter().enumerate() {
                let tx_hash = tx.hash().unwrap_or_default();

                let real_tx = RealTransactionData {
                    hash: hex::encode(tx_hash.as_bytes()),
                    from: hex::encode(&tx.from),
                    to: hex::encode(&tx.to),
                    amount: tx.amount,
                    nonce: tx.nonce,
                    block_height: height,
                    position_in_block: position,
                };
                real_transactions.push(real_tx);

                if real_transactions.len() >= limit {
                    break;
                }
            }
        }

        if real_transactions.len() >= limit {
            break;
        }
    }

    Ok(Json(real_transactions))
}

/// Basic health check
pub async fn real_health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "version": "real-time-api-1.0.0",
        "mock_data": false,
        "real_time_only": true
    }))
}

/// Create router with REAL-TIME APIs only
pub fn create_real_time_router(state: Arc<RwLock<State>>) -> Router {
    Router::new()
        .route("/api/real/health", get(real_health_check))
        .route("/api/real/stats", get(get_real_stats))
        .route("/api/real/latest-block", get(get_real_latest_block))
        .route("/api/real/blocks/recent", get(get_real_recent_blocks))
        .route(
            "/api/real/transactions/recent",
            get(get_real_recent_transactions),
        )
        .layer(Extension(state))
        .layer(
            CorsLayer::new()
                .allow_methods(Any)
                .allow_headers(Any)
                .allow_origin(Any),
        )
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Load configuration
    let config = Config::default();

    // Initialize blockchain state
    let state = Arc::new(RwLock::new(State::new(&config)?));

    // Create the real-time API router
    let app = create_real_time_router(state);

    // Bind to port 8081 (different from main API on 8080)
    let addr = "0.0.0.0:8081";
    let listener = tokio::net::TcpListener::bind(addr).await?;

    println!("🔍 REAL-TIME API Server starting...");
    println!("📡 Listening on http://{}", addr);
    println!("🚫 ZERO mock data - real blockchain state only");
    println!("🌐 Real-time API Endpoints:");
    println!("   GET  /api/real/health           - Health check");
    println!("   GET  /api/real/stats            - Real blockchain statistics");
    println!("   GET  /api/real/latest-block     - Real latest block");
    println!("   GET  /api/real/blocks/recent    - Real recent blocks only");
    println!("   GET  /api/real/transactions/recent - Real transactions only");
    println!("🎯 Ready for pure real-time data connections!");

    // Start the server
    axum::serve(listener, app).await?;

    Ok(())
}
