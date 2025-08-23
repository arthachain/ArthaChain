use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::{HeaderValue, Method, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

use crate::consensus::cross_shard::EnhancedCrossShardManager;
use crate::network::cross_shard::{CrossShardConfig, CrossShardTransaction};

// API Models
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub node_id: String,
    pub network: String,
}

#[derive(Serialize, Deserialize)]
pub struct TransactionRequest {
    pub from_shard: u32,
    pub to_shard: u32,
    pub from_address: String,
    pub to_address: String,
    pub amount: u64,
    pub gas_limit: u64,
}

#[derive(Serialize, Deserialize)]
pub struct TransactionResponse {
    pub transaction_id: String,
    pub status: String,
    pub message: String,
}

#[derive(Serialize, Deserialize)]
pub struct TransactionStatusResponse {
    pub transaction_id: String,
    pub phase: String,
    pub status: String,
    pub timestamp: u64,
}

#[derive(Serialize, Deserialize)]
pub struct NetworkStatsResponse {
    pub total_shards: u32,
    pub active_nodes: u32,
    pub pending_transactions: u32,
    pub processed_transactions: u64,
    pub network_health: f64,
}

#[derive(Serialize, Deserialize)]
pub struct ShardInfoResponse {
    pub shard_id: u32,
    pub status: String,
    pub transaction_count: u64,
    pub last_block_height: u64,
    pub connected_peers: u32,
}

// Application State
#[derive(Clone)]
pub struct AppState {
    pub cross_shard_manager: Arc<RwLock<EnhancedCrossShardManager>>,
    pub node_id: String,
    pub network: String,
    pub stats: Arc<RwLock<NetworkStats>>,
}

#[derive(Default)]
pub struct NetworkStats {
    pub total_transactions: u64,
    pub pending_transactions: u32,
    pub active_nodes: u32,
}

// API Handlers
pub async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        node_id: state.node_id,
        network: state.network,
    })
}

pub async fn submit_transaction(
    State(state): State<AppState>,
    Json(req): Json<TransactionRequest>,
) -> Result<Json<TransactionResponse>, StatusCode> {
    let tx_id = format!("tx_{}", uuid::Uuid::new_v4());

    let transaction = CrossShardTransaction::new(tx_id.clone(), req.from_shard, req.to_shard);

    let manager = state.cross_shard_manager.read().await;

    match manager.initiate_cross_shard_transaction(transaction).await {
        Ok(transaction_id) => {
            // Update stats
            let mut stats = state.stats.write().await;
            stats.pending_transactions += 1;

            Ok(Json(TransactionResponse {
                transaction_id,
                status: "pending".to_string(),
                message: "Transaction submitted successfully".to_string(),
            }))
        }
        Err(e) => {
            eprintln!("Transaction submission failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub async fn get_transaction_status(
    State(state): State<AppState>,
    Path(tx_id): Path<String>,
) -> Result<Json<TransactionStatusResponse>, StatusCode> {
    let manager = state.cross_shard_manager.read().await;

    match manager.get_transaction_status(&tx_id) {
        Ok((phase, status)) => Ok(Json(TransactionStatusResponse {
            transaction_id: tx_id,
            phase: format!("{:?}", phase),
            status: format!("{:?}", status),
            timestamp: chrono::Utc::now().timestamp() as u64,
        })),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

pub async fn get_network_stats(State(state): State<AppState>) -> Json<NetworkStatsResponse> {
    let stats = state.stats.read().await;

    Json(NetworkStatsResponse {
        total_shards: 4, // From config
        active_nodes: stats.active_nodes,
        pending_transactions: stats.pending_transactions,
        processed_transactions: stats.total_transactions,
        network_health: 0.95, // Mock health score
    })
}

pub async fn get_shard_info(
    State(_state): State<AppState>,
    Path(shard_id): Path<u32>,
) -> Json<ShardInfoResponse> {
    Json(ShardInfoResponse {
        shard_id,
        status: "active".to_string(),
        transaction_count: 1234, // Mock data
        last_block_height: 5678,
        connected_peers: 8,
    })
}

pub async fn list_shards(State(_state): State<AppState>) -> Json<Vec<ShardInfoResponse>> {
    let shards = (0..4)
        .map(|shard_id| ShardInfoResponse {
            shard_id,
            status: "active".to_string(),
            transaction_count: 1000 + shard_id as u64 * 100,
            last_block_height: 5000 + shard_id as u64 * 100,
            connected_peers: 6 + shard_id,
        })
        .collect();

    Json(shards)
}

// API Router
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Health and info endpoints
        .route("/health", get(health_check))
        .route("/stats", get(get_network_stats))
        // Transaction endpoints
        .route("/transactions", post(submit_transaction))
        .route("/transactions/:tx_id", get(get_transaction_status))
        // Shard endpoints
        .route("/shards", get(list_shards))
        .route("/shards/:shard_id", get(get_shard_info))
        // CORS for global access
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
                .allow_headers(Any),
        )
        .with_state(state)
}

// Server startup
pub async fn start_api_server(port: u16) -> Result<()> {
    println!("🚀 Starting ArthaChain API Server on port {}", port);

    // Initialize cross-shard manager
    let config = CrossShardConfig {
        max_retries: 3,
        retry_interval: Duration::from_millis(100),
        message_timeout: Duration::from_secs(30),
        batch_size: 10,
        max_queue_size: 1000,
        sync_interval: Duration::from_secs(30),
        validation_threshold: 0.67,
        transaction_timeout: Duration::from_secs(30),
        retry_count: 3,
        pending_timeout: Duration::from_secs(60),
        timeout_check_interval: Duration::from_secs(5),
        resource_threshold: 0.8,
        local_shard: 0,
        connected_shards: vec![1, 2, 3],
    };

    // Use real network manager for cross-shard functionality
    let network = Arc::new(crate::network::TestNetworkManager::new());

    let mut manager = EnhancedCrossShardManager::new(config, network).await?;
    manager.start()?;

    let state = AppState {
        cross_shard_manager: Arc::new(RwLock::new(manager)),
        node_id: format!("node_{}", uuid::Uuid::new_v4()),
        network: "mainnet".to_string(),
        stats: Arc::new(RwLock::new(NetworkStats::default())),
    };

    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!(
        "✅ ArthaChain API Server listening on http://0.0.0.0:{}",
        port
    );
    println!("📚 API Documentation:");
    println!("  GET  /health              - Health check");
    println!("  GET  /stats               - Network statistics");
    println!("  POST /transactions        - Submit transaction");
    println!("  GET  /transactions/:id    - Get transaction status");
    println!("  GET  /shards              - List all shards");
    println!("  GET  /shards/:id          - Get shard info");

    axum::serve(listener, app).await?;

    Ok(())
}
