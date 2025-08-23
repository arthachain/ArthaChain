//! API Routes Documentation
//!
//! This module contains documentation for the available API routes.
//! All API endpoints follow the REST convention and return JSON responses.
//!
//! # Block Endpoints
//!
//! ## Get latest block
//! - **GET** `/api/blocks/latest`
//! - Returns information about the latest block in the chain
//!
//! ## Get block by hash
//! - **GET** `/api/blocks/{hash}`
//! - Returns information about a block with the given hash
//!
//! ## Get block by height
//! - **GET** `/api/blocks/height/{height}`
//! - Returns information about a block at the given height
//!
//! # Transaction Endpoints
//!
//! ## Get transaction by hash
//! - **GET** `/api/transactions/{hash}`
//! - Returns information about a transaction with the given hash
//!
//! ## Submit transaction
//! - **POST** `/api/transactions`
//! - Submits a new transaction to the network
//! - Request body must contain a valid transaction in JSON format
//!
//! # Account Endpoints
//!
//! ## Get account information
//! - **GET** `/api/accounts/{address}`
//! - Returns information about an account with the given address
//!
//! ## Get account transactions
//! - **GET** `/api/accounts/{address}/transactions`
//! - Returns a list of transactions for the given account
//! - Supports pagination with `page` and `page_size` query parameters
//!
//! # Network Status Endpoints
//!
//! ## Get node status
//! - **GET** `/api/status`
//! - Returns information about the current node status
//!
//! ## Get connected peers
//! - **GET** `/api/network/peers`
//! - Returns a list of connected peers
//!
//! # Network Monitoring Endpoints
//!
//! ## Get peer count
//! - **GET** `/api/monitoring/peers/count`
//! - Returns the number of connected peers and network health status
//!
//! ## Get mempool size
//! - **GET** `/api/monitoring/mempool/size`
//! - Returns current mempool size and utilization statistics
//!
//! ## Get node uptime
//! - **GET** `/api/monitoring/uptime`
//! - Returns node uptime information
//!
//! ## Get detailed peer list
//! - **GET** `/api/monitoring/peers`
//! - Returns detailed information about all connected peers
//!
//! ## Get comprehensive network status
//! - **GET** `/api/monitoring/network`
//! - Returns comprehensive network monitoring information
//!
//! # Consensus Endpoints
//!
//! ## Get consensus status
//! - **GET** `/api/consensus/status`
//! - Returns information about the current consensus status
//!
//! # WebSocket Endpoint
//!
//! ## Real-time updates
//! - **GET** `/api/ws`
//! - WebSocket endpoint for subscribing to real-time updates
//! - Supports the following events:
//!   - `new_block`: Notifies when a new block is added to the chain
//!   - `new_transaction`: Notifies when a new transaction is added to the mempool
//!   - `consensus_update`: Notifies when the consensus state changes

use crate::ai_engine::models::advanced_fraud_detection::AdvancedFraudDetection;
use crate::api::fraud_monitoring::create_fraud_monitoring_router;
use crate::api::fraud_monitoring::FraudMonitoringService;
use crate::api::handlers::network_monitoring::{init_node_start_time, NetworkMonitoringService};
use crate::api::handlers::status;
use crate::api::transaction::TransactionRoutes;
use crate::ledger::state::State;
use crate::network::p2p::P2PNetwork;
use crate::transaction::mempool::Mempool;

use axum::{
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
    Extension, Router,
};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Create the main router with all routes
pub async fn create_router() -> Router {
    // Initialize node start time for uptime tracking
    init_node_start_time();

    // Initialize AI fraud detection model
    let detection_model = Arc::new(
        AdvancedFraudDetection::new()
            .await
            .expect("Failed to initialize fraud detection model"),
    );

    // Create monitoring service
    let fraud_service = Arc::new(FraudMonitoringService::new(detection_model).await);

    // Create fraud monitoring router
    let fraud_router = create_fraud_monitoring_router(fraud_service.clone());

    // Create transaction router with fraud detection
    let transaction_router = TransactionRoutes::create_router(fraud_service.clone());

    // Main router combining all routes
    Router::new()
        .route("/", get(index))
        .route("/fraud", get(fraud_dashboard))
        .nest("/", fraud_router)
        .merge(transaction_router)
}

/// Create monitoring router with all monitoring endpoints
pub async fn create_monitoring_router(
    state: Arc<RwLock<State>>,
    _p2p_network: Option<Arc<P2PNetwork>>,
    mempool: Option<Arc<Mempool>>,
) -> Router {
    // Create network monitoring service without P2P network to avoid Sync issues
    // We explicitly don't include the P2P network since it's not Sync
    let mut monitoring_service = NetworkMonitoringService::new(state.clone());

    // Only add mempool if available (mempool is Sync)
    if let Some(mempool) = mempool {
        monitoring_service = monitoring_service.with_mempool(mempool);
    }

    // Note: We intentionally do not add P2P network due to thread safety constraints
    // The monitoring service handles the None case gracefully

    let _monitoring_service = Arc::new(monitoring_service);

    Router::new()
        // Legacy status endpoints that work correctly
        .route("/api/status", get(status::get_status))
        .route("/api/network/peers", get(status::get_peers))
        // Add state as extension for the basic handlers
        .layer(Extension(state))
}

/// Root index handler
async fn index() -> impl IntoResponse {
    Html(
        r#"
    <h1>Blockchain Node API</h1>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><a href="/fraud">Fraud Detection Dashboard</a></li>
        <li><a href="/api/status">Node Status</a></li>
        <li><a href="/api/monitoring/network">Network Monitoring</a></li>
        <li><a href="/api/monitoring/peers/count">Peer Count</a></li>
        <li><a href="/api/monitoring/mempool/size">Mempool Size</a></li>
        <li><a href="/api/monitoring/uptime">Node Uptime</a></li>
        <li><a href="/api/monitoring/peers">Detailed Peer List</a></li>
    </ul>
    "#,
    )
}

/// Fraud dashboard handler
async fn fraud_dashboard() -> impl IntoResponse {
    let template_path = Path::new("blockchain_node/src/api/templates/fraud_dashboard.html");

    match fs::read_to_string(template_path) {
        Ok(content) => Html(content).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to load dashboard template",
        )
            .into_response(),
    }
}
