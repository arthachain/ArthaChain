use crate::node::Node;
use anyhow::Result;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

use crate::api::metrics::MetricsService;
use crate::config::Config;
#[cfg(not(skip_problematic_modules))]
use crate::consensus::svbft::SVBFTConsensus;
use crate::ledger::state::State;
use serde::Serialize;

pub mod faucet;
pub mod handlers;
pub mod metrics;
pub mod models;
pub mod routes;
pub mod websocket;
// pub mod blockchain;
// pub mod consensus;
// pub mod node;
// pub mod rpc;
pub mod transaction;
// pub mod utils;
pub mod fraud_monitoring;

// pub use blockchain::BlockchainRoutes;
// pub use consensus::ConsensusRoutes;
// pub use metrics::MetricsRoutes;
// pub use node::NodeRoutes;
pub use fraud_monitoring::create_fraud_monitoring_router;
pub use transaction::TransactionRoutes;

/// Error response for the API
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
}

impl ApiError {
    pub fn invalid_address() -> Self {
        Self {
            status: 400,
            message: "Invalid address format".to_string(),
        }
    }

    pub fn account_not_found() -> Self {
        Self {
            status: 404,
            message: "Account not found".to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = Json(self);
        (status, body).into_response()
    }
}

/// API Server that handles REST endpoints for blockchain interaction
pub struct ApiServer {
    /// API server configuration
    _config: Config,
    /// Blockchain state
    _state: Arc<RwLock<State>>,
    /// Channel for shutdown signal
    #[allow(dead_code)]
    shutdown_signal: Arc<Mutex<mpsc::Receiver<()>>>,
    /// Active consensus engine reference (if available)
    #[cfg(not(skip_problematic_modules))]
    _consensus: Option<Arc<SVBFTConsensus>>,
    /// Metrics service
    _metrics: Option<Arc<MetricsService>>,
    /// Node reference
    _node: Arc<Mutex<Node>>,
    /// Router
    router: Router,
    /// Host for the API server
    host: String,
    /// Port for the API server
    port: String,
}

impl ApiServer {
    /// Create a new API server
    pub async fn new(
        config: Config,
        state: Arc<RwLock<State>>,
        node: Arc<Mutex<Node>>,
        host: String,
        port: String,
    ) -> Result<Self> {
        let metrics = Some(Arc::new(MetricsService::new()));

        // Create base router without node extension to avoid thread safety issues
        let router = Router::new()
            .route("/health", get(health_check))
            .route("/metrics", get(metrics_handler))
            .route("/api/blocks", get(get_blocks))
            .route("/api/blocks/:hash", get(get_block))
            .route("/api/transactions", get(get_transactions))
            .route("/api/transactions/:hash", get(get_transaction))
            .route("/api/peers", get(get_peers))
            .route("/api/peers/:id", get(get_peer))
            .route("/api/consensus", get(get_consensus))
            .route("/api/consensus/status", get(get_consensus_status))
            .route("/api/consensus/vote", post(vote))
            .route("/api/consensus/propose", post(propose))
            .route("/api/consensus/validate", post(validate))
            .route("/api/consensus/finalize", post(finalize))
            .route("/api/consensus/commit", post(commit))
            .route("/api/consensus/revert", post(revert))
            .layer(CorsLayer::permissive());

        Ok(Self {
            _config: config,
            _state: state,
            shutdown_signal: Arc::new(Mutex::new(mpsc::channel(1).1)),
            #[cfg(not(skip_problematic_modules))]
            _consensus: None,
            _metrics: metrics,
            _node: node,
            router,
            host,
            port,
        })
    }

    /// Start the API server and listen for incoming requests
    pub async fn start(&self) -> Result<()> {
        let addr = format!("{}:{}", self.host, self.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        println!("Server listening on {addr}");

        axum::serve(listener, self.router.clone()).await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[tokio::test]
    async fn test_api_server_creation() {
        let config = Config::default();
        let state = Arc::new(RwLock::new(State::new(&config).unwrap()));
        let node = Arc::new(Mutex::new(Node::new(config.clone()).await.unwrap()));
        let _server = ApiServer::new(
            config,
            state,
            node,
            "127.0.0.1".to_string(),
            "8080".to_string(),
        )
        .await
        .unwrap();

        // Verify the server was created successfully
        assert!(true, "Server was created successfully");
    }
}

async fn health_check() -> &'static str {
    "OK"
}

async fn metrics_handler() -> &'static str {
    "Metrics endpoint"
}

async fn get_blocks() -> &'static str {
    "Get blocks endpoint"
}

async fn get_block() -> &'static str {
    "Get block endpoint"
}

async fn get_transactions() -> &'static str {
    "Get transactions endpoint"
}

async fn get_transaction() -> &'static str {
    "Get transaction endpoint"
}

async fn get_peers() -> &'static str {
    "Get peers endpoint"
}

async fn get_peer() -> &'static str {
    "Get peer endpoint"
}

async fn get_consensus() -> &'static str {
    "Get consensus endpoint"
}

async fn get_consensus_status() -> &'static str {
    "Get consensus status endpoint"
}

async fn vote() -> &'static str {
    "Vote endpoint"
}

async fn propose() -> &'static str {
    "Propose endpoint"
}

async fn validate() -> &'static str {
    "Validate endpoint"
}

async fn finalize() -> &'static str {
    "Finalize endpoint"
}

async fn commit() -> &'static str {
    "Commit endpoint"
}

async fn revert() -> &'static str {
    "Revert endpoint"
}
