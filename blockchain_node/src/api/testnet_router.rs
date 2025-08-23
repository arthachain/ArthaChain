use anyhow::Result;
use axum::{
    extract::{Extension, Path, Query},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

use crate::api::{
    create_fraud_monitoring_router, fraud_monitoring::FraudMonitoringService,
    handlers::{
        accounts, blocks, consensus, faucet, gas_free, metrics, network_monitoring, status,
        transactions, transaction_submission, validators,
    },
    routes::create_monitoring_router,
    wallet_integration,
    websocket::{websocket_handler, EventManager},
};
use crate::consensus::validator_set::ValidatorSetManager;
use crate::ledger::state::State;
use crate::transaction::mempool::Mempool;

/// Testnet-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestnetRouterConfig {
    pub enable_faucet: bool,
    pub enable_testnet_features: bool,
    pub max_transactions_per_block: u32,
    pub block_time: u64,
    pub chain_id: u64,
}

impl Default for TestnetRouterConfig {
    fn default() -> Self {
        Self {
            enable_faucet: true,
            enable_testnet_features: true,
            max_transactions_per_block: 1000,
            block_time: 5,
            chain_id: 201766, // ArthaChain testnet
        }
    }
}

/// Create the main testnet router with all routes
pub async fn create_testnet_router(
    state: Arc<RwLock<State>>,
    validator_manager: Arc<ValidatorSetManager>,
    mempool: Arc<RwLock<Mempool>>,
) -> Router {
    // Create a simple router without complex AI dependencies for now
    // TODO: Re-enable fraud detection once AI models are properly configured
    
    // Create monitoring router
    let monitoring_router = create_monitoring_router(
        state.clone(), 
        None, 
        None  // TODO: Fix mempool integration when monitoring router is updated
    ).await;

    // Create WebSocket event manager for real-time updates
    let event_manager = Arc::new(EventManager::new());

    // Create faucet service
    let faucet_service = match faucet::Faucet::new(&crate::config::Config::default(), state.clone(), None).await {
        Ok(faucet) => Arc::new(faucet),
        Err(e) => {
            eprintln!("⚠️ Warning: Could not create faucet service: {}", e);
            // Create a dummy faucet for now
            Arc::new(faucet::Faucet::new_dummy().await)
        }
    };

    // Create gas-free manager
    let gas_free_manager = Arc::new(crate::gas_free::GasFreeManager::new());
    
    // Initialize demo gas-free applications
    if let Err(e) = gas_free_manager.create_demo_apps().await {
        eprintln!("⚠️ Warning: Could not create demo gas-free apps: {}", e);
    }

    // Main testnet router combining all routes
    Router::new()
        .route("/", get(testnet_index))
        .route("/health", get(health_check))
        .route("/status", get(testnet_status))
        .route("/config", get(get_testnet_config))
        .route("/faucet", get(faucet::faucet_dashboard))
        .route("/wallet", get(wallet_integration::wallet_connect_page))
        .route("/ide", get(wallet_integration::ide_setup_page))
        .route("/docs", get(api_documentation))
        .route("/metrics", get(metrics::get_metrics))
        .route("/api/v1/network/stats", get(network_stats))
        .route("/api/v1/network/peers", get(network_peers))
        .route("/api/v1/network/status", get(network_status))
        .route("/api/v1/consensus/validators", get(validators::get_validators_list))
        .route("/api/v1/consensus/status", get(consensus::get_consensus_status))
        .route("/api/v1/blocks/latest", get(blocks::get_latest_block))
        .route("/api/v1/blocks/:hash", get(blocks::get_block_by_hash))
        .route("/api/v1/blocks/height/:height", get(blocks::get_block_by_height))
        .route("/api/v1/blocks", get(blocks::get_blocks))
        .route("/api/v1/transactions/:hash", get(transactions::get_transaction))
        .route("/api/v1/transactions", post(transaction_submission::submit_transaction))
        .route("/api/v1/accounts/:address", get(accounts::get_account))
        .route("/api/v1/accounts/:address/transactions", get(accounts::get_account_transactions))
        .route("/api/v1/accounts/:address/balance", get(accounts::get_account_balance))
        // Contract endpoints - to be implemented
        // .route("/api/v1/contracts/:address", get(contracts::get_contract))
        // .route("/api/v1/contracts/:address/code", get(contracts::get_contract_code))
        // .route("/api/v1/contracts/:address/storage", get(contracts::get_contract_storage))
        // Identity endpoints - to be implemented
        // .route("/api/v1/identity/:address", get(identity::get_identity))
        // Security endpoints - to be implemented
        // .route("/api/v1/security/alerts", get(security::get_security_alerts))
        // .route("/api/v1/security/status", get(security::get_security_status))
        .route("/api/v1/testnet/faucet/request", post(faucet::request_tokens))
        .route("/api/v1/testnet/faucet/status", get(faucet::get_faucet_status))
        .route("/api/v1/testnet/faucet/history", get(faucet::get_faucet_history))
        // Gas-free application endpoints
        .route("/gas-free", get(gas_free::gas_free_dashboard))
        .route("/api/v1/testnet/gas-free/register", post(gas_free::register_gas_free_app))
        .route("/api/v1/testnet/gas-free/check", post(gas_free::check_gas_free_eligibility))
        .route("/api/v1/testnet/gas-free/apps", get(gas_free::get_active_gas_free_apps))
        .route("/api/v1/testnet/gas-free/stats", get(gas_free::get_gas_free_stats))
        .route("/api/v1/testnet/gas-free/process", post(gas_free::process_gas_free_transaction))
        .route("/api/v1/wallet/supported", get(wallet_integration::get_supported_wallets))
        .route("/api/v1/wallet/ides", get(wallet_integration::get_supported_ides))
        .route("/api/v1/wallet/connect", get(wallet_integration::wallet_connect_page))
        .route("/api/v1/wallet/setup", get(wallet_integration::ide_setup_page))
        .route("/api/v1/rpc", post(handle_rpc_request))
        .route("/api/v1/ws", get(websocket_handler))
        // .nest("/fraud", fraud_router)  // TODO: Re-enable when fraud detection is ready
        // .merge(transaction_router)      // TODO: Re-enable when transaction router is ready
        .merge(monitoring_router)
        .layer(CorsLayer::permissive())
        .layer(Extension(state))
        .layer(Extension(validator_manager))
        .layer(Extension(mempool))
        .layer(Extension(event_manager))
        .layer(Extension(faucet_service))
        .layer(Extension(gas_free_manager))
}

/// Testnet index page
async fn testnet_index() -> impl IntoResponse {
    Html(
        r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>ArthaChain Testnet API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .section { margin: 30px 0; padding: 20px; border: 1px solid #ecf0f1; border-radius: 8px; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .method { display: inline-block; background: #3498db; color: white; padding: 5px 10px; border-radius: 3px; font-size: 12px; font-weight: bold; }
            .url { font-family: monospace; color: #2c3e50; }
            .description { color: #7f8c8d; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 ArthaChain Testnet API</h1>
            <p style="text-align: center; color: #7f8c8d;">Next-generation blockchain with AI-native features, quantum resistance, and ultra-high performance</p>
            
            <div class="section">
                <h2>🔗 Quick Links</h2>
                <div class="endpoint">
                    <a href="/health">Health Check</a> - <span class="description">API server status</span>
                </div>
                <div class="endpoint">
                    <a href="/faucet">Faucet</a> - <span class="description">Get testnet tokens</span>
                </div>
                <div class="endpoint">
                    <a href="/gas-free">Gas-Free Apps</a> - <span class="description">Enterprise gas-free applications</span>
                </div>
                <div class="endpoint">
                    <a href="/wallet">Wallet Integration</a> - <span class="description">Connect your wallet</span>
                </div>
                <div class="endpoint">
                    <a href="/docs">API Documentation</a> - <span class="description">Complete API reference</span>
                </div>
            </div>

            <div class="section">
                <h2>📡 Core Endpoints</h2>
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/api/v1/blocks/latest</span>
                    <div class="description">Get the latest block information</div>
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/api/v1/transactions/:hash</span>
                    <div class="description">Get transaction details by hash</div>
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <span class="url">/api/v1/transactions</span>
                    <div class="description">Submit a new transaction</div>
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/api/v1/accounts/:address</span>
                    <div class="description">Get account information</div>
                </div>
            </div>

            <div class="section">
                <h2>🔧 Testnet Features</h2>
                <div class="endpoint">
                    <span class="method">POST</span> <span class="url">/api/v1/testnet/faucet/request</span>
                    <div class="description">Request testnet tokens (no staking required!)</div>
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/consensus/validators</span>
                    <div class="description">View active validators</div>
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/api/v1/network/stats</span>
                    <div class="description">Network statistics and health</div>
                </div>
            </div>

            <div class="section">
                <h2>🚀 Gas-Free Applications</h2>
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/gas-free</span>
                    <div class="description">Gas-free application dashboard (company access only)</div>
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <span class="url">/api/v1/testnet/gas-free/register</span>
                    <div class="description">Register new gas-free application (company only)</div>
                </div>
                <div class="endpoint">
                    <span class="method">POST</span> <span class="url">/api/v1/testnet/gas-free/check</span>
                    <div class="description">Check transaction eligibility for gas-free processing</div>
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/api/v1/testnet/gas-free/apps</span>
                    <div class="description">List all active gas-free applications</div>
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/api/v1/testnet/gas-free/stats</span>
                    <div class="description">Gas-free application statistics</div>
                </div>
            </div>

            <div class="section">
                <h2>📊 Network Information</h2>
                <div class="endpoint">
                    <strong>Chain ID:</strong> 201766 (ArthaChain Testnet)
                </div>
                <div class="endpoint">
                    <strong>Currency:</strong> ARTHA (18 decimals)
                </div>
                <div class="endpoint">
                    <strong>Block Time:</strong> 5 seconds
                </div>
                <div class="endpoint">
                    <strong>Target TPS:</strong> 100,000+
                </div>
            </div>

            <div class="section">
                <h2>🛡️ Security Features</h2>
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/api/v1/security/status</span>
                    <div class="description">Security monitoring and alerts</div>
                </div>
                <div class="endpoint">
                    <span class="method">GET</span> <span class="url">/fraud</span>
                    <div class="description">AI-powered fraud detection dashboard</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    "#,
    )
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "service": "ArthaChain Testnet API",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime": "running"
    }))
}

/// Testnet status endpoint
async fn testnet_status() -> impl IntoResponse {
    Json(serde_json::json!({
        "network": "ArthaChain Testnet",
        "chain_id": 201766,
        "status": "active",
        "latest_block": "pending",
        "total_validators": 1,
        "consensus_mechanism": "SVBFT",
        "features": [
            "AI-powered fraud detection",
            "Quantum resistance",
            "Cross-shard transactions",
            "WASM smart contracts",
            "EVM compatibility"
        ]
    }))
}

/// Get testnet configuration
async fn get_testnet_config() -> impl IntoResponse {
    let config = TestnetRouterConfig::default();
    Json(config)
}

/// Network statistics endpoint
async fn network_stats() -> impl IntoResponse {
    Json(serde_json::json!({
        "total_nodes": 1,
        "active_validators": 1,
        "total_transactions": 0,
        "pending_transactions": 0,
        "network_health": 1.0,
        "consensus_participation": 1.0,
        "block_production_rate": "5s",
        "last_block_time": chrono::Utc::now().to_rfc3339()
    }))
}

/// Network peers endpoint
async fn network_peers() -> impl IntoResponse {
    Json(serde_json::json!({
        "connected_peers": 0,
        "total_peers": 0,
        "peer_list": [],
        "network_topology": "single_node"
    }))
}

/// Network status endpoint
async fn network_status() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "active",
        "sync_status": "synced",
        "network_id": "arthachain_testnet_201766",
        "protocol_version": "1.0.0",
        "capabilities": [
            "p2p_networking",
            "consensus_participation",
            "transaction_processing",
            "smart_contract_execution"
        ]
    }))
}

/// API documentation endpoint
async fn api_documentation() -> impl IntoResponse {
    Html(
        r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>ArthaChain API Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1, h2 { color: #2c3e50; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .method { display: inline-block; background: #3498db; color: white; padding: 5px 10px; border-radius: 3px; font-size: 12px; font-weight: bold; }
            .url { font-family: monospace; color: #2c3e50; }
            .description { color: #7f8c8d; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 ArthaChain API Documentation</h1>
            <p>Complete API reference for ArthaChain testnet</p>
            
            <h2>🔗 Core Blockchain API</h2>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/blocks/latest</span>
                <div class="description">Get the latest block information</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/blocks/:hash</span>
                <div class="description">Get block by hash</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/blocks/height/:height</span>
                <div class="description">Get block by height</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/transactions/:hash</span>
                <div class="description">Get transaction by hash</div>
            </div>
            <div class="endpoint">
                <span class="method">POST</span> <span class="url">/api/v1/transactions</span>
                <div class="description">Submit new transaction</div>
            </div>
            
            <h2>👤 Account Management</h2>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/accounts/:address</span>
                <div class="description">Get account information</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/accounts/:address/balance</span>
                <div class="description">Get account balance</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/accounts/:address/transactions</span>
                <div class="description">Get account transaction history</div>
            </div>
            
            <h2>🔧 Testnet Features</h2>
            <div class="endpoint">
                <span class="method">POST</span> <span class="url">/api/v1/testnet/faucet/request</span>
                <div class="description">Request testnet tokens</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/testnet/faucet/status</span>
                <div class="description">Check faucet status</div>
            </div>
            
            <h2>🌐 Network & Consensus</h2>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/network/stats</span>
                <div class="description">Network statistics</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/consensus/validators</span>
                <div class="description">Active validators</div>
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/api/v1/consensus/status</span>
                <div class="description">Consensus status</div>
            </div>
        </div>
    </body>
    </html>
    "#,
    )
}

/// Handle RPC requests
async fn handle_rpc_request() -> impl IntoResponse {
    Json(serde_json::json!({
        "jsonrpc": "2.0",
        "error": {
            "code": -32601,
            "message": "Method not found"
        },
        "id": null
    }))
}

// WebSocket handler is now imported from websocket module

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    #[tokio::test]
    async fn test_health_check() {
        let response = health_check().await;
        assert!(response.into_response().status().is_success());
    }

    #[tokio::test]
    async fn test_testnet_status() {
        let response = testnet_status().await;
        assert!(response.into_response().status().is_success());
    }

    #[tokio::test]
    async fn test_get_testnet_config() {
        let response = get_testnet_config().await;
        assert!(response.into_response().status().is_success());
    }

    #[test]
    fn test_testnet_router_config_default() {
        let config = TestnetRouterConfig::default();
        assert_eq!(config.chain_id, 201766);
        assert_eq!(config.block_time, 5);
        assert!(config.enable_faucet);
        assert!(config.enable_testnet_features);
    }
}
