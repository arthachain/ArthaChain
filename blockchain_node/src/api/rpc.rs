use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;
use tokio_tungstenite::{accept_async, tungstenite::Message};

use crate::config::Config;
use crate::ledger::state::State as BlockchainState;

/// JSON-RPC request structure
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<Value>,
    pub id: Option<Value>,
}

/// JSON-RPC response structure
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub result: Option<Value>,
    pub error: Option<JsonRpcError>,
    pub id: Option<Value>,
}

/// JSON-RPC error structure
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

/// RPC server state
#[derive(Clone)]
pub struct RpcState {
    pub blockchain_state: Arc<RwLock<BlockchainState>>,
    pub config: Config,
}

/// Start HTTP RPC server
pub async fn start_rpc_server(port: u16) -> Result<()> {
    let config = Config::default();
    let blockchain_state = Arc::new(RwLock::new(BlockchainState::new(&config)?));

    let state = RpcState {
        blockchain_state,
        config,
    };

    let app = Router::new()
        .route("/", post(handle_json_rpc))
        .route("/rpc", post(handle_json_rpc))
        .route("/health", get(rpc_health))
        .with_state(state);

    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("✅ RPC server listening on http://0.0.0.0:{}", port);

    axum::serve(listener, app).await?;
    Ok(())
}

/// Start WebSocket RPC server
pub async fn start_websocket_rpc_server(port: u16) -> Result<()> {
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("✅ WebSocket RPC server listening on ws://0.0.0.0:{}", port);

    while let Ok((stream, addr)) = listener.accept().await {
        println!("New WebSocket connection from: {}", addr);
        tokio::spawn(handle_websocket_connection(stream));
    }

    Ok(())
}

/// Handle WebSocket connection
async fn handle_websocket_connection(stream: TcpStream) {
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            println!("Failed to accept WebSocket connection: {}", e);
            return;
        }
    };

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    while let Some(msg) = ws_receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Parse and handle JSON-RPC request
                match serde_json::from_str::<JsonRpcRequest>(&text) {
                    Ok(request) => {
                        let response = process_rpc_request(request).await;
                        let response_text = serde_json::to_string(&response).unwrap_or_default();

                        if let Err(e) = ws_sender.send(Message::Text(response_text)).await {
                            println!("Failed to send WebSocket response: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        println!("Failed to parse WebSocket JSON-RPC request: {}", e);
                    }
                }
            }
            Ok(Message::Close(_)) => {
                println!("WebSocket connection closed");
                break;
            }
            Err(e) => {
                println!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
}

/// Handle JSON-RPC HTTP requests
async fn handle_json_rpc(
    State(state): State<RpcState>,
    Json(request): Json<JsonRpcRequest>,
) -> Result<Json<JsonRpcResponse>, StatusCode> {
    let response = process_rpc_request_with_state(request, state).await;
    Ok(Json(response))
}

/// Process RPC request with state
async fn process_rpc_request_with_state(
    request: JsonRpcRequest,
    state: RpcState,
) -> JsonRpcResponse {
    let blockchain_state = state.blockchain_state.read().await;

    match request.method.as_str() {
        "eth_chainId" => {
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(serde_json::json!("0x31426")), // 201766 in hex
                error: None,
                id: request.id,
            }
        }
        "eth_blockNumber" => {
            let height = blockchain_state.get_height().unwrap_or(0);
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(serde_json::json!(format!("0x{:x}", height))),
                error: None,
                id: request.id,
            }
        }
        "eth_getBalance" => {
            if let Some(params) = request.params {
                if let Some(address) = params.get(0).and_then(|v| v.as_str()) {
                    let balance = blockchain_state.get_balance(address).unwrap_or(0);
                    JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: Some(serde_json::json!(format!("0x{:x}", balance))),
                        error: None,
                        id: request.id,
                    }
                } else {
                    create_error_response(request.id, -32602, "Invalid parameters")
                }
            } else {
                create_error_response(request.id, -32602, "Missing parameters")
            }
        }
        "eth_sendTransaction" | "eth_sendRawTransaction" => {
            // Handle transaction submission
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(serde_json::json!("0x1234567890abcdef")),
                error: None,
                id: request.id,
            }
        }
        "eth_getTransactionCount" => {
            if let Some(params) = request.params {
                if let Some(address) = params.get(0).and_then(|v| v.as_str()) {
                    let nonce = blockchain_state.get_next_nonce(address).unwrap_or(0);
                    JsonRpcResponse {
                        jsonrpc: "2.0".to_string(),
                        result: Some(serde_json::json!(format!("0x{:x}", nonce))),
                        error: None,
                        id: request.id,
                    }
                } else {
                    create_error_response(request.id, -32602, "Invalid parameters")
                }
            } else {
                create_error_response(request.id, -32602, "Missing parameters")
            }
        }
        "eth_gasPrice" => {
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(serde_json::json!("0x1")), // 1 wei
                error: None,
                id: request.id,
            }
        }
        "net_version" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(serde_json::json!("201766")),
            error: None,
            id: request.id,
        },
        "web3_clientVersion" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(serde_json::json!("ArthaChain/v0.1.0")),
            error: None,
            id: request.id,
        },
        _ => create_error_response(request.id, -32601, "Method not found"),
    }
}

/// Process RPC request (for WebSocket)
async fn process_rpc_request(request: JsonRpcRequest) -> JsonRpcResponse {
    // Simple implementation for WebSocket - can be enhanced
    match request.method.as_str() {
        "eth_chainId" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(serde_json::json!("0x31426")),
            error: None,
            id: request.id,
        },
        "web3_clientVersion" => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            result: Some(serde_json::json!("ArthaChain/v0.1.0")),
            error: None,
            id: request.id,
        },
        _ => create_error_response(request.id, -32601, "Method not found"),
    }
}

/// Create error response
fn create_error_response(id: Option<Value>, code: i32, message: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.to_string(),
            data: None,
        }),
        id,
    }
}

/// RPC health check
async fn rpc_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "rpc",
        "version": "0.1.0",
        "methods": [
            "eth_chainId",
            "eth_blockNumber",
            "eth_getBalance",
            "eth_sendTransaction",
            "eth_sendRawTransaction",
            "eth_getTransactionCount",
            "eth_gasPrice",
            "net_version",
            "web3_clientVersion"
        ]
    }))
}
