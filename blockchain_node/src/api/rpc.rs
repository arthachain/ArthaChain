use crate::api::blockchain_api::BlockchainApi;
use crate::types::{Address, Hash, Transaction};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

/// JSON-RPC API handler
pub struct RpcHandler {
    /// Blockchain API instance
    blockchain_api: Arc<BlockchainApi>,
}

/// JSON-RPC request structure
#[derive(Debug, Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<Value>,
    pub id: Option<Value>,
}

/// JSON-RPC response structure
#[derive(Debug, Serialize)]
pub struct RpcResponse {
    pub jsonrpc: String,
    pub result: Option<Value>,
    pub error: Option<RpcError>,
    pub id: Option<Value>,
}

/// JSON-RPC error structure
#[derive(Debug, Serialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

impl RpcHandler {
    /// Create new RPC handler
    pub fn new(blockchain_api: Arc<BlockchainApi>) -> Self {
        Self { blockchain_api }
    }

    /// Handle JSON-RPC request
    pub async fn handle_request(&self, request: RpcRequest) -> RpcResponse {
        let result = match request.method.as_str() {
            "getBlockchainInfo" => self.handle_get_blockchain_info().await,
            "getBlockByHash" => self.handle_get_block_by_hash(request.params).await,
            "getBlockByHeight" => self.handle_get_block_by_height(request.params).await,
            "submitTransaction" => self.handle_submit_transaction(request.params).await,
            "getBalance" => self.handle_get_balance(request.params).await,
            "getNonce" => self.handle_get_nonce(request.params).await,
            "getPendingTransactions" => self.handle_get_pending_transactions(request.params).await,
            "getRecentBlocks" => self.handle_get_recent_blocks(request.params).await,
            _ => Err(anyhow::anyhow!("Method not found: {}", request.method)),
        };

        match result {
            Ok(value) => RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(value),
                error: None,
                id: request.id,
            },
            Err(err) => RpcResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(RpcError {
                    code: -32603,
                    message: err.to_string(),
                    data: None,
                }),
                id: request.id,
            },
        }
    }

    async fn handle_get_blockchain_info(&self) -> Result<Value> {
        let info = self.blockchain_api.get_blockchain_info().await?;
        Ok(serde_json::to_value(info)?)
    }

    async fn handle_get_block_by_hash(&self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| anyhow::anyhow!("Missing parameters"))?;
        let hash_str: String = serde_json::from_value(params)?;
        let hash = Hash::from_hex(&hash_str)?;
        let block = self.blockchain_api.get_block_by_hash(&hash).await?;
        Ok(serde_json::to_value(block)?)
    }

    async fn handle_get_block_by_height(&self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| anyhow::anyhow!("Missing parameters"))?;
        let height: u64 = serde_json::from_value(params)?;
        let block = self.blockchain_api.get_block_by_height(height).await?;
        Ok(serde_json::to_value(block)?)
    }

    async fn handle_submit_transaction(&self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| anyhow::anyhow!("Missing parameters"))?;
        let transaction: Transaction = serde_json::from_value(params)?;
        let tx_hash = self.blockchain_api.submit_transaction(transaction).await?;
        Ok(json!({ "transaction_hash": tx_hash }))
    }

    async fn handle_get_balance(&self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| anyhow::anyhow!("Missing parameters"))?;
        let address_str: String = serde_json::from_value(params)?;
        let address = Address::from_string(&address_str)
            .map_err(|e| anyhow::anyhow!("Invalid address: {}", e))?;
        let balance = self.blockchain_api.get_balance(&address).await?;
        Ok(json!({ "balance": balance }))
    }

    async fn handle_get_nonce(&self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| anyhow::anyhow!("Missing parameters"))?;
        let address_str: String = serde_json::from_value(params)?;
        let address = Address::from_string(&address_str)
            .map_err(|e| anyhow::anyhow!("Invalid address: {}", e))?;
        let nonce = self.blockchain_api.get_nonce(&address).await?;
        Ok(json!({ "nonce": nonce }))
    }

    async fn handle_get_pending_transactions(&self, params: Option<Value>) -> Result<Value> {
        let limit = if let Some(params) = params {
            serde_json::from_value(params)?
        } else {
            100 // Default limit
        };
        let transactions = self.blockchain_api.get_pending_transactions(limit).await?;
        Ok(serde_json::to_value(transactions)?)
    }

    async fn handle_get_recent_blocks(&self, params: Option<Value>) -> Result<Value> {
        let limit = if let Some(params) = params {
            serde_json::from_value(params)?
        } else {
            10 // Default limit
        };
        let blocks = self.blockchain_api.get_recent_blocks(limit).await?;
        Ok(serde_json::to_value(blocks)?)
    }
}
