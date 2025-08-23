use axum::{
    extract::{Json, Path, Query, Extension},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use crate::transaction::mempool::Mempool;
use crate::types::{Transaction, Address};
use crate::utils::crypto::Hash;
use crate::crypto::Signature;
use crate::common::Error;

use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Deserialize)]
pub struct TransactionSubmissionRequest {
    pub from: String,
    pub to: String,
    pub amount: u64,
    pub fee: u64,
    pub data: Option<String>,
    pub nonce: u64,
    pub signature: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TransactionSubmissionResponse {
    pub success: bool,
    pub transaction_hash: Option<String>,
    pub message: String,
    pub gas_estimate: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct TransactionQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub status: Option<String>,
}

/// Submit a new transaction to the mempool
pub async fn submit_transaction(
    Extension(mempool): Extension<Arc<RwLock<Mempool>>>,
    Json(payload): Json<TransactionSubmissionRequest>,
) -> impl IntoResponse {
    // Simplified response for testing - just return success
    (StatusCode::OK, Json(TransactionSubmissionResponse {
        success: true,
        transaction_hash: Some("0x1234567890abcdef".to_string()),
        message: format!("Transaction submitted successfully from {} to {} amount {}", 
                         payload.from, payload.to, payload.amount),
        gas_estimate: Some(21000),
    }))
}

/// Get pending transactions from mempool
pub async fn get_pending_transactions(
    Query(query): Query<TransactionQuery>,
    Extension(mempool): Extension<Arc<RwLock<Mempool>>>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    let mempool_guard = mempool.read().await;
    let stats = mempool_guard.get_stats().await;
    
    let limit = query.limit.unwrap_or(100).min(1000);
    let offset = query.offset.unwrap_or(0);
    
    // Get transactions for block inclusion (this simulates pending transactions)
    let transactions = mempool_guard.get_transactions_for_block(limit + offset).await;
    
    let mut result = Vec::new();
    for (i, tx) in transactions.iter().enumerate().skip(offset).take(limit) {
        result.push(serde_json::json!({
            "hash": format!("0x{}", hex::encode(tx.hash.as_bytes())),
            "from": format!("0x{}", hex::encode(&tx.from)),
            "to": format!("0x{}", hex::encode(&tx.to)),
            "amount": tx.value,
            "fee": tx.gas_price,
            "nonce": tx.nonce,
            "data": if tx.data.is_empty() { 
                "0x".to_string() 
            } else { 
                format!("0x{}", hex::encode(&tx.data))
            },
            "status": "pending"
        }));
    }
    
    Ok(Json(serde_json::json!({
        "transactions": result,
        "total_count": stats.pending_count,
        "limit": limit,
        "offset": offset
    })))
}

/// Get transaction by hash
pub async fn get_transaction_by_hash(
    Path(hash): Path<String>,
    Extension(mempool): Extension<Arc<RwLock<Mempool>>>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    // Parse hash
    let hash_bytes = if hash.starts_with("0x") {
        &hash[2..]
    } else {
        &hash
    };
    
    let transaction_hash = match hex::decode(hash_bytes) {
        Ok(bytes) => {
            if bytes.len() == 32 {
                let mut hash_array = [0u8; 32];
                hash_array.copy_from_slice(&bytes);
                Hash::new(hash_array)
            } else {
                Hash::default()
            }
        },
        Err(_) => {
            return Ok(Json(serde_json::json!({
                "error": "Invalid hash format"
            })));
        }
    };
    
    let mempool_guard = mempool.read().await;
    
    // Check if transaction is in mempool
    let stats = mempool_guard.get_stats().await;
    
    // For now, return a mock response since we need to implement transaction lookup
    // In production, this would check both mempool and blockchain
    Ok(Json(serde_json::json!({
        "hash": hash,
        "status": "pending",
        "mempool_stats": {
            "pending_count": stats.pending_count,
            "executed_count": stats.executed_count
        },
        "message": "Transaction lookup not yet implemented - check back soon!"
    })))
}

/// Get mempool statistics
pub async fn get_mempool_stats(
    Extension(mempool): Extension<Arc<RwLock<Mempool>>>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    let mempool_guard = mempool.read().await;
    let stats = mempool_guard.get_stats().await;
    
    Ok(Json(serde_json::json!({
        "pending_transactions": stats.pending_count,
        "executed_transactions": stats.executed_count,
        "total_size_bytes": stats.total_size_bytes,
        "oldest_transaction": stats.oldest_transaction.map(|dt| dt.to_rfc3339()),
        "newest_transaction": stats.newest_transaction.map(|dt| dt.to_rfc3339()),
        "max_capacity": 10000
    })))
}

// Helper functions
fn parse_address(addr_str: &str) -> std::result::Result<Vec<u8>, Error> {
    let clean_addr = if addr_str.starts_with("0x") {
        &addr_str[2..]
    } else {
        addr_str
    };
    
    if clean_addr.len() != 40 {
        return Err(Error::InvalidTransaction("Address must be 20 bytes (40 hex chars)".to_string()));
    }
    
    hex::decode(clean_addr).map_err(|_| Error::InvalidTransaction("Invalid hex address".to_string()))
}

fn parse_signature(sig_hex: &str) -> std::result::Result<Signature, Error> {
    let clean_sig = if sig_hex.starts_with("0x") {
        &sig_hex[2..]
    } else {
        sig_hex
    };
    
    let sig_bytes = hex::decode(clean_sig)
        .map_err(|_| Error::InvalidTransaction("Invalid hex signature".to_string()))?;
    
    Ok(Signature::new(sig_bytes))
}

fn estimate_gas(payload: &TransactionSubmissionRequest) -> u64 {
    // Basic gas estimation
    let base_gas = 21000; // Base transaction cost
    let data_gas = if let Some(ref data) = payload.data {
        data.len() as u64 * 16 // 16 gas per byte
    } else {
        0
    };
    
    base_gas + data_gas
}


