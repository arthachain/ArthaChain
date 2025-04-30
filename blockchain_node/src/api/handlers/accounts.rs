use std::sync::Arc;
use axum::{
    extract::{Path, Extension, Query},
    Json,
};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use crate::ledger::state::State;
#[cfg(feature = "evm")]
use crate::evm::backend::EvmBackend;
use crate::api::{ApiError, handlers::transactions::TransactionResponse};

/// Response for an account
#[derive(Serialize)]
pub struct AccountResponse {
    /// Account balance
    pub balance: String,
    /// Account nonce
    pub nonce: u64,
    /// Account has code (smart contract)
    pub code: Option<String>,
    /// Storage entries count
    pub storage_entries: Option<u64>,
}

/// Query parameters for transaction list
#[derive(Deserialize)]
pub struct TransactionListParams {
    /// Page number (0-based)
    #[serde(default)]
    pub page: usize,
    /// Items per page
    #[serde(default = "default_page_size")]
    pub page_size: usize,
}

fn default_page_size() -> usize {
    20
}

/// Response for a list of transactions
#[derive(Serialize)]
pub struct TransactionListResponse {
    /// Transactions
    pub transactions: Vec<TransactionResponse>,
    /// Total count
    pub total: usize,
    /// Page number
    pub page: usize,
    /// Page size
    pub page_size: usize,
}

/// Get account information
pub async fn get_account(
    Path(address): Path<String>,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<AccountResponse>, ApiError> {
    // Check if it's an EVM address (0x prefix and 40 hex chars)
    if address.starts_with("0x") && address.len() == 42 {
        #[cfg(feature = "evm")]
        {
            // Handle EVM account
            let address = H160::from_str(&address[2..])
                .map_err(|_| ApiError::INVALID_ADDRESS)?;
            
            let state = state.read().await;
            let backend = EvmBackend::new(&state);
            
            let basic = backend.basic(address);
            let code = backend.code(address);
            let code_hex = if !code.is_empty() {
                Some(hex::encode(&code))
            } else {
                None
            };
            
            // Count storage entries
            let mut storage_count = 0;
            for key in backend.storage_keys(address) {
                if backend.storage(address, key) != H256::zero() {
                    storage_count += 1;
                }
            }

            Ok(Json(AccountResponse {
                balance: basic.balance.to_string(),
                nonce: basic.nonce.as_u64(),
                code: code_hex,
                storage_entries: Some(storage_count),
            }))
        }

        #[cfg(not(feature = "evm"))]
        Err(ApiError {
            status: 400,
            message: "EVM support is not enabled".to_string(),
        })
    } else {
        // Handle native account
        let state = state.read().await;
        let account = state.get_account(&address)
            .ok_or_else(ApiError::account_not_found)?;

        Ok(Json(AccountResponse {
            balance: account.balance.to_string(),
            nonce: account.nonce,
            code: None,
            storage_entries: None,
        }))
    }
}

/// Get transactions for an account
pub async fn get_account_transactions(
    Path(address): Path<String>,
    Query(params): Query<TransactionListParams>,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<TransactionListResponse>, ApiError> {
    let state = state.read().await;
    
    // Get transactions for this account
    let transactions = state.get_account_transactions(&address);
    
    // Apply pagination
    let total = transactions.len();
    let start = params.page * params.page_size;
    let end = (start + params.page_size).min(total);
    
    let transactions = if start < total {
        transactions[start..end].iter()
            .map(|tx| {
                // Convert types::Transaction to ledger::transaction::Transaction
                let ledger_tx: crate::ledger::transaction::Transaction = tx.clone().into();
                // For now, transactions are not yet in blocks, so no confirmations
                TransactionResponse::from_tx(&ledger_tx, None, None, 0)
            })
            .collect()
    } else {
        Vec::new()
    };
    
    Ok(Json(TransactionListResponse {
        transactions,
        total,
        page: params.page,
        page_size: params.page_size,
    }))
} 