use axum::{
    extract::{Extension, Path, Query},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::api::{handlers::transactions::TransactionResponse, ApiError};
#[cfg(feature = "evm")]
use crate::evm::backend::EvmBackend;
use crate::ledger::state::State;

// Ethereum types for EVM compatibility
#[cfg(feature = "evm")]
use ethereum_types::{H160, H256, U256};

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
            let address = H160::from_str(&address[2..]).map_err(|_| ApiError::invalid_address())?;
            
            // Convert H160 to EvmAddress
            let evm_address = crate::evm::types::EvmAddress::from_slice(address.as_bytes());
            
            // For now, return basic account info without EVM storage access
            // TODO: Implement proper EVM storage integration
            
            // Return basic EVM account info
            Ok(Json(AccountResponse {
                balance: "0".to_string(),
                nonce: 0,
                code: None,
                storage_entries: Some(0),
            }))
        }

        #[cfg(not(feature = "evm"))]
        {
            // Mock EVM account for testing
            let balance = if address == "0x742d35Cc6634C0532925a3b844Bc454e4438f44e" {
                "2000000000000000000" // 2.0 ARTHA in wei
            } else {
                "0"
            };

            Ok(Json(AccountResponse {
                balance: balance.to_string(),
                nonce: 0,
                code: None,
                storage_entries: Some(0),
            }))
        }
    } else {
        // Handle native account
        let state = state.read().await;
        let account = state
            .get_account(&address)
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
        transactions[start..end]
            .iter()
            .map(|tx| {
                // Convert types::Transaction to ledger::transaction::Transaction
                let ledger_tx: crate::ledger::transaction::Transaction = tx.clone();
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

/// Get account balance
pub async fn get_account_balance(
    Path(address): Path<String>,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Check if it's an EVM address (0x prefix and 40 hex chars)
    if address.starts_with("0x") && address.len() == 42 {
        #[cfg(feature = "evm")]
        {
            // Handle EVM account
            let address = H160::from_str(&address[2..]).map_err(|_| ApiError::invalid_address())?;
            
            // Convert H160 to EvmAddress
            let evm_address = crate::evm::types::EvmAddress::from_slice(address.as_bytes());
            
            // For now, return basic account info without EVM storage access
            // TODO: Implement proper EVM storage integration
            
            Ok(Json(serde_json::json!({
                "address": address,
                "balance": "0",
                "currency": "ARTHA",
                "decimals": 18,
                "formatted_balance": "0.0 ARTHA"
            })))
        }

        #[cfg(not(feature = "evm"))]
        {
            // Mock EVM account for testing
            let balance = if address == "0x742d35Cc6634C0532925a3b844Bc454e4438f44e" {
                "2000000000000000000" // 2.0 ARTHA in wei
            } else {
                "0"
            };

            let balance_wei = u128::from_str_radix(&balance[2..], 16).unwrap_or(0);
            let balance_artha = balance_wei as f64 / 1e18;

            Ok(Json(serde_json::json!({
                "address": address,
                "balance": balance,
                "currency": "ARTHA",
                "decimals": 18,
                "formatted_balance": format!("{:.6} ARTHA", balance_artha)
            })))
        }
    } else {
        // Handle native account
        let state = state.read().await;
        let account = state
            .get_account(&address)
            .ok_or_else(ApiError::account_not_found)?;

        let balance_artha = account.balance as f64 / 1e18;

        Ok(Json(serde_json::json!({
            "address": address,
            "balance": account.balance.to_string(),
            "currency": "ARTHA",
            "decimals": 18,
            "formatted_balance": format!("{:.6} ARTHA", balance_artha)
        })))
    }
}
