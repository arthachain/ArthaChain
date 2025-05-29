use axum::{
    extract::{Extension, Path},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::api::ApiError;
use crate::ledger::state::State;
use crate::ledger::transaction::Transaction;
use crate::ledger::transaction::TransactionType;
use crate::utils::crypto::Hash;

/// Response for a transaction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransactionResponse {
    /// Transaction hash
    pub hash: String,
    /// Sender address
    pub sender: String,
    /// Recipient address (if applicable)
    pub recipient: Option<String>,
    /// Transaction amount
    pub amount: u64,
    /// Transaction fee
    pub fee: u64,
    /// Transaction nonce
    pub nonce: u64,
    /// Transaction timestamp
    pub timestamp: u64,
    /// Block hash (if confirmed)
    pub block_hash: Option<String>,
    /// Block height (if confirmed)
    pub block_height: Option<u64>,
    /// Number of confirmations
    pub confirmations: u64,
    /// Transaction type
    pub tx_type: u8,
    /// Transaction data (hex encoded)
    pub data: Option<String>,
}

impl TransactionResponse {
    pub fn from_tx(
        tx: &Transaction,
        block_hash: Option<&Hash>,
        block_height: Option<u64>,
        confirmations: u64,
    ) -> Self {
        Self {
            hash: tx.hash().to_string(),
            sender: tx.sender.clone(),
            recipient: Some(tx.recipient.clone()),
            amount: tx.amount,
            fee: tx.gas_price * tx.gas_limit, // Use gas_price * gas_limit as fee
            nonce: tx.nonce,
            timestamp: tx.timestamp,
            block_hash: block_hash.map(|h| hex::encode(h.as_bytes())),
            block_height,
            confirmations,
            tx_type: match tx.tx_type {
                TransactionType::Transfer => 0,
                TransactionType::ContractCreate => 1,
                TransactionType::Deploy => 1, // Same as ContractCreate
                TransactionType::Call => 2,
                TransactionType::ValidatorRegistration => 3,
                TransactionType::Stake => 4,
                TransactionType::Unstake => 5,
                TransactionType::Delegate => 6,
                TransactionType::ClaimReward => 7,
                TransactionType::Batch => 8,
                TransactionType::System => 9,
                TransactionType::ContractCall => 2, // Same as Call
                TransactionType::Undelegate => 5,   // Same as Unstake
                TransactionType::ClaimRewards => 7, // Same as ClaimReward
                TransactionType::SetValidator => 3, // Same as ValidatorRegistration
                TransactionType::Custom(_) => 10,
            },
            data: if tx.data.is_empty() {
                None
            } else {
                Some(hex::encode(&tx.data))
            },
        }
    }
}

/// Request to submit a new transaction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitTransactionRequest {
    /// Sender address
    pub sender: String,
    /// Recipient address (if applicable)
    pub recipient: Option<String>,
    /// Transaction amount
    pub amount: u64,
    /// Transaction fee
    pub fee: u64,
    /// Transaction nonce
    pub nonce: u64,
    /// Transaction type
    pub tx_type: u8,
    /// Transaction data (hex encoded)
    pub data: Option<String>,
    /// Transaction signature (hex encoded)
    pub signature: String,
}

/// Response for a transaction submission
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubmitTransactionResponse {
    /// Transaction hash
    pub hash: String,
    /// Success status
    pub success: bool,
    /// Message
    pub message: String,
}

/// Get a transaction by its hash
pub async fn get_transaction(
    Path(hash_str): Path<String>,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<Json<TransactionResponse>, ApiError> {
    // Convert hash from hex string to bytes
    let hash_bytes = match hex::decode(&hash_str) {
        Ok(bytes) => bytes,
        Err(_) => {
            return Err(ApiError {
                status: 400,
                message: "Invalid transaction hash format".to_string(),
            })
        }
    };

    // Create a Hash object from bytes
    let hash = Hash::from_bytes(&hash_bytes).map_err(|_| ApiError {
        status: 400,
        message: "Invalid hash length".to_string(),
    })?;

    let state = state.read().await;

    if let Some((tx, block_hash, block_height)) = state.get_transaction_by_hash(&hash.to_string()) {
        // Convert types::Transaction to ledger::transaction::Transaction
        let ledger_tx: crate::ledger::transaction::Transaction = tx.clone();

        // Calculate confirmations if the transaction is in a block
        let confirmations = if let Some(latest_block) = state.latest_block() {
            latest_block.header.height.saturating_sub(block_height) + 1
        } else {
            0
        };

        let block_hash: Option<String> = Some(block_hash);
        let block_hash_ref: Option<crate::utils::crypto::Hash> = block_hash
            .as_ref()
            .and_then(|h| crate::types::Hash::from_hex(h.as_str()).ok())
            .and_then(|h| crate::utils::crypto::Hash::from_bytes(&h.0).ok());
        let response = TransactionResponse::from_tx(
            &ledger_tx,
            block_hash_ref.as_ref(),
            Some(block_height),
            confirmations,
        );
        Ok(Json(response))
    } else {
        Err(ApiError {
            status: 404,
            message: "Transaction not found".to_string(),
        })
    }
}

/// Submit a new transaction to the network
pub async fn submit_transaction(
    Extension(state): Extension<Arc<RwLock<State>>>,
    Json(req): Json<SubmitTransactionRequest>,
) -> Result<Json<SubmitTransactionResponse>, ApiError> {
    // Convert data from hex if provided
    let data = if let Some(data_hex) = req.data {
        hex::decode(&data_hex).map_err(|_| ApiError {
            status: 400,
            message: "Invalid data format".to_string(),
        })?
    } else {
        Vec::new()
    };

    // Convert signature from hex
    let signature = hex::decode(&req.signature).map_err(|_| ApiError {
        status: 400,
        message: "Invalid signature format".to_string(),
    })?;

    // Parse transaction type
    let tx_type = match req.tx_type {
        0 => TransactionType::Transfer,
        1 => TransactionType::ContractCreate,
        2 => TransactionType::Call,
        3 => TransactionType::ValidatorRegistration,
        4 => TransactionType::Stake,
        5 => TransactionType::Unstake,
        6 => TransactionType::Delegate,
        7 => TransactionType::ClaimReward,
        8 => TransactionType::Batch,
        9 => TransactionType::System,
        _ => {
            return Err(ApiError {
                status: 400,
                message: "Invalid transaction type".to_string(),
            })
        }
    };

    // Create the transaction
    let recipient = req.recipient.unwrap_or_default();
    let mut tx = Transaction::new(
        tx_type,
        req.sender.clone(),
        recipient,
        req.amount,
        req.nonce,
        1,     // Default gas price
        21000, // Default gas limit
        data,
    );

    // Set the signature after creation
    tx.signature = signature;

    let state = state.write().await;
    let types_tx = crate::types::Transaction {
        from: crate::types::Address::from_string(&tx.sender).unwrap_or_default(),
        to: crate::types::Address::from_string(&tx.recipient).unwrap_or_default(),
        value: tx.amount,
        gas_price: tx.gas_price,
        gas_limit: tx.gas_limit,
        nonce: tx.nonce,
        data: tx.data.clone(),
        signature: tx.signature.clone(),
        hash: crate::utils::crypto::Hash::default(), // You may want to compute the hash
    };
    state
        .add_pending_transaction(types_tx.into())
        .map_err(|e| ApiError {
            status: 500,
            message: format!("Failed to add transaction: {e}"),
        })?;

    Ok(Json(SubmitTransactionResponse {
        hash: tx.hash().to_string(),
        success: true,
        message: "Transaction submitted successfully".to_string(),
    }))
}
