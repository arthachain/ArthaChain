use crate::ledger::state::State;
use crate::types::{Address, Block, Hash, Transaction};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Blockchain API for external interactions
pub struct BlockchainApi {
    /// Reference to blockchain state
    state: Arc<RwLock<State>>,
}

/// API request for getting block by hash
#[derive(Debug, Serialize, Deserialize)]
pub struct GetBlockRequest {
    pub block_hash: String,
}

/// API request for getting transaction by hash
#[derive(Debug, Serialize, Deserialize)]
pub struct GetTransactionRequest {
    pub tx_hash: String,
}

/// API request for submitting transaction
#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitTransactionRequest {
    pub transaction: Transaction,
}

/// API response for blockchain info
#[derive(Debug, Serialize, Deserialize)]
pub struct BlockchainInfoResponse {
    pub latest_block_height: u64,
    pub latest_block_hash: String,
    pub total_transactions: usize,
    pub total_validators: usize,
}

impl BlockchainApi {
    /// Create new blockchain API instance
    pub fn new(state: Arc<RwLock<State>>) -> Self {
        Self { state }
    }

    /// Get blockchain information
    pub async fn get_blockchain_info(&self) -> Result<BlockchainInfoResponse> {
        let state = self.state.read().await;
        let height = state.get_height().unwrap_or(0);
        let latest_hash = state
            .get_latest_block_hash()
            .unwrap_or_else(|_| "".to_string());
        let total_transactions = state.get_total_transactions();
        let total_validators = state.get_validator_count();

        Ok(BlockchainInfoResponse {
            latest_block_height: height,
            latest_block_hash: latest_hash,
            total_transactions,
            total_validators,
        })
    }

    /// Get block by hash
    pub async fn get_block_by_hash(&self, hash: &Hash) -> Result<Option<Block>> {
        let state = self.state.read().await;
        match state.get_block_by_hash(hash) {
            Some(ledger_block) => {
                // Convert from ledger::Block to types::Block
                let block: Block = ledger_block.into();
                Ok(Some(block))
            }
            None => Ok(None),
        }
    }

    /// Get block by height
    pub async fn get_block_by_height(&self, height: u64) -> Result<Option<Block>> {
        let state = self.state.read().await;
        match state.get_block_by_height(height) {
            Some(ledger_block) => {
                // Convert from ledger::Block to types::Block
                let block: Block = ledger_block.into();
                Ok(Some(block))
            }
            None => Ok(None),
        }
    }

    /// Submit transaction to mempool
    pub async fn submit_transaction(&self, transaction: Transaction) -> Result<String> {
        let mut state = self.state.write().await;
        state.add_pending_transaction(transaction.clone().into())?;
        Ok(hex::encode(transaction.hash().as_ref()))
    }

    /// Get account balance
    pub async fn get_balance(&self, address: &Address) -> Result<u64> {
        let state = self.state.read().await;
        state.get_balance(&address.to_hex())
    }

    /// Get account nonce
    pub async fn get_nonce(&self, address: &Address) -> Result<u64> {
        let state = self.state.read().await;
        state.get_nonce(&address.to_hex())
    }

    /// Get pending transactions
    pub async fn get_pending_transactions(&self, limit: usize) -> Result<Vec<Transaction>> {
        let state = self.state.read().await;
        Ok(state
            .get_pending_transactions(limit)
            .into_iter()
            .map(|tx| tx.into())
            .collect())
    }

    /// Get recent blocks
    pub async fn get_recent_blocks(&self, limit: u64) -> Result<Vec<Block>> {
        let state = self.state.read().await;
        let current_height = state.get_height().unwrap_or(0);
        let start_height = if current_height >= limit {
            current_height - limit + 1
        } else {
            0
        };
        // Get blocks and convert them
        let ledger_blocks = state.get_blocks(start_height, limit)?;
        let blocks: Vec<Block> = ledger_blocks
            .into_iter()
            .map(|ledger_block| ledger_block.into())
            .collect();
        Ok(blocks)
    }
}
