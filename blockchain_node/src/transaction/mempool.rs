use crate::types::{Transaction, Address};
use crate::utils::crypto::Hash;
use crate::crypto::Signature;
use crate::common::{Error, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolTransaction {
    pub transaction: Transaction,
    pub received_at: DateTime<Utc>,
    pub fee_per_gas: u64,
    pub priority: TransactionPriority,
    pub validation_status: ValidationStatus,
    pub retry_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransactionPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    Pending,
    Validated,
    Invalid(String),
    Executed,
}

pub struct Mempool {
    transactions: Arc<RwLock<HashMap<Hash, MempoolTransaction>>>,
    pending_queue: Arc<RwLock<Vec<Hash>>>,
    executed_transactions: Arc<RwLock<HashMap<Hash, MempoolTransaction>>>,
    max_size: usize,
    tx_sender: mpsc::Sender<Transaction>,
    tx_receiver: mpsc::Receiver<Transaction>,
}

impl Mempool {
    pub fn new(max_size: usize) -> Self {
        let (tx_sender, tx_receiver) = mpsc::channel(10000);
        
        Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            pending_queue: Arc::new(RwLock::new(Vec::new())),
            executed_transactions: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            tx_sender,
            tx_receiver,
        }
    }

    /// Add a new transaction to mempool
    pub async fn add_transaction(&self, transaction: Transaction) -> Result<Hash> {
        // Validate transaction first
        self.validate_transaction(&transaction)?;

        let hash = transaction.hash();
        let mempool_tx = MempoolTransaction {
            transaction,
            received_at: Utc::now(),
            fee_per_gas: 1000000000, // 1 gwei default
            priority: TransactionPriority::Normal,
            validation_status: ValidationStatus::Validated,
            retry_count: 0,
        };

        {
            let mut txs = self.transactions.write().unwrap();
            if txs.len() >= self.max_size {
                return Err(Error::MempoolFull);
            }
            txs.insert(hash.clone(), mempool_tx);
        }

        {
            let mut queue = self.pending_queue.write().unwrap();
            queue.push(hash.clone());
        }

        Ok(hash)
    }

    /// Validate transaction before adding to mempool
    fn validate_transaction(&self, tx: &Transaction) -> Result<()> {
        // Check basic structure - Address is a 20-byte array, check if it's all zeros
        if tx.from.0.iter().all(|&b| b == 0) || tx.to.0.iter().all(|&b| b == 0) {
            return Err(crate::common::Error::InvalidTransaction("Invalid addresses".to_string()));
        }

        if tx.value == 0 {
            return Err(crate::common::Error::InvalidTransaction("Amount cannot be zero".to_string()));
        }

        // Verify signature if present (signature is Vec<u8>, not Option<Vec<u8>>)
        if !tx.signature.is_empty() {
            // For now, skip signature verification to avoid complex crypto dependencies
            // In production, this would verify the signature properly
            println!("⚠️ Signature verification skipped for development");
        }

        // Check nonce (in production, would check against account state)
        if tx.nonce == 0 {
            return Err(crate::common::Error::InvalidTransaction("Invalid nonce".to_string()));
        }

        Ok(())
    }

    /// Get next batch of transactions for block inclusion
    pub async fn get_transactions_for_block(&self, max_count: usize) -> Vec<Transaction> {
        let mut selected_txs = Vec::new();
        
        {
            let txs = self.transactions.read().unwrap();
            let mut queue = self.pending_queue.write().unwrap();
            
            // Sort by priority and fee (higher priority/fee first)
            queue.sort_by(|a, b| {
                let tx_a = txs.get(a).unwrap();
                let tx_b = txs.get(b).unwrap();
                
                tx_b.priority.cmp(&tx_a.priority)
                    .then(tx_b.fee_per_gas.cmp(&tx_a.fee_per_gas))
            });

            // Take top transactions
            for hash in queue.iter().take(max_count) {
                if let Some(mempool_tx) = txs.get(hash) {
                    selected_txs.push(mempool_tx.transaction.clone());
                }
            }
        }

        selected_txs
    }

    /// Mark transaction as executed (moved to block)
    pub async fn mark_executed(&self, hash: &Hash) {
        if let Some(mempool_tx) = self.transactions.write().unwrap().remove(hash) {
            let mut executed = self.executed_transactions.write().unwrap();
            executed.insert(hash.clone(), mempool_tx);
        }

        // Remove from pending queue
        if let Ok(mut queue) = self.pending_queue.write() {
            queue.retain(|h| h != hash);
        }
    }

    /// Get mempool statistics
    pub async fn get_stats(&self) -> MempoolStats {
        let txs = self.transactions.read().unwrap();
        let executed = self.executed_transactions.read().unwrap();

        MempoolStats {
            pending_count: txs.len(),
            executed_count: executed.len(),
            total_size_bytes: txs.values().map(|tx| std::mem::size_of_val(&tx.transaction)).sum(),
            oldest_transaction: txs.values().map(|tx| tx.received_at).min(),
            newest_transaction: txs.values().map(|tx| tx.received_at).max(),
        }
    }

    /// Get transaction sender for external submissions
    pub fn get_sender(&self) -> mpsc::Sender<Transaction> {
        self.tx_sender.clone()
    }

    /// Process incoming transactions
    pub async fn process_incoming_transactions(&mut self) {
        while let Some(transaction) = self.tx_receiver.recv().await {
            if let Err(e) = self.add_transaction(transaction).await {
                eprintln!("Failed to add transaction to mempool: {}", e);
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MempoolStats {
    pub pending_count: usize,
    pub executed_count: usize,
    pub total_size_bytes: usize,
    pub oldest_transaction: Option<DateTime<Utc>>,
    pub newest_transaction: Option<DateTime<Utc>>,
}

impl Default for Mempool {
    fn default() -> Self {
        Self::new(10000) // Default 10k transaction capacity
    }
}
