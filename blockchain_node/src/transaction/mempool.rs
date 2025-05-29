use anyhow::Result;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::ledger::transaction::{Transaction, TransactionStatus};
use crate::types::{AccountId, TransactionHash};
use crate::utils::crypto::quantum_resistant_hash;

/// Configuration for mempool
#[derive(Clone, Debug)]
pub struct MempoolConfig {
    /// Maximum size of mempool in bytes
    pub max_size_bytes: usize,
    /// Maximum number of transactions in mempool
    pub max_transactions: usize,
    /// Default time-to-live for transactions
    pub default_ttl: Duration,
    /// Minimum gas price to accept a transaction
    pub min_gas_price: u64,
    /// Whether to use quantum-resistant hashing
    pub use_quantum_resistant: bool,
    /// Cleanup interval for expired transactions
    pub cleanup_interval: Duration,
    /// Maximum number of transactions per account
    pub max_txs_per_account: usize,
}

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 1024 * 1024 * 1024, // 1GB
            max_transactions: 100_000,
            default_ttl: Duration::from_secs(3600), // 1 hour
            min_gas_price: 1,
            use_quantum_resistant: true,
            cleanup_interval: Duration::from_secs(60),
            max_txs_per_account: 100,
        }
    }
}

/// Transaction with metadata for mempool management
#[derive(Clone, Debug)]
pub struct MempoolTransaction {
    /// The actual transaction
    pub transaction: Transaction,
    /// When the transaction was added to the mempool
    pub added_at: Instant,
    /// When the transaction expires (TTL)
    pub expires_at: Instant,
    /// Size of the transaction in bytes
    pub size: usize,
    /// Hash of the transaction
    pub hash: TransactionHash,
    /// Priority score (calculated based on gas price, size, etc.)
    pub priority: u64,
    /// Whether this transaction uses quantum-resistant signature
    pub quantum_resistant: bool,
}

impl MempoolTransaction {
    /// Create a new mempool transaction with default TTL
    pub fn new(transaction: Transaction, config: &MempoolConfig) -> Result<Self> {
        let size = transaction.estimate_size();
        let hash = if config.use_quantum_resistant {
            // Use quantum-resistant hashing
            let tx_bytes = transaction.serialize()?;
            let hash_result = quantum_resistant_hash(&tx_bytes)?;
            TransactionHash::from(hash_result)
        } else {
            transaction.hash()
        };

        let now = Instant::now();
        let expires_at = now + config.default_ttl;

        // Calculate priority based on gas price and other factors
        let base_priority = transaction.gas_price;
        // Adjust priority based on size (smaller is better)
        let size_factor = 1000.0 / (size as f64).max(1.0);
        // Adjust priority based on time in mempool (newer is better)
        let time_factor = 1.0;

        // Combine factors for final priority score
        let priority = ((base_priority as f64) * size_factor * time_factor) as u64;

        Ok(Self {
            transaction,
            added_at: now,
            expires_at,
            size,
            hash,
            priority,
            quantum_resistant: config.use_quantum_resistant,
        })
    }

    /// Check if transaction has expired
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }

    /// Get time until expiration
    pub fn time_to_expiry(&self) -> Duration {
        if self.is_expired() {
            Duration::from_secs(0)
        } else {
            self.expires_at.duration_since(Instant::now())
        }
    }

    /// Extend transaction TTL by the specified duration
    pub fn extend_ttl(&mut self, duration: Duration) {
        self.expires_at = self.expires_at.max(Instant::now()) + duration;
    }
}

/// Enhanced mempool with TTL and prioritization
pub struct EnhancedMempool {
    /// Mempool configuration
    config: MempoolConfig,
    /// Transactions by hash
    transactions: Arc<RwLock<HashMap<TransactionHash, MempoolTransaction>>>,
    /// Transactions by account
    by_account: Arc<RwLock<HashMap<AccountId, HashSet<TransactionHash>>>>,
    /// Transactions by priority
    by_priority: Arc<RwLock<BTreeMap<u64, HashSet<TransactionHash>>>>,
    /// Transactions by expiry
    by_expiry: Arc<RwLock<BTreeMap<u64, HashSet<TransactionHash>>>>,
    /// Current size in bytes
    current_size: Arc<RwLock<usize>>,
    /// Last cleanup timestamp
    last_cleanup: Arc<RwLock<Instant>>,
}

impl EnhancedMempool {
    /// Create a new enhanced mempool
    pub fn new(config: MempoolConfig) -> Self {
        Self {
            config,
            transactions: Arc::new(RwLock::new(HashMap::new())),
            by_account: Arc::new(RwLock::new(HashMap::new())),
            by_priority: Arc::new(RwLock::new(BTreeMap::new())),
            by_expiry: Arc::new(RwLock::new(BTreeMap::new())),
            current_size: Arc::new(RwLock::new(0)),
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Add a transaction to the mempool
    pub async fn add_transaction(&self, transaction: Transaction) -> Result<TransactionHash> {
        let tx_hash = transaction.hash();

        // Check if transaction already exists
        {
            let txs = self.transactions.read().await;
            if txs.contains_key(&tx_hash) {
                return Ok(tx_hash);
            }
        }

        // Validate transaction
        self.validate_transaction(&transaction)?;

        // Create mempool transaction
        let mempool_tx = MempoolTransaction::new(transaction.clone(), &self.config)?;

        // Check account transaction limit
        {
            let by_account = self.by_account.read().await;
            let account_id = AccountId::from(transaction.sender.clone());
            if let Some(account_txs) = by_account.get(&account_id) {
                if account_txs.len() >= self.config.max_txs_per_account {
                    return Err(anyhow::anyhow!("Too many transactions for account"));
                }
            }
        }

        // Check mempool capacity
        {
            let size = *self.current_size.read().await;
            let txs = self.transactions.read().await;

            if txs.len() >= self.config.max_transactions {
                // Remove lowest priority transactions if full
                self.evict_transactions(1).await?;
            }

            if size + mempool_tx.size > self.config.max_size_bytes {
                // Calculate how much space we need to free
                let space_needed = (size + mempool_tx.size) - self.config.max_size_bytes;
                // Evict transactions to free space
                self.evict_by_size(space_needed).await?;
            }
        }

        // Add transaction to indices
        {
            let expire_time = mempool_tx.expires_at.elapsed().as_secs();
            let priority = mempool_tx.priority;
            let account = transaction.sender.clone();
            let hash = mempool_tx.hash.clone();

            // Update by_priority index
            {
                let mut by_priority = self.by_priority.write().await;
                by_priority
                    .entry(priority)
                    .or_default()
                    .insert(hash.clone());
            }

            // Update by_account index
            {
                let mut by_account = self.by_account.write().await;
                let account_id = AccountId::from(account.clone());
                by_account
                    .entry(account_id)
                    .or_default()
                    .insert(hash.clone());
            }

            // Update by_expiry index
            {
                let mut by_expiry = self.by_expiry.write().await;
                by_expiry
                    .entry(expire_time)
                    .or_default()
                    .insert(hash.clone());
            }

            // Add transaction to main storage
            {
                let mut txs = self.transactions.write().await;
                let mut size = self.current_size.write().await;

                *size += mempool_tx.size;
                txs.insert(hash.clone(), mempool_tx);
            }
        }

        // Clean up expired transactions occasionally
        self.maybe_cleanup().await;

        Ok(tx_hash)
    }

    /// Remove a transaction from the mempool
    pub async fn remove_transaction(
        &self,
        tx_hash: &TransactionHash,
    ) -> Result<Option<Transaction>> {
        let mut txs = self.transactions.write().await;

        if let Some(mempool_tx) = txs.remove(tx_hash) {
            let mut size = self.current_size.write().await;
            *size -= mempool_tx.size;

            // Remove from indices
            {
                let mut by_priority = self.by_priority.write().await;
                if let Some(set) = by_priority.get_mut(&mempool_tx.priority) {
                    set.remove(tx_hash);
                    if set.is_empty() {
                        by_priority.remove(&mempool_tx.priority);
                    }
                }
            }

            {
                let mut by_account = self.by_account.write().await;
                let account_id = AccountId::from(mempool_tx.transaction.sender.clone());
                if let Some(set) = by_account.get_mut(&account_id) {
                    set.remove(tx_hash);
                    if set.is_empty() {
                        by_account.remove(&account_id);
                    }
                }
            }

            {
                let expire_time = mempool_tx.expires_at.elapsed().as_secs();
                let mut by_expiry = self.by_expiry.write().await;
                if let Some(set) = by_expiry.get_mut(&expire_time) {
                    set.remove(tx_hash);
                    if set.is_empty() {
                        by_expiry.remove(&expire_time);
                    }
                }
            }

            return Ok(Some(mempool_tx.transaction));
        }

        Ok(None)
    }

    /// Get a transaction from the mempool
    pub async fn get_transaction(&self, tx_hash: &TransactionHash) -> Option<Transaction> {
        let txs = self.transactions.read().await;
        txs.get(tx_hash).map(|tx| tx.transaction.clone())
    }

    /// Get best transactions for inclusion in a block
    pub async fn get_best_transactions(&self, max_count: usize) -> Vec<Transaction> {
        let txs = self.transactions.read().await;
        let by_priority = self.by_priority.read().await;

        let mut result = Vec::new();

        // Iterate through transactions by priority (highest first)
        for (_priority, hashes) in by_priority.iter().rev() {
            for hash in hashes {
                if let Some(tx) = txs.get(hash) {
                    if !tx.is_expired() {
                        result.push(tx.transaction.clone());
                        if result.len() >= max_count {
                            return result;
                        }
                    }
                }
            }
        }

        result
    }

    /// Get transactions by account
    pub async fn get_account_transactions(&self, account: &AccountId) -> Vec<Transaction> {
        let txs = self.transactions.read().await;
        let by_account = self.by_account.read().await;

        let mut result = Vec::new();

        if let Some(hashes) = by_account.get(account) {
            for hash in hashes {
                if let Some(tx) = txs.get(hash) {
                    if !tx.is_expired() {
                        result.push(tx.transaction.clone());
                    }
                }
            }
        }

        result
    }

    /// Get mempool statistics
    pub async fn get_stats(&self) -> MempoolStats {
        let txs = self.transactions.read().await;
        let size = *self.current_size.read().await;

        // Count transactions by status
        let mut pending_count = 0;
        let mut expired_count = 0;

        for tx in txs.values() {
            if tx.is_expired() {
                expired_count += 1;
            } else {
                pending_count += 1;
            }
        }

        MempoolStats {
            total_transactions: txs.len(),
            pending_transactions: pending_count,
            expired_transactions: expired_count,
            size_bytes: size,
            max_size_bytes: self.config.max_size_bytes,
            min_gas_price: self.config.min_gas_price,
        }
    }

    /// Validate transaction for mempool inclusion
    fn validate_transaction(&self, transaction: &Transaction) -> Result<()> {
        // Check gas price
        if transaction.gas_price < self.config.min_gas_price {
            return Err(anyhow::anyhow!("Gas price too low"));
        }

        // Check transaction status
        if transaction.status != TransactionStatus::Pending {
            return Err(anyhow::anyhow!("Transaction not in pending state"));
        }

        // Check nonce
        // This would typically validate against account state
        // but we're simplifying for this implementation

        // Check signature
        // This would validate the transaction signature
        // but we're simplifying for this implementation

        Ok(())
    }

    /// Clean up expired transactions
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let now = Instant::now();
        let mut expired_hashes = Vec::new();

        // Find expired transactions
        {
            let txs = self.transactions.read().await;
            for (hash, tx) in txs.iter() {
                if tx.is_expired() {
                    expired_hashes.push(hash.clone());
                }
            }
        }

        // Remove expired transactions
        let mut removed_count = 0;
        for hash in expired_hashes {
            if self.remove_transaction(&hash).await?.is_some() {
                removed_count += 1;
            }
        }

        // Update cleanup timestamp
        *self.last_cleanup.write().await = now;

        if removed_count > 0 {
            info!(
                "Cleaned up {} expired transactions from mempool",
                removed_count
            );
        }

        Ok(removed_count)
    }

    /// Clean up if interval has passed
    async fn maybe_cleanup(&self) {
        let last = *self.last_cleanup.read().await;
        let now = Instant::now();

        if now.duration_since(last) >= self.config.cleanup_interval {
            if let Err(e) = self.cleanup_expired().await {
                warn!("Error during mempool cleanup: {}", e);
            }
        }
    }

    /// Evict lowest priority transactions
    async fn evict_transactions(&self, count: usize) -> Result<usize> {
        let mut to_remove = Vec::new();

        // Find lowest priority transactions
        {
            let by_priority = self.by_priority.read().await;
            let mut remaining = count;

            // Start with lowest priorities
            for (_, hashes) in by_priority.iter() {
                for hash in hashes {
                    to_remove.push(hash.clone());
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                }
                if remaining == 0 {
                    break;
                }
            }
        }

        // Remove transactions
        let mut removed = 0;
        for hash in to_remove {
            if self.remove_transaction(&hash).await?.is_some() {
                removed += 1;
            }
        }

        Ok(removed)
    }

    /// Evict transactions to free a specific amount of space
    async fn evict_by_size(&self, space_needed: usize) -> Result<usize> {
        let mut to_remove = Vec::new();
        let mut space_freed = 0;

        // Find transactions to remove
        {
            let txs = self.transactions.read().await;
            let by_priority = self.by_priority.read().await;

            // Start with lowest priorities
            for (_, hashes) in by_priority.iter() {
                for hash in hashes {
                    if let Some(tx) = txs.get(hash) {
                        to_remove.push(hash.clone());
                        space_freed += tx.size;
                        if space_freed >= space_needed {
                            break;
                        }
                    }
                }
                if space_freed >= space_needed {
                    break;
                }
            }
        }

        // Remove transactions
        let mut removed = 0;
        for hash in to_remove {
            if self.remove_transaction(&hash).await?.is_some() {
                removed += 1;
            }
        }

        Ok(removed)
    }

    /// Check if mempool contains a transaction
    pub async fn contains(&self, tx_hash: &TransactionHash) -> bool {
        let txs = self.transactions.read().await;
        txs.contains_key(tx_hash)
    }

    /// Get current size of mempool in bytes
    pub async fn size_bytes(&self) -> usize {
        *self.current_size.read().await
    }

    /// Get number of transactions in mempool
    pub async fn transaction_count(&self) -> usize {
        self.transactions.read().await.len()
    }
}

/// Mempool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolStats {
    /// Total number of transactions in mempool
    pub total_transactions: usize,
    /// Number of pending (not expired) transactions
    pub pending_transactions: usize,
    /// Number of expired transactions
    pub expired_transactions: usize,
    /// Current size in bytes
    pub size_bytes: usize,
    /// Maximum size in bytes
    pub max_size_bytes: usize,
    /// Minimum gas price accepted
    pub min_gas_price: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::transaction::TransactionType;
    use std::time::SystemTime;

    fn create_test_transaction(sender: &str, nonce: u64, gas_price: u64) -> Transaction {
        Transaction {
            tx_type: TransactionType::Transfer,
            sender: sender.to_string(),
            recipient: "recipient".to_string(),
            amount: 100,
            nonce,
            gas_price,
            gas_limit: 21000,
            data: Vec::new(),
            signature: Vec::new(),
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            #[cfg(feature = "bls")]
            bls_signature: None,
            status: TransactionStatus::Pending,
        }
    }

    #[tokio::test]
    async fn test_add_and_remove_transaction() {
        let config = MempoolConfig::default();
        let mempool = EnhancedMempool::new(config);

        // Add transaction
        let tx = create_test_transaction("sender1", 1, 10);
        let tx_hash = mempool.add_transaction(tx.clone()).await.unwrap();

        // Verify transaction was added
        assert!(mempool.contains(&tx_hash).await);
        assert_eq!(mempool.transaction_count().await, 1);

        // Get transaction
        let retrieved = mempool.get_transaction(&tx_hash).await.unwrap();
        assert_eq!(retrieved.nonce, tx.nonce);

        // Remove transaction
        let removed = mempool.remove_transaction(&tx_hash).await.unwrap().unwrap();
        assert_eq!(removed.nonce, tx.nonce);

        // Verify transaction was removed
        assert!(!mempool.contains(&tx_hash).await);
        assert_eq!(mempool.transaction_count().await, 0);
    }

    #[tokio::test]
    async fn test_transaction_prioritization() {
        let config = MempoolConfig::default();
        let mempool = EnhancedMempool::new(config);

        // Add transactions with different gas prices
        let tx1 = create_test_transaction("sender1", 1, 10);
        let tx2 = create_test_transaction("sender2", 1, 20);
        let tx3 = create_test_transaction("sender3", 1, 5);

        mempool.add_transaction(tx1).await.unwrap();
        mempool.add_transaction(tx2).await.unwrap();
        mempool.add_transaction(tx3).await.unwrap();

        // Get best transactions
        let best = mempool.get_best_transactions(2).await;

        // Should return tx2 (gas price 20) and tx1 (gas price 10)
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].gas_price, 20);
        assert_eq!(best[1].gas_price, 10);
    }

    #[tokio::test]
    async fn test_transaction_expiry() {
        let mut config = MempoolConfig::default();
        // Set TTL to 1 millisecond for testing
        config.default_ttl = Duration::from_millis(1);
        let mempool = EnhancedMempool::new(config);

        // Add transaction
        let tx = create_test_transaction("sender1", 1, 10);
        let tx_hash = mempool.add_transaction(tx).await.unwrap();

        // Wait for transaction to expire
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Clean up expired transactions
        let removed = mempool.cleanup_expired().await.unwrap();
        assert_eq!(removed, 1);

        // Verify transaction was removed
        assert!(!mempool.contains(&tx_hash).await);
    }

    #[tokio::test]
    async fn test_account_transactions() {
        let config = MempoolConfig::default();
        let mempool = EnhancedMempool::new(config);

        // Add transactions from different accounts
        let tx1 = create_test_transaction("sender1", 1, 10);
        let tx2 = create_test_transaction("sender1", 2, 20);
        let tx3 = create_test_transaction("sender2", 1, 15);

        mempool.add_transaction(tx1).await.unwrap();
        mempool.add_transaction(tx2).await.unwrap();
        mempool.add_transaction(tx3).await.unwrap();

        // Get transactions for sender1
        let sender1_id = AccountId::from("sender1".to_string());
        let sender1_txs = mempool.get_account_transactions(&sender1_id).await;
        assert_eq!(sender1_txs.len(), 2);

        // Get transactions for sender2
        let sender2_id = AccountId::from("sender2".to_string());
        let sender2_txs = mempool.get_account_transactions(&sender2_id).await;
        assert_eq!(sender2_txs.len(), 1);

        // Get transactions for unknown sender
        let unknown_id = AccountId::from("unknown".to_string());
        let unknown_txs = mempool.get_account_transactions(&unknown_id).await;
        assert_eq!(unknown_txs.len(), 0);
    }
}
