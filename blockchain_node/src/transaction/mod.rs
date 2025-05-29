// Re-export the Transaction type from the ledger
pub use crate::ledger::transaction::Transaction;

pub mod mempool;
pub mod parallel_processor;

pub use mempool::{EnhancedMempool, MempoolConfig, MempoolStats};

// Extension for benchmarking
impl Transaction {
    /// Create a new test transaction for benchmarking
    pub fn new_test_transaction(id: usize) -> Self {
        use crate::ledger::transaction::TransactionType;
        use std::time::{SystemTime, UNIX_EPOCH};

        // Create random-looking addresses from the ID
        let sender = format!("sender{}", id);
        let recipient = format!("recipient{}", id);

        // Generate random data based on ID
        let mut data = Vec::new();
        data.extend_from_slice(&id.to_be_bytes());
        data.extend_from_slice(&[id as u8; 32]);

        // Create signature - empty for test transactions
        let signature = Vec::new();

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            tx_type: TransactionType::Transfer,
            sender,
            recipient,
            amount: (id as u64) * 100,
            nonce: id as u64,
            gas_price: 1,
            gas_limit: 21000,
            data,
            signature,
            timestamp,
            #[cfg(feature = "bls")]
            bls_signature: None,
            status: crate::ledger::transaction::TransactionStatus::Pending,
        }
    }
}
