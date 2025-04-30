use std::collections::BTreeMap;
use std::time::{Duration, SystemTime};

use super::cross_shard::CrossShardTransaction;

#[derive(Debug, Clone)]
pub struct BatchProcessor {
    pub batch_size: usize,
    pub pending_txs: BTreeMap<u8, Vec<CrossShardTransaction>>, // Priority-based batching
    pub batch_timeout: Duration,
    pub last_batch_time: SystemTime,
    pub last_update: Option<SystemTime>,
}

impl BatchProcessor {
    pub fn new(batch_size: usize, batch_timeout: Duration) -> Self {
        Self {
            batch_size,
            pending_txs: BTreeMap::new(),
            batch_timeout,
            last_batch_time: SystemTime::now(),
            last_update: Some(SystemTime::now()),
        }
    }

    pub fn add_transaction(&mut self, tx: CrossShardTransaction) {
        self.pending_txs.entry(tx.priority)
            .or_insert_with(Vec::new)
            .push(tx);
    }

    pub fn should_process(&self) -> bool {
        let elapsed = SystemTime::now()
            .duration_since(self.last_batch_time)
            .unwrap_or(Duration::from_secs(0));
        
        self.pending_txs.values().map(|v| v.len()).sum::<usize>() >= self.batch_size
            || elapsed >= self.batch_timeout
    }

    pub fn get_next_batch(&mut self) -> Vec<CrossShardTransaction> {
        let mut batch = Vec::new();
        for txs in self.pending_txs.values_mut().rev() { // Reverse to process high priority first
            while batch.len() < self.batch_size && !txs.is_empty() {
                batch.push(txs.remove(0));
            }
            if batch.len() >= self.batch_size {
                break;
            }
        }
        self.last_batch_time = SystemTime::now();
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_transaction(priority: u8) -> CrossShardTransaction {
        let now = SystemTime::now();
        CrossShardTransaction {
            tx_hash: vec![1, 2, 3, 4],
            tx_type: super::super::cross_shard::CrossShardTxType::DirectTransfer {
                from: vec![5, 6, 7, 8],
                to: vec![9, 10, 11, 12],
                amount: 100,
            },
            source_shard: 0,
            target_shards: vec![1],
            data: vec![13, 14, 15, 16],
            status: super::super::cross_shard::CrossShardTxStatus::Pending,
            timestamp: now
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            size: 100,
            priority,
            locality_hint: Some(1),
            merkle_proof: None,
            witness_data: None,
            last_update: Some(now),
        }
    }

    #[test]
    fn test_batch_processing() {
        let mut processor = BatchProcessor::new(10, Duration::from_secs(5));

        // Add transactions with different priorities
        for i in 0..15 {
            processor.add_transaction(create_test_transaction((i % 3) as u8));
        }

        assert!(processor.should_process());
        let batch = processor.get_next_batch();
        assert_eq!(batch.len(), 10);

        // Verify priority ordering
        let mut last_priority = 3;
        for tx in batch {
            assert!(tx.priority <= last_priority);
            last_priority = tx.priority;
        }
    }

    #[test]
    fn test_timeout_trigger() {
        let mut processor = BatchProcessor::new(100, Duration::from_millis(100));
        
        // Add a single transaction
        processor.add_transaction(create_test_transaction(1));
        
        // Should not process yet (not enough transactions)
        assert!(!processor.should_process());
        
        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));
        
        // Should process now due to timeout
        assert!(processor.should_process());
    }
} 