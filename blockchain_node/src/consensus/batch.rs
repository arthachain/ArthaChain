use std::collections::BTreeMap;
use std::time::{Duration, SystemTime};

use crate::sharding::CrossShardReference;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationBatch {
    pub transactions: Vec<CrossShardReference>,
    pub timestamp: SystemTime,
    pub batch_id: String,
}

#[derive(Debug, Clone)]
pub struct BatchProcessor {
    pub batch_size: usize,
    pub pending_txs: BTreeMap<u8, Vec<CrossShardReference>>, // Priority-based batching
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

    pub fn add_transaction(&mut self, tx: CrossShardReference) {
        // Since there's no priority field, we'll use the first shard in involved_shards as priority
        // This is just an example - you might want to implement a different prioritization strategy
        let priority = if let Some(shard) = tx.involved_shards.first() {
            (*shard % 256) as u8
        } else {
            0 // Default priority if no shards involved
        };

        self.pending_txs
            .entry(priority)
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

    pub fn get_next_batch(&mut self) -> Vec<CrossShardReference> {
        let mut batch = Vec::new();
        for txs in self.pending_txs.values_mut().rev() {
            // Reverse to process high priority first
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
    use crate::crypto::hash::Hash;
    use crate::sharding::CrossShardStatus;

    fn create_test_transaction(priority: u8) -> CrossShardReference {
        // Create a test hash using a placeholder value
        let tx_hash = Hash::new([priority; 32]);
        let involved_shards = vec![priority as u32, 1, 2];

        CrossShardReference {
            tx_hash: tx_hash.to_string(),
            involved_shards,
            status: CrossShardStatus::Pending,
            created_at_height: 100,
        }
    }

    #[test]
    fn test_batch_processing() {
        let mut processor = BatchProcessor::new(10, Duration::from_secs(5));

        // Add transactions with different priorities (based on shard ID)
        for i in 0..15 {
            processor.add_transaction(create_test_transaction((i % 3) as u8));
        }

        assert!(processor.should_process());
        let batch = processor.get_next_batch();
        assert_eq!(batch.len(), 10);

        // We can't easily verify the shard ordering as it matches priority logic,
        // but we can verify we get the expected number of transactions
        assert_eq!(batch.len(), 10);
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
