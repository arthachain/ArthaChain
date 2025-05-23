use async_trait::async_trait;
use blockchain_node::consensus::reputation::{ReputationConfig, ReputationManager};
use blockchain_node::network::cross_shard::{
    CrossShardConfig, CrossShardManager, CrossShardMessage,
};
use blockchain_node::storage::{Storage, StorageError};
use blockchain_node::types::Hash;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

struct MockStorage;

impl MockStorage {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Storage for MockStorage {
    async fn store(&self, _data: &[u8]) -> Result<Hash, StorageError> {
        Ok(Hash::default())
    }

    async fn retrieve(&self, _hash: &Hash) -> Result<Option<Vec<u8>>, StorageError> {
        Ok(Some(Vec::new()))
    }

    async fn exists(&self, _hash: &Hash) -> Result<bool, StorageError> {
        Ok(false)
    }

    async fn delete(&self, _hash: &Hash) -> Result<(), StorageError> {
        Ok(())
    }

    async fn verify(&self, _hash: &Hash, _data: &[u8]) -> Result<bool, StorageError> {
        Ok(true)
    }

    async fn close(&self) -> Result<(), StorageError> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn cross_shard_benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("cross_shard_consensus", |b| {
        b.iter(|| {
            runtime.block_on(async {
                // Setup test environment
                let _storage = Arc::new(MockStorage::new());

                // Create a reputation manager with the proper config
                let reputation_config = ReputationConfig {
                    min_reputation: 0.3,
                    initial_reputation: 0.5,
                    max_adjustment: 0.1,
                    decay_factor: 0.99,
                    decay_interval_secs: 3600,
                };
                let _reputation_manager = Arc::new(ReputationManager::new(reputation_config));

                let (_tx, _rx) = mpsc::channel::<CrossShardMessage>(100);

                // Create config for CrossShardManager
                let config = CrossShardConfig {
                    max_retries: 3,
                    retry_interval: Duration::from_secs(5),
                    message_timeout: Duration::from_secs(30),
                    batch_size: 100,
                    max_queue_size: 1000,
                    sync_interval: Duration::from_secs(60),
                };

                let manager = CrossShardManager::new(config);

                black_box(manager);
            });
        })
    });
}

criterion_group!(benches, cross_shard_benchmark);
criterion_main!(benches);
