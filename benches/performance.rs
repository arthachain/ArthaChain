use criterion::{black_box, criterion_group, criterion_main, Criterion};
use blockchain_node::consensus::cross_shard::CrossShardManager;
use blockchain_node::consensus::reputation::ReputationManager;
use blockchain_node::storage::Storage;
use blockchain_node::types::Hash;
use tokio::sync::mpsc;
use std::sync::Arc;
use async_trait::async_trait;
use anyhow::Result;

struct MockStorage;

impl MockStorage {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Storage for MockStorage {
    async fn store(&self, _data: &[u8]) -> Result<Hash> {
        Ok(Hash::default())
    }

    async fn retrieve(&self, _hash: &Hash) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }

    async fn exists(&self, _hash: &Hash) -> Result<bool> {
        Ok(false)
    }

    async fn delete(&self, _hash: &Hash) -> Result<()> {
        Ok(())
    }

    async fn verify(&self, _hash: &Hash, _data: &[u8]) -> Result<bool> {
        Ok(true)
    }

    async fn close(&self) -> Result<()> {
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
                
                // Create a reputation manager
                let reputation_manager = Arc::new(ReputationManager::new(
                    1.0,  // max_score
                    100,  // history_size
                    0.5,  // slashing_threshold
                    1000, // min_stake
                ));
                
                let (tx, rx) = mpsc::channel(100);
                let manager = CrossShardManager::new(
                    0, // shard_id
                    3, // total_shards
                    rx,
                    tx,
                    2, // required_signatures
                    5, // finalization_timeout
                    reputation_manager,
                    10, // recovery_timeout
                    3,  // max_recovery_attempts
                );

                black_box(manager);
            });
        })
    });
}

criterion_group!(benches, cross_shard_benchmark);
criterion_main!(benches); 