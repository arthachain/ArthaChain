use crate::storage::{Storage, StorageInit, StorageError, Result};
use crate::types::Hash;
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

pub struct BlockchainStorage {
    rocksdb: Arc<dyn Storage>,
}

impl BlockchainStorage {
    pub fn new(rocksdb: Arc<dyn Storage>) -> Self {
        Self { rocksdb }
    }
}

#[async_trait]
impl Storage for BlockchainStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        self.rocksdb.store(data).await
    }

    async fn retrieve(&self, hash: &Hash) -> Result<Option<Vec<u8>>> {
        self.rocksdb.retrieve(hash).await
    }

    async fn exists(&self, hash: &Hash) -> Result<bool> {
        self.rocksdb.exists(hash).await
    }

    async fn delete(&self, hash: &Hash) -> Result<()> {
        self.rocksdb.delete(hash).await
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        self.rocksdb.verify(hash, data).await
    }

    async fn close(&self) -> Result<()> {
        self.rocksdb.close().await
    }
}

#[async_trait]
impl StorageInit for BlockchainStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        if let Some(storage) = Arc::get_mut(&mut self.rocksdb) {
            storage.init(path).await
        } else {
            Err(StorageError::Other("Failed to get mutable reference to RocksDB storage".to_string()))
        }
    }
} 