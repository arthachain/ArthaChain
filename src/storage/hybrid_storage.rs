use crate::storage::{Storage, StorageInit, StorageError, Result};
use crate::types::Hash;
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

pub struct HybridStorage {
    rocksdb: Arc<dyn Storage>,
    svdb: Arc<dyn Storage>,
    size_threshold: usize,
}

impl HybridStorage {
    pub fn new(rocksdb: Arc<dyn Storage>, svdb: Arc<dyn Storage>, size_threshold: usize) -> Self {
        Self {
            rocksdb,
            svdb,
            size_threshold,
        }
    }

    fn should_use_rocksdb(&self, data: &[u8]) -> bool {
        data.len() <= self.size_threshold
    }
}

#[async_trait]
impl Storage for HybridStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        if self.should_use_rocksdb(data) {
            self.rocksdb.store(data).await
        } else {
            self.svdb.store(data).await
        }
    }

    async fn retrieve(&self, hash: &Hash) -> Result<Option<Vec<u8>>> {
        // Try RocksDB first
        match self.rocksdb.retrieve(hash).await {
            Ok(Some(data)) => Ok(Some(data)),
            Ok(None) => self.svdb.retrieve(hash).await,
            Err(e) => {
                // If RocksDB fails, try SVDB
                match self.svdb.retrieve(hash).await {
                    Ok(data) => Ok(data),
                    Err(_) => Err(e) // Return original error if both fail
                }
            }
        }
    }

    async fn exists(&self, hash: &Hash) -> Result<bool> {
        // Check both storages
        let rocks_exists = self.rocksdb.exists(hash).await?;
        if rocks_exists {
            return Ok(true);
        }
        self.svdb.exists(hash).await
    }

    async fn delete(&self, hash: &Hash) -> Result<()> {
        // Try to delete from both storages
        let rocks_result = self.rocksdb.delete(hash).await;
        let svdb_result = self.svdb.delete(hash).await;
        
        match (rocks_result, svdb_result) {
            (Ok(_), Ok(_)) => Ok(()),
            (Err(e), Ok(_)) | (Ok(_), Err(e)) => Err(e),
            (Err(e1), Err(_)) => Err(e1)
        }
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        if self.should_use_rocksdb(data) {
            self.rocksdb.verify(hash, data).await
        } else {
            self.svdb.verify(hash, data).await
        }
    }

    async fn close(&self) -> Result<()> {
        // Close both storages
        self.rocksdb.close().await?;
        self.svdb.close().await?;
        Ok(())
    }
}

#[async_trait]
impl StorageInit for HybridStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        // Initialize RocksDB with the provided path
        let mut rocksdb = Arc::get_mut(&mut self.rocksdb)
            .ok_or_else(|| StorageError::Other("Failed to get mutable reference to RocksDB storage".to_string()))?;
        rocksdb.init(path).await
    }
} 