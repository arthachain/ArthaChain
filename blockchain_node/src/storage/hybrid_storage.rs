use super::{Result, Storage, StorageInit};
use async_trait::async_trait;
use std::any::Any;

type StorageResult<T> = Result<T>;

/// Hybrid storage combining RocksDB and SVDB
pub struct HybridStorage {
    /// RocksDB storage for on-chain data
    rocksdb: Box<dyn Storage>,

    /// SVDB storage for off-chain data
    svdb: Box<dyn Storage>,

    /// Size threshold (in bytes) for deciding between RocksDB and SVDB
    size_threshold: usize,
}

impl HybridStorage {
    /// Create a new hybrid storage instance
    pub fn new(_svdb_url: String, size_threshold: usize) -> anyhow::Result<Self> {
        // Assume both storages are provided externally in production; here we keep placeholders
        let rocksdb = crate::storage::rocksdb_storage::RocksDbStorage::new();
        let svdb = crate::storage::svdb_storage::SvdbStorage::new("memory://".to_string())?;
        let rocksdb: Box<dyn Storage> = Box::new(rocksdb);
        let svdb: Box<dyn Storage> = Box::new(svdb);

        Ok(Self {
            rocksdb,
            svdb,
            size_threshold,
        })
    }
    
    /// Clone the hybrid storage (creates new instances)
    pub fn clone(&self) -> anyhow::Result<Self> {
        // Create new storage instances for the clone
        let rocksdb = crate::storage::rocksdb_storage::RocksDbStorage::new();
        let svdb = crate::storage::svdb_storage::SvdbStorage::new("memory://".to_string())?;
        let rocksdb: Box<dyn Storage> = Box::new(rocksdb);
        let svdb: Box<dyn Storage> = Box::new(svdb);
        
        Ok(Self {
            rocksdb,
            svdb,
            size_threshold: self.size_threshold,
        })
    }
}

#[async_trait]
impl Storage for HybridStorage {
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Try RocksDB first for smaller data, then SVDB
        match self.rocksdb.get(key).await {
            Ok(Some(data)) => Ok(Some(data)),
            Ok(None) => self.svdb.get(key).await,
            Err(_) => self.svdb.get(key).await,
        }
    }

    async fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        // Use SVDB for larger data, RocksDB for smaller
        if value.len() > self.size_threshold {
            self.svdb.put(key, value).await
        } else {
            self.rocksdb.put(key, value).await
        }
    }

    async fn delete(&self, key: &[u8]) -> Result<()> {
        // Try to delete from both storages
        let rocksdb_result = self.rocksdb.delete(key).await;
        let svdb_result = self.svdb.delete(key).await;

        // Return Ok if at least one deletion succeeded
        match (rocksdb_result, svdb_result) {
            (Ok(_), _) | (_, Ok(_)) => Ok(()),
            (Err(_), Err(e)) => Err(e),
        }
    }

    async fn exists(&self, key: &[u8]) -> Result<bool> {
        // Check both storages
        match self.rocksdb.exists(key).await {
            Ok(true) => Ok(true),
            Ok(false) => self.svdb.exists(key).await,
            Err(_) => self.svdb.exists(key).await,
        }
    }

    async fn list_keys(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        // Combine keys from both storages
        let mut keys = self.rocksdb.list_keys(prefix).await.unwrap_or_default();
        let svdb_keys = self.svdb.list_keys(prefix).await.unwrap_or_default();
        keys.extend(svdb_keys);
        keys.sort();
        keys.dedup();
        Ok(keys)
    }

    async fn get_stats(&self) -> Result<crate::storage::StorageStats> {
        Ok(crate::storage::StorageStats::default())
    }

    async fn flush(&self) -> Result<()> {
        self.rocksdb.flush().await?;
        self.svdb.flush().await?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.rocksdb.close().await?;
        self.svdb.close().await?;
        Ok(())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[async_trait]
impl StorageInit for HybridStorage {
    async fn init(&self, _config: &crate::storage::StorageConfig) -> Result<()> {
        Ok(())
    }
}
