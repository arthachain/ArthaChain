use crate::storage::{Storage, StorageInit, StorageError, Result};
use crate::types::Hash;
use async_trait::async_trait;
use std::path::Path;
use rocksdb::DB;

pub struct RocksDbStorage {
    db: Option<DB>,
}

impl RocksDbStorage {
    pub fn new() -> Self {
        Self { db: None }
    }
}

#[async_trait]
impl Storage for RocksDbStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        let hash = Hash::new(blake3::hash(data).as_bytes().to_vec());
        if let Some(db) = &self.db {
            db.put(hash.as_bytes(), data)
                .map_err(|e| StorageError::Other(e.to_string()))?;
            Ok(hash)
        } else {
            Err(StorageError::Other("Database not initialized".to_string()))
        }
    }

    async fn retrieve(&self, hash: &Hash) -> Result<Option<Vec<u8>>> {
        if let Some(db) = &self.db {
            db.get(hash.as_bytes())
                .map_err(|e| StorageError::Other(e.to_string()))
        } else {
            Err(StorageError::Other("Database not initialized".to_string()))
        }
    }

    async fn exists(&self, hash: &Hash) -> Result<bool> {
        if let Some(db) = &self.db {
            Ok(db.get(hash.as_bytes())
                .map_err(|e| StorageError::Other(e.to_string()))?
                .is_some())
        } else {
            Err(StorageError::Other("Database not initialized".to_string()))
        }
    }

    async fn delete(&self, hash: &Hash) -> Result<()> {
        if let Some(db) = &self.db {
            db.delete(hash.as_bytes())
                .map_err(|e| StorageError::Other(e.to_string()))
        } else {
            Err(StorageError::Other("Database not initialized".to_string()))
        }
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        let stored_data = self.retrieve(hash).await?;
        Ok(stored_data.map_or(false, |d| d == data))
    }

    async fn close(&self) -> Result<()> {
        // RocksDB will close automatically when dropped
        Ok(())
    }
}

#[async_trait]
impl StorageInit for RocksDbStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        let opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        
        self.db = Some(DB::open(&opts, path.as_ref())
            .map_err(|e| StorageError::Other(e.to_string()))?);
        Ok(())
    }
} 