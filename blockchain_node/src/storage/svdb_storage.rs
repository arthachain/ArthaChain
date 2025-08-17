use super::{Result, Storage, StorageError, StorageInit, StorageStats};
use crate::types::Hash;
use async_trait::async_trait;
use blake3;
use hex;
use log::debug;
use reqwest::Client;
use rocksdb::{Options, DB};

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Duration;

/// SVDB client for off-chain storage
#[derive(Debug)]
pub struct SvdbStorage {
    /// HTTP client for SVDB API requests
    _client: Client,

    /// Base URL for SVDB API
    _base_url: String,

    /// Database instance
    db: Arc<RwLock<Option<DB>>>,

    /// Path to database for reopening
    db_path: Arc<RwLock<Option<std::path::PathBuf>>>,

    _data: HashMap<String, Vec<u8>>,
}

impl Default for SvdbStorage {
    fn default() -> Self {
        Self::new("http://localhost:8080".to_string()).unwrap_or_else(|_| Self {
            _client: Client::new(),
            _base_url: "http://localhost:8080".to_string(),
            db: Arc::new(RwLock::new(None)),
            db_path: Arc::new(RwLock::new(None)),
            _data: HashMap::new(),
        })
    }
}

impl SvdbStorage {
    /// Create a new SVDB storage instance
    pub fn new(base_url: String) -> anyhow::Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| anyhow::anyhow!(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            _client: client,
            _base_url: base_url,
            db: Arc::new(RwLock::new(None)),
            db_path: Arc::new(RwLock::new(None)),
            _data: HashMap::new(),
        })
    }

    /// Get a reference to the database
    async fn check_db(&self) -> anyhow::Result<()> {
        let db = self
            .db
            .read()
            .map_err(|e| anyhow::anyhow!(format!("Lock error: {e}")))?;
        if db.is_none() {
            // If DB is None, attempt to reopen from path
            let path_clone = {
                let path_lock = self
                    .db_path
                    .read()
                    .map_err(|e| anyhow::anyhow!(format!("Lock error: {e}")))?;
                if let Some(path) = &*path_lock {
                    path.clone()
                } else {
                    return Err(anyhow::anyhow!("Database not initialized".to_string()));
                }
            }; // path_lock is dropped here

            let mut options = Options::default();
            options.create_if_missing(true);

            let db_instance = DB::open(&options, &path_clone)
                .map_err(|e| anyhow::anyhow!(format!("Failed to reopen DB: {e}")))?;

            let mut db_write = self
                .db
                .write()
                .map_err(|e| anyhow::anyhow!(format!("Lock error: {e}")))?;
            *db_write = Some(db_instance);
        }
        Ok(())
    }

    /// Get a value by key (direct method - use trait method instead)
    pub async fn get_direct(&self, key: &[u8]) -> Option<Vec<u8>> {
        let db = self.db.read().ok()?;
        match &*db {
            Some(db) => db.get(key).ok()?,
            None => None,
        }
    }
}

#[async_trait]
impl Storage for SvdbStorage {
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.check_db()
            .await
            .map_err(|_| StorageError::ConnectionError("Database check failed".to_string()))?;
        let db_read = self
            .db
            .read()
            .map_err(|_| StorageError::Other("Lock error".to_string()))?;

        if let Some(db) = &*db_read {
            match db.get(key) {
                Ok(Some(data)) => Ok(Some(data.to_vec())),
                Ok(None) => Ok(None),
                Err(_) => Ok(None),
            }
        } else {
            Err(StorageError::ConnectionError(
                "Database not available".to_string(),
            ))
        }
    }

    async fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.check_db()
            .await
            .map_err(|_| StorageError::ConnectionError("Database check failed".to_string()))?;
        let db_read = self
            .db
            .read()
            .map_err(|_| StorageError::Other("Lock error".to_string()))?;

        if let Some(db) = &*db_read {
            db.put(key, value)
                .map_err(|_| StorageError::WriteError("Put failed".to_string()))?;
            debug!("Stored data with key: {}", hex::encode(key));
            Ok(())
        } else {
            Err(StorageError::ConnectionError(
                "Database not available".to_string(),
            ))
        }
    }

    async fn delete(&self, key: &[u8]) -> Result<()> {
        self.check_db()
            .await
            .map_err(|_| StorageError::ConnectionError("Database check failed".to_string()))?;
        let db_read = self
            .db
            .read()
            .map_err(|_| StorageError::Other("Lock error".to_string()))?;

        if let Some(db) = &*db_read {
            db.delete(key)
                .map_err(|_| StorageError::WriteError("Delete failed".to_string()))?;
            Ok(())
        } else {
            Err(StorageError::ConnectionError(
                "Database not available".to_string(),
            ))
        }
    }

    async fn exists(&self, key: &[u8]) -> Result<bool> {
        match self.get(key).await {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn list_keys(&self, _prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        // SVDB doesn't have a simple way to list keys - simplified implementation
        Ok(Vec::new())
    }

    async fn get_stats(&self) -> Result<StorageStats> {
        Ok(StorageStats {
            total_size: 0,
            used_size: 0,
            num_entries: 0,
            read_operations: 0,
            write_operations: 0,
        })
    }

    async fn flush(&self) -> Result<()> {
        self.check_db()
            .await
            .map_err(|_| StorageError::ConnectionError("Database check failed".to_string()))?;
        let db_read = self
            .db
            .read()
            .map_err(|_| StorageError::Other("Lock error".to_string()))?;

        if let Some(db) = &*db_read {
            db.flush()
                .map_err(|_| StorageError::WriteError("Flush failed".to_string()))?;
            Ok(())
        } else {
            Err(StorageError::ConnectionError(
                "Database not available".to_string(),
            ))
        }
    }

    async fn close(&self) -> Result<()> {
        let mut db_write = self
            .db
            .write()
            .map_err(|_| StorageError::Other("Lock error".to_string()))?;
        *db_write = None;
        Ok(())
    }
}

// Additional helper methods for blockchain storage (not part of Storage trait)
impl SvdbStorage {
    /// Store data and return hash (blockchain-specific method)
    pub async fn store_with_hash(&self, data: &[u8]) -> anyhow::Result<Hash> {
        let hash_bytes = blake3::hash(data).as_bytes().to_vec();
        let hash = Hash::new(hash_bytes);
        self.put(hash.as_ref(), data)
            .await
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        Ok(hash)
    }

    /// Retrieve data by hash (blockchain-specific method)
    pub async fn retrieve_by_hash(&self, hash: &Hash) -> anyhow::Result<Option<Vec<u8>>> {
        self.get(hash.as_ref())
            .await
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    /// Check if data exists by hash (blockchain-specific method)
    pub async fn exists_by_hash(&self, hash: &Hash) -> anyhow::Result<bool> {
        self.exists(hash.as_ref())
            .await
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    /// Delete data by hash (blockchain-specific method)  
    pub async fn delete_by_hash(&self, hash: &Hash) -> anyhow::Result<()> {
        self.delete(hash.as_ref())
            .await
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    /// Verify hash matches data (blockchain-specific method)
    pub async fn verify_hash(&self, hash: &Hash, data: &[u8]) -> anyhow::Result<bool> {
        let calculated_hash = blake3::hash(data).as_bytes().to_vec();
        Ok(calculated_hash == hash.as_ref())
    }
}

#[async_trait]
impl StorageInit for SvdbStorage {
    async fn init(&self, config: &crate::storage::StorageConfig) -> Result<()> {
        let mut options = Options::default();
        options.create_if_missing(true);

        let db_path = Path::new(&config.data_dir);
        let db = DB::open(&options, db_path)
            .map_err(|e| StorageError::ConnectionError(format!("Failed to open SVDB: {e}")))?;

        // Store the path for potential reopening
        let mut path_lock = self
            .db_path
            .write()
            .map_err(|e| StorageError::Other(format!("Lock error: {e}")))?;
        *path_lock = Some(db_path.to_path_buf());

        let mut db_lock = self
            .db
            .write()
            .map_err(|e| StorageError::Other(format!("Lock error: {e}")))?;
        *db_lock = Some(db);

        debug!("SVDB storage initialized successfully");
        Ok(())
    }
}
