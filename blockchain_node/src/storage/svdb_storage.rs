use std::path::Path;
use async_trait::async_trait;
use crate::types::Hash;
use super::{Storage, StorageInit, StorageError, Result};
use log::debug;
use rocksdb::{DB, Options};
use std::sync::{Arc, RwLock};
use reqwest::Client;
use blake3;
use hex;
use std::time::Duration;
use std::collections::HashMap;
use std::any::Any;

/// SVDB client for off-chain storage
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

impl SvdbStorage {
    /// Create a new SVDB storage instance
    pub fn new(base_url: String) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| StorageError::Other(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            _client: client,
            _base_url: base_url,
            db: Arc::new(RwLock::new(None)),
            db_path: Arc::new(RwLock::new(None)),
            _data: HashMap::new(),
        })
    }

    /// Get a reference to the database
    async fn check_db(&self) -> Result<()> {
        let db = self.db.read().map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        if db.is_none() {
            // If DB is None, attempt to reopen from path
            let path_clone = {
                let path_lock = self.db_path.read().map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
                if let Some(path) = &*path_lock {
                    path.clone()
                } else {
                    return Err(StorageError::Other("Database not initialized".to_string()));
                }
            }; // path_lock is dropped here
            
            let mut options = Options::default();
            options.create_if_missing(true);
            
            let db_instance = DB::open(&options, &path_clone)
                .map_err(|e| StorageError::Other(format!("Failed to reopen DB: {}", e)))?;
            
            let mut db_write = self.db.write().map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
            *db_write = Some(db_instance);
        }
        Ok(())
    }

    /// Get a value by key
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let db = self.db.read().ok()?;
        match &*db {
            Some(db) => db.get(key).ok()?,
            None => None,
        }
    }
}

#[async_trait]
impl Storage for SvdbStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        self.check_db().await?;
        let hash_bytes = blake3::hash(data).as_bytes().to_vec();
        let hash = Hash::new(hash_bytes);
        
        let db_read = self.db.read().map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        if let Some(db) = &*db_read {
            db.put(hash.as_bytes(), data).map_err(|e| StorageError::Other(e.to_string()))?;
            debug!("Stored data with hash: {}", hex::encode(&hash));
            Ok(hash)
        } else {
            Err(StorageError::Other("Database not initialized".to_string()))
        }
    }

    async fn retrieve(&self, hash: &Hash) -> Result<Option<Vec<u8>>> {
        self.check_db().await?;
        let db_read = self.db.read().map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        
        if let Some(db) = &*db_read {
            match db.get(hash.as_bytes()) {
                Ok(Some(data)) => {
                    debug!("Retrieved data for hash: {}", hex::encode(hash));
                    Ok(Some(data))
                },
                Ok(None) => {
                    debug!("Data not found for hash: {}", hex::encode(hash));
                    Ok(None)
                },
                Err(e) => Err(StorageError::Other(e.to_string())),
            }
        } else {
            Err(StorageError::Other("Database not initialized".to_string()))
        }
    }

    async fn exists(&self, hash: &Hash) -> Result<bool> {
        self.check_db().await?;
        let db_read = self.db.read().map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        
        if let Some(db) = &*db_read {
            match db.get(hash.as_bytes()) {
                Ok(Some(_)) => {
                    debug!("Data exists for hash: {}", hex::encode(hash));
                    Ok(true)
                },
                Ok(None) => {
                    debug!("Data does not exist for hash: {}", hex::encode(hash));
                    Ok(false)
                },
                Err(e) => Err(StorageError::Other(e.to_string())),
            }
        } else {
            Err(StorageError::Other("Database not initialized".to_string()))
        }
    }

    async fn delete(&self, hash: &Hash) -> Result<()> {
        self.check_db().await?;
        let db_read = self.db.read().map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        
        if let Some(db) = &*db_read {
            db.delete(hash.as_bytes()).map_err(|e| StorageError::Other(e.to_string()))?;
            debug!("Deleted data for hash: {}", hex::encode(hash));
            Ok(())
        } else {
            Err(StorageError::Other("Database not initialized".to_string()))
        }
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        let calculated_hash = blake3::hash(data).as_bytes().to_vec();
        let matches = calculated_hash == hash.as_bytes();
        debug!("Verified data hash {} matches: {}", hex::encode(hash), matches);
        Ok(matches)
    }

    async fn close(&self) -> Result<()> {
        let mut db = self.db.write().map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        *db = None;
        debug!("SVDB storage closed successfully");
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[async_trait]
impl StorageInit for SvdbStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        let mut options = Options::default();
        options.create_if_missing(true);
        
        let path_ref = path.as_ref();
        let db = DB::open(&options, path_ref.as_ref())
            .map_err(|e| StorageError::Other(format!("Failed to open RocksDB: {}", e)))?;
        
        // Store the path for potential reopening
        let mut path_lock = self.db_path.write().map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        *path_lock = Some(path_ref.as_ref().to_path_buf());
        
        let mut db_lock = self.db.write()
            .map_err(|e| StorageError::Other(format!("Lock error: {}", e)))?;
        *db_lock = Some(db);
        
        debug!("SVDB storage initialized successfully");
        Ok(())
    }
} 