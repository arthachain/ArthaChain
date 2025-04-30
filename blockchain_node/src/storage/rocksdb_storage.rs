use anyhow::{Result, anyhow};
use rocksdb::{DB, Options};
use std::path::Path;
use blake3;
use hex;
use super::Storage;
use log::debug;
use async_trait::async_trait;
use std::sync::{Arc, RwLock};
use crate::types::Hash;
use super::StorageInit;
use anyhow::Error;

/// RocksDB storage implementation for on-chain data
pub struct RocksDbStorage {
    /// RocksDB database instance
    db: Arc<RwLock<Option<DB>>>,
}

impl RocksDbStorage {
    /// Create a new RocksDB storage
    pub fn new() -> Self {
        Self {
            db: Arc::new(RwLock::new(None)),
        }
    }

    /// Check if database is initialized
    async fn check_db(&self) -> Result<Arc<DB>> {
        self.get_db()
    }

    fn get_db(&self) -> Result<Arc<DB>> {
        let read_guard = self.db.read().map_err(|e| anyhow!("Lock error: {}", e))?;
        match &*read_guard {
            Some(db) => {
                // Use Arc::new to clone the DB instead of unsafe code
                // This works because DB is Clonable
                let db_clone = DB::open_default(db.path()).unwrap();
                Ok(Arc::new(db_clone))
            },
            None => Err(anyhow!("RocksDB not initialized")),
        }
    }
}

#[async_trait]
impl Storage for RocksDbStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        let hash_bytes = blake3::hash(data).as_bytes().to_vec();
        let hash = Hash::new(hash_bytes);
        
        let db = self.check_db().await?;
        db.put(hash.as_bytes(), data)?;
        debug!("Stored data with hash: {}", hex::encode(&hash));
        Ok(hash)
    }
    
    async fn retrieve(&self, hash: &Hash) -> Result<Vec<u8>> {
        let db = self.check_db().await?;
        if let Some(data) = db.get(hash.as_bytes())? {
            debug!("Retrieved data for hash: {}", hex::encode(hash));
            Ok(data)
        } else {
            debug!("Data not found for hash: {}", hex::encode(hash));
            Err(Error::msg("Key not found"))
        }
    }
    
    async fn exists(&self, hash: &Hash) -> Result<bool> {
        let db = self.check_db().await?;
        let exists = db.get(hash.as_bytes())?.is_some();
        debug!("Checked existence of hash {}: {}", hex::encode(hash), exists);
        Ok(exists)
    }
    
    async fn delete(&self, hash: &Hash) -> Result<()> {
        let db = self.check_db().await?;
        db.delete(hash.as_bytes())?;
        debug!("Deleted data for hash: {}", hex::encode(hash));
        Ok(())
    }
    
    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        let calculated_hash_bytes = blake3::hash(data).as_bytes().to_vec();
        let calculated_hash = Hash::new(calculated_hash_bytes);
        let matches = calculated_hash == *hash;
        debug!("Verified data hash {} matches: {}", hex::encode(hash), matches);
        Ok(matches)
    }

    async fn close(&self) -> Result<()> {
        debug!("Closing RocksDB storage");
        *self.db.write().map_err(|e| anyhow!("Lock error: {}", e))? = None;
        debug!("RocksDB storage closed successfully");
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[async_trait]
impl StorageInit for RocksDbStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        let path_ref = path.as_ref();
        debug!("Initializing RocksDbStorage at path: {}", path_ref.as_ref().display());
        
        // Create directory if it doesn't exist
        let path_ref = path_ref.as_ref();
        if !path_ref.exists() {
            std::fs::create_dir_all(path_ref)?;
        }
        
        // Create database options
        let mut opts = Options::default();
        opts.create_if_missing(true);
        
        // Open the database
        let db_path = path_ref.join("blockchain.db");
        debug!("Opening RocksDB at: {}", db_path.display());
        
        // Create the database instance
        let db = DB::open(&opts, db_path)?;
        *self.db.write().map_err(|e| anyhow!("Lock error: {}", e))? = Some(db);
        
        debug!("RocksDbStorage initialized successfully");
        Ok(())
    }
} 