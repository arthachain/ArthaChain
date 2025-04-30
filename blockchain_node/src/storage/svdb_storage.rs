use std::path::Path;
use async_trait::async_trait;
use anyhow::{Result, anyhow};
use crate::types::Hash;
use super::Storage;
use super::StorageInit;
use log::debug;
use rocksdb::{DB, Options};
use std::sync::{Arc, RwLock};
use reqwest::Client;
use blake3;
use hex;
use std::time::Duration;
use std::collections::HashMap;

/// SVDB client for off-chain storage
pub struct SvdbStorage {
    /// HTTP client for SVDB API requests
    _client: Client,
    
    /// Base URL for SVDB API
    _base_url: String,

    /// Database instance
    db: Arc<RwLock<Option<DB>>>,

    _data: HashMap<String, Vec<u8>>,
}

impl SvdbStorage {
    /// Create a new SVDB storage client
    pub fn new(svdb_url: &str) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
            
        Ok(Self {
            _client: client,
            _base_url: svdb_url.to_string(),
            db: Arc::new(RwLock::new(None)),
            _data: HashMap::new(),
        })
    }

    /// Check if database is initialized
    async fn check_db(&self) -> Result<Arc<DB>> {
        let db = self.db.read().map_err(|e| anyhow!("Lock error: {}", e))?;
        match &*db {
            Some(db) => {
                // We can't clone DB directly, so we'll need to create a new Arc
                let path = db.path().to_owned();
                let new_db = DB::open_default(path)?;
                Ok(Arc::new(new_db))
            },
            None => Err(anyhow!("Database not initialized"))
        }
    }
}

#[async_trait]
impl Storage for SvdbStorage {
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
            Err(anyhow!("Key not found"))
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
        debug!("Closing SVDB storage");
        *self.db.write().map_err(|e| anyhow!("Lock error: {}", e))? = None;
        debug!("SVDB storage closed successfully");
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
impl StorageInit for SvdbStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        let path_ref = path.as_ref();
        debug!("Initializing SVDB storage at path: {}", path_ref.as_ref().display());
        
        // Create directory if it doesn't exist
        let path_ref = path_ref.as_ref();
        if !path_ref.exists() {
            std::fs::create_dir_all(path_ref)?;
        }
        
        // Create database options
        let mut opts = Options::default();
        opts.create_if_missing(true);
        
        // Open the database
        let db_path = path_ref.join("svdb.db");
        debug!("Opening SVDB at: {}", db_path.display());
        
        // Create the database instance
        let db = DB::open(&opts, db_path)?;
        *self.db.write().map_err(|e| anyhow!("Lock error: {}", e))? = Some(db);
        
        debug!("SVDB storage initialized successfully");
        Ok(())
    }
} 