use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use async_trait::async_trait;
use crate::types::Hash;
use crate::config::Config;
use super::Storage;
use super::RocksDbStorage;
use super::StorageInit;
use log::debug;
use hex;
use std::path::Path;
use downcast_rs::Downcast;

/// Storage for blockchain data
pub struct BlockchainStorage {
    /// RocksDB storage for on-chain data
    #[allow(dead_code)]
    rocksdb: Box<dyn Storage>,
    /// In-memory storage for key-value pairs (used in tests)
    memory: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
}

impl BlockchainStorage {
    /// Create a new blockchain storage instance
    pub fn new(_config: &Config) -> Result<Self> {
        let rocksdb = Box::new(RocksDbStorage::new());
        let memory = Arc::new(Mutex::new(HashMap::new()));
        
        Ok(Self {
            rocksdb,
            memory,
        })
    }
    
    /// Calculate hash for data using blake3
    fn calculate_hash(data: &[u8]) -> Hash {
        let hash = blake3::hash(data);
        hash.as_bytes().to_vec().try_into().unwrap()
    }
}

#[async_trait]
impl StorageInit for BlockchainStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        debug!("Initializing BlockchainStorage");
        
        // Initialize RocksDB storage
        if let Some(storage) = Downcast::as_any_mut(&mut *self.rocksdb).downcast_mut::<RocksDbStorage>() {
            let path_buf = path.as_ref().as_ref().to_path_buf();
            let box_path = Box::new(path_buf) as Box<dyn AsRef<Path> + Send + Sync>;
            storage.init(box_path).await?;
        } else {
            return Err(anyhow!("Failed to get mutable reference to RocksDB storage"));
        }
        
        // No SVDB field in this struct, so we'll just initialize RocksDB
        
        debug!("BlockchainStorage initialized successfully");
        Ok(())
    }
}

#[async_trait]
impl Storage for BlockchainStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        let hash = Self::calculate_hash(data);
        let mut memory = self.memory.lock().unwrap();
        memory.insert(hash.as_ref().to_vec(), data.to_vec());
        debug!("Stored data with hash: {}", hex::encode(&hash));
        Ok(hash)
    }

    async fn retrieve(&self, hash: &Hash) -> Result<Vec<u8>> {
        let memory = self.memory.lock().unwrap();
        match memory.get(hash.as_ref()) {
            Some(data) => {
                debug!("Retrieved data for hash: {}", hex::encode(hash));
                Ok(data.clone())
            },
            None => {
                debug!("Data not found for hash: {}", hex::encode(hash));
                Err(anyhow!("Data not found for hash: {}", hex::encode(hash)))
            }
        }
    }

    async fn exists(&self, hash: &Hash) -> Result<bool> {
        let memory = self.memory.lock().unwrap();
        let exists = memory.contains_key(hash.as_ref());
        debug!("Checked existence of hash {}: {}", hex::encode(hash), exists);
        Ok(exists)
    }

    async fn delete(&self, hash: &Hash) -> Result<()> {
        let mut memory = self.memory.lock().unwrap();
        memory.remove(hash.as_ref());
        debug!("Deleted data for hash: {}", hex::encode(hash));
        Ok(())
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        let calculated_hash = Self::calculate_hash(data);
        let matches = calculated_hash == *hash;
        debug!("Verified data hash {} matches: {}", hex::encode(hash), matches);
        Ok(matches)
    }
    
    async fn close(&self) -> Result<()> {
        debug!("Closing blockchain storage");
        // We don't need to close anything since we're using in-memory storage
        debug!("Blockchain storage closed successfully");
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
} 