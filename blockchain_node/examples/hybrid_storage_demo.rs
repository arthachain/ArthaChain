use blockchain_node::storage::{Storage, StorageInit};
use blockchain_node::types::Hash;
use std::path::Path;
use tempfile::tempdir;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::any::Any;
use anyhow::Result;
use async_trait::async_trait;
use tokio;

// Mock SVDB Storage implementation
struct MockSvdbStorage {
    data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl MockSvdbStorage {
    fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl Storage for MockSvdbStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        // Generate random hash as key
        let mut rng = thread_rng();
        let hash_bytes: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();
        let hash = Hash::new(hash_bytes.clone());
        let hash_str = hash.to_hex();
        
        // Store data
        let mut storage = self.data.lock().unwrap();
        storage.insert(hash_str, data.to_vec());
        
        Ok(hash)
    }
    
    async fn retrieve(&self, hash: &Hash) -> Result<Vec<u8>> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        match storage.get(&hash_str) {
            Some(data) => Ok(data.clone()),
            None => Err(anyhow::anyhow!("Data not found"))
        }
    }
    
    async fn exists(&self, hash: &Hash) -> Result<bool> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        Ok(storage.contains_key(&hash_str))
    }
    
    async fn delete(&self, hash: &Hash) -> Result<()> {
        let hash_str = hash.to_hex();
        let mut storage = self.data.lock().unwrap();
        storage.remove(&hash_str);
        Ok(())
    }
    
    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        let exists = self.exists(hash).await?;
        if !exists {
            return Ok(false);
        }
        
        let stored_data = self.retrieve(hash).await?;
        Ok(stored_data == data)
    }

    async fn close(&self) -> Result<()> {
        // No need to close anything for the mock
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
impl StorageInit for MockSvdbStorage {
    async fn init(&mut self, _path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        // No initialization needed for mock
        Ok(())
    }
}

// Mock RocksDB Storage implementation
struct MockRocksDbStorage {
    data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl MockRocksDbStorage {
    fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl Storage for MockRocksDbStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        // Generate random hash as key
        let mut rng = thread_rng();
        let hash_bytes: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();
        let hash = Hash::new(hash_bytes.clone());
        let hash_str = hash.to_hex();
        
        // Store data
        let mut storage = self.data.lock().unwrap();
        storage.insert(hash_str, data.to_vec());
        
        Ok(hash)
    }
    
    async fn retrieve(&self, hash: &Hash) -> Result<Vec<u8>> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        match storage.get(&hash_str) {
            Some(data) => Ok(data.clone()),
            None => Err(anyhow::anyhow!("Data not found"))
        }
    }
    
    async fn exists(&self, hash: &Hash) -> Result<bool> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        Ok(storage.contains_key(&hash_str))
    }
    
    async fn delete(&self, hash: &Hash) -> Result<()> {
        let hash_str = hash.to_hex();
        let mut storage = self.data.lock().unwrap();
        storage.remove(&hash_str);
        Ok(())
    }
    
    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        let exists = self.exists(hash).await?;
        if !exists {
            return Ok(false);
        }
        
        let stored_data = self.retrieve(hash).await?;
        Ok(stored_data == data)
    }

    async fn close(&self) -> Result<()> {
        // No need to close anything for the mock
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
impl StorageInit for MockRocksDbStorage {
    async fn init(&mut self, _path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        // No initialization needed for mock
        Ok(())
    }
}

// Custom HybridStorage Implementation for Demo
struct CustomHybridStorage {
    rocksdb: Box<dyn Storage>,
    svdb: Box<dyn Storage>,
    size_threshold: usize,
}

impl CustomHybridStorage {
    fn new(rocksdb: Box<dyn Storage>, svdb: Box<dyn Storage>) -> Self {
        Self { 
            rocksdb, 
            svdb,
            size_threshold: 1024 * 1024, // 1MB
        }
    }
    
    // Determine if data should be stored in RocksDB or SVDB
    fn should_use_rocksdb(&self, data: &[u8]) -> bool {
        data.len() < self.size_threshold
    }
}

#[async_trait]
impl Storage for CustomHybridStorage {
    async fn store(&self, data: &[u8]) -> Result<Hash> {
        if self.should_use_rocksdb(data) {
            println!("Storing in RocksDB (small data)");
            self.rocksdb.store(data).await
        } else {
            println!("Storing in SVDB (large data)");
            self.svdb.store(data).await
        }
    }
    
    async fn retrieve(&self, hash: &Hash) -> Result<Vec<u8>> {
        // First try to get directly from RocksDB
        match self.rocksdb.retrieve(hash).await {
            Ok(data) => {
                println!("Retrieved from RocksDB");
                Ok(data)
            },
            Err(_) => {
                // Check if it's in SVDB
                let result = self.svdb.retrieve(hash).await;
                if result.is_ok() {
                    println!("Retrieved from SVDB");
                }
                result
            }
        }
    }
    
    async fn exists(&self, hash: &Hash) -> Result<bool> {
        // Check if exists directly in RocksDB
        if self.rocksdb.exists(hash).await? {
            return Ok(true);
        }
        
        // Check if exists in SVDB
        self.svdb.exists(hash).await
    }
    
    async fn delete(&self, hash: &Hash) -> Result<()> {
        // Try to delete from both storages
        let _ = self.rocksdb.delete(hash).await;
        let _ = self.svdb.delete(hash).await;
        Ok(())
    }
    
    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool> {
        // Try to verify in both storages
        if self.rocksdb.exists(hash).await? {
            return self.rocksdb.verify(hash, data).await;
        }
        if self.svdb.exists(hash).await? {
            return self.svdb.verify(hash, data).await;
        }
        Ok(false)
    }

    async fn close(&self) -> Result<()> {
        let _ = self.rocksdb.close().await;
        let _ = self.svdb.close().await;
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
impl StorageInit for CustomHybridStorage {
    async fn init(&mut self, _path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()> {
        // No initialization needed for custom hybrid storage
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Starting Hybrid Storage Demo");
    
    // Create mock storage backends
    let mut svdb = MockSvdbStorage::new();
    let mut rocksdb = MockRocksDbStorage::new();
    
    // Initialize storage backends with temporary paths
    let temp_dir = tempdir()?;
    let svdb_path = temp_dir.path().join("svdb");
    let rocksdb_path = temp_dir.path().join("rocksdb");
    
    svdb.init(Box::new(svdb_path)).await?;
    rocksdb.init(Box::new(rocksdb_path)).await?;
    
    // Create hybrid storage with custom implementation
    let hybrid_storage = CustomHybridStorage::new(Box::new(rocksdb), Box::new(svdb));
    
    // Demo data
    let small_data = b"This is a small piece of data that should go to RocksDB".to_vec();
    let large_data = vec![0u8; 5 * 1024 * 1024]; // 5MB of data that should go to SVDB
    
    println!("Storing small data...");
    let small_hash = hybrid_storage.store(&small_data).await?;
    println!("Small data stored with hash: {}", small_hash);
    
    println!("Storing large data...");
    let large_hash = hybrid_storage.store(&large_data).await?;
    println!("Large data stored with hash: {}", large_hash);
    
    // Retrieve data
    println!("Retrieving small data...");
    let retrieved_small = hybrid_storage.retrieve(&small_hash).await?;
    assert_eq!(retrieved_small, small_data);
    println!("Successfully retrieved small data");
    
    println!("Retrieving large data...");
    let retrieved_large = hybrid_storage.retrieve(&large_hash).await?;
    assert_eq!(retrieved_large, large_data);
    println!("Successfully retrieved large data");
    
    // Check existence
    println!("Checking if data exists...");
    assert!(hybrid_storage.exists(&small_hash).await?);
    assert!(hybrid_storage.exists(&large_hash).await?);
    println!("Both data items exist in storage");
    
    // Verify data
    println!("Verifying data integrity...");
    assert!(hybrid_storage.verify(&small_hash, &small_data).await?);
    assert!(hybrid_storage.verify(&large_hash, &large_data).await?);
    println!("Data integrity verified");
    
    // Delete data
    println!("Deleting data...");
    hybrid_storage.delete(&small_hash).await?;
    hybrid_storage.delete(&large_hash).await?;
    println!("Data deleted");
    
    // Verify deletion
    println!("Verifying deletion...");
    assert!(!hybrid_storage.exists(&small_hash).await?);
    assert!(!hybrid_storage.exists(&large_hash).await?);
    println!("Deletion verified");
    
    // Close storage
    hybrid_storage.close().await?;
    
    println!("Hybrid Storage Demo completed successfully!");
    Ok(())
} 