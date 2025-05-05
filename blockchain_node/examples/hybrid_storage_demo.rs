use blockchain_node::storage::{Storage, StorageInit};
use blockchain_node::types::Hash;
use std::path::{Path, PathBuf};
use tempfile::tempdir;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::any::Any;
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
    async fn store(&self, data: &[u8]) -> blockchain_node::storage::Result<Hash> {
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
    
    async fn retrieve(&self, hash: &Hash) -> blockchain_node::storage::Result<Option<Vec<u8>>> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        match storage.get(&hash_str) {
            Some(data) => Ok(Some(data.clone())),
            None => Ok(None)
        }
    }
    
    async fn exists(&self, hash: &Hash) -> blockchain_node::storage::Result<bool> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        Ok(storage.contains_key(&hash_str))
    }
    
    async fn delete(&self, hash: &Hash) -> blockchain_node::storage::Result<()> {
        let hash_str = hash.to_hex();
        let mut storage = self.data.lock().unwrap();
        storage.remove(&hash_str);
        Ok(())
    }
    
    async fn verify(&self, hash: &Hash, data: &[u8]) -> blockchain_node::storage::Result<bool> {
        let exists = self.exists(hash).await?;
        if !exists {
            return Ok(false);
        }
        
        let stored_data = self.retrieve(hash).await?;
        Ok(stored_data.map_or(false, |d| d == data))
    }

    async fn close(&self) -> blockchain_node::storage::Result<()> {
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
    async fn init(&mut self, _path: Box<dyn AsRef<Path> + Send + Sync>) -> blockchain_node::storage::Result<()> {
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
    async fn store(&self, data: &[u8]) -> blockchain_node::storage::Result<Hash> {
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
    
    async fn retrieve(&self, hash: &Hash) -> blockchain_node::storage::Result<Option<Vec<u8>>> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        match storage.get(&hash_str) {
            Some(data) => Ok(Some(data.clone())),
            None => Ok(None)
        }
    }
    
    async fn exists(&self, hash: &Hash) -> blockchain_node::storage::Result<bool> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        Ok(storage.contains_key(&hash_str))
    }
    
    async fn delete(&self, hash: &Hash) -> blockchain_node::storage::Result<()> {
        let hash_str = hash.to_hex();
        let mut storage = self.data.lock().unwrap();
        storage.remove(&hash_str);
        Ok(())
    }
    
    async fn verify(&self, hash: &Hash, data: &[u8]) -> blockchain_node::storage::Result<bool> {
        let exists = self.exists(hash).await?;
        if !exists {
            return Ok(false);
        }
        
        let stored_data = self.retrieve(hash).await?;
        Ok(stored_data.map_or(false, |d| d == data))
    }

    async fn close(&self) -> blockchain_node::storage::Result<()> {
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
    async fn init(&mut self, _path: Box<dyn AsRef<Path> + Send + Sync>) -> blockchain_node::storage::Result<()> {
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
    async fn store(&self, data: &[u8]) -> blockchain_node::storage::Result<Hash> {
        if self.should_use_rocksdb(data) {
            println!("Storing in RocksDB (small data)");
            self.rocksdb.store(data).await
        } else {
            println!("Storing in SVDB (large data)");
            self.svdb.store(data).await
        }
    }
    
    async fn retrieve(&self, hash: &Hash) -> blockchain_node::storage::Result<Option<Vec<u8>>> {
        // First try to get directly from RocksDB
        match self.rocksdb.retrieve(hash).await {
            Ok(Some(data)) => {
                println!("Retrieved from RocksDB");
                Ok(Some(data))
            },
            Ok(None) => {
                // Check if it's in SVDB
                let result = self.svdb.retrieve(hash).await;
                if result.is_ok() {
                    println!("Retrieved from SVDB");
                }
                result
            },
            Err(e) => Err(e)
        }
    }
    
    async fn exists(&self, hash: &Hash) -> blockchain_node::storage::Result<bool> {
        // Check if exists directly in RocksDB
        if self.rocksdb.exists(hash).await? {
            return Ok(true);
        }
        
        // Check if exists in SVDB
        self.svdb.exists(hash).await
    }
    
    async fn delete(&self, hash: &Hash) -> blockchain_node::storage::Result<()> {
        // Try to delete from both storages
        let _ = self.rocksdb.delete(hash).await;
        let _ = self.svdb.delete(hash).await;
        Ok(())
    }
    
    async fn verify(&self, hash: &Hash, data: &[u8]) -> blockchain_node::storage::Result<bool> {
        // Try to verify in both storages
        if self.rocksdb.exists(hash).await? {
            return self.rocksdb.verify(hash, data).await;
        }
        
        self.svdb.verify(hash, data).await
    }
    
    async fn close(&self) -> blockchain_node::storage::Result<()> {
        // Close both storages
        self.rocksdb.close().await?;
        self.svdb.close().await?;
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
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> blockchain_node::storage::Result<()> {
        // Initialize both storages with new Box<PathBuf> that don't borrow from anywhere
        let path_buf = PathBuf::from(path.as_ref().as_ref());
        
        // Create new boxed PathBuf for each storage
        let rocksdb_path = Box::new(path_buf.clone());
        let svdb_path = Box::new(path_buf);
        
        self.rocksdb.as_any_mut().downcast_mut::<MockRocksDbStorage>().unwrap().init(rocksdb_path).await?;
        self.svdb.as_any_mut().downcast_mut::<MockSvdbStorage>().unwrap().init(svdb_path).await?;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create temporary directories for the storages
    let rocksdb_dir = tempdir()?;
    let svdb_dir = tempdir()?;
    
    println!("RocksDB path: {:?}", rocksdb_dir.path());
    println!("SVDB path: {:?}", svdb_dir.path());
    
    // Create mock storage instances
    let mut rocksdb = MockRocksDbStorage::new();
    let mut svdb = MockSvdbStorage::new();
    
    // Initialize storages with owned PathBuf objects
    let rocksdb_path = Box::new(PathBuf::from(rocksdb_dir.path()));
    let svdb_path = Box::new(PathBuf::from(svdb_dir.path()));
    
    rocksdb.init(rocksdb_path).await.map_err(|e| anyhow::anyhow!("RocksDB error: {}", e))?;
    svdb.init(svdb_path).await.map_err(|e| anyhow::anyhow!("SVDB error: {}", e))?;
    
    // Create hybrid storage
    let hybrid = CustomHybridStorage::new(
        Box::new(rocksdb),
        Box::new(svdb)
    );
    
    // Demo - Store small data (should use RocksDB)
    let small_data = b"This is a small piece of data";
    let small_hash = hybrid.store(small_data).await.map_err(|e| anyhow::anyhow!("Storage error: {}", e))?;
    println!("Stored small data with hash: {}", small_hash.to_hex());
    
    // Demo - Store large data (should use SVDB)
    let large_data = vec![0u8; 2 * 1024 * 1024]; // 2MB of zeros
    let large_hash = hybrid.store(&large_data).await.map_err(|e| anyhow::anyhow!("Storage error: {}", e))?;
    println!("Stored large data with hash: {}", large_hash.to_hex());
    
    // Demo - Retrieve data
    let retrieved_small = hybrid.retrieve(&small_hash).await.map_err(|e| anyhow::anyhow!("Storage error: {}", e))?;
    if let Some(data) = retrieved_small {
        println!("Retrieved small data: {} bytes", data.len());
    } else {
        println!("Small data not found!");
    }
    
    let retrieved_large = hybrid.retrieve(&large_hash).await.map_err(|e| anyhow::anyhow!("Storage error: {}", e))?;
    if let Some(data) = retrieved_large {
        println!("Retrieved large data: {} bytes", data.len());
    } else {
        println!("Large data not found!");
    }
    
    // Demo - Verify data
    let small_verified = hybrid.verify(&small_hash, small_data).await.map_err(|e| anyhow::anyhow!("Storage error: {}", e))?;
    println!("Small data verified: {}", small_verified);
    
    let large_verified = hybrid.verify(&large_hash, &large_data).await.map_err(|e| anyhow::anyhow!("Storage error: {}", e))?;
    println!("Large data verified: {}", large_verified);
    
    // Demo - Delete data
    hybrid.delete(&small_hash).await.map_err(|e| anyhow::anyhow!("Storage error: {}", e))?;
    println!("Deleted small data");
    
    let exists = hybrid.exists(&small_hash).await.map_err(|e| anyhow::anyhow!("Storage error: {}", e))?;
    println!("Small data exists after deletion: {}", exists);
    
    // Close storage
    hybrid.close().await.map_err(|e| anyhow::anyhow!("Storage error: {}", e))?;
    println!("Storage closed");
    
    println!("Demo completed successfully!");
    Ok(())
} 