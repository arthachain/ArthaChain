use async_trait::async_trait;
use blockchain_node::storage::{Storage, StorageInit, StorageStats};
use blockchain_node::types::Hash;
use rand::{thread_rng, Rng};
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

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

    fn _get_base_path(&self) -> &Path {
        // Mock implementation: return a dummy path
        Path::new("mock_cold_storage")
    }
}

#[async_trait]
impl Storage for MockSvdbStorage {
    async fn store(&self, data: &[u8]) -> anyhow::Result<Hash> {
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

    async fn retrieve(&self, hash: &Hash) -> anyhow::Result<Option<Vec<u8>>> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        match storage.get(&hash_str) {
            Some(data) => Ok(Some(data.clone())),
            None => Ok(None),
        }
    }

    async fn exists(&self, hash: &Hash) -> anyhow::Result<bool> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        Ok(storage.contains_key(&hash_str))
    }

    async fn delete(&self, hash: &Hash) -> anyhow::Result<()> {
        let hash_str = hash.to_hex();
        let mut storage = self.data.lock().unwrap();
        storage.remove(&hash_str);
        Ok(())
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> anyhow::Result<bool> {
        let stored_data = self.retrieve(hash).await?;
        match stored_data {
            Some(ref stored) => Ok(stored == data),
            None => Ok(false),
        }
    }

    async fn close(&self) -> anyhow::Result<()> {
        // Mock close operation
        Ok(())
    }

    async fn get_stats(&self) -> anyhow::Result<StorageStats> {
        let storage = self.data.lock().unwrap();
        let total_entries = storage.len() as u64;
        let total_size = storage.values().map(|v| v.len() as u64).sum::<u64>();
        Ok(StorageStats {
            total_entries,
            total_size_bytes: total_size,
            average_entry_size: if total_entries > 0 {
                total_size as f64 / total_entries as f64
            } else {
                0.0
            },
            storage_efficiency: 1.0, // Mock 100% efficiency
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[async_trait::async_trait]
impl StorageInit for MockSvdbStorage {
    async fn init(&mut self, _path: Box<dyn AsRef<Path> + Send + Sync>) -> anyhow::Result<()> {
        // Mock initialization: no actual path operations needed
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

    fn _get_base_path(&self) -> &Path {
        // Mock implementation: return a dummy path
        Path::new("mock_hot_storage")
    }
}

#[async_trait]
impl Storage for MockRocksDbStorage {
    async fn store(&self, data: &[u8]) -> anyhow::Result<Hash> {
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

    async fn retrieve(&self, hash: &Hash) -> anyhow::Result<Option<Vec<u8>>> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        match storage.get(&hash_str) {
            Some(data) => Ok(Some(data.clone())),
            None => Ok(None),
        }
    }

    async fn exists(&self, hash: &Hash) -> anyhow::Result<bool> {
        let hash_str = hash.to_hex();
        let storage = self.data.lock().unwrap();
        Ok(storage.contains_key(&hash_str))
    }

    async fn delete(&self, hash: &Hash) -> anyhow::Result<()> {
        let hash_str = hash.to_hex();
        let mut storage = self.data.lock().unwrap();
        storage.remove(&hash_str);
        Ok(())
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> anyhow::Result<bool> {
        let stored_data = self.retrieve(hash).await?;
        match stored_data {
            Some(ref stored) => Ok(stored == data),
            None => Ok(false),
        }
    }

    async fn close(&self) -> anyhow::Result<()> {
        // Mock close operation
        Ok(())
    }

    async fn get_stats(&self) -> anyhow::Result<StorageStats> {
        let storage = self.data.lock().unwrap();
        let total_entries = storage.len() as u64;
        let total_size = storage.values().map(|v| v.len() as u64).sum::<u64>();
        Ok(StorageStats {
            total_entries,
            total_size_bytes: total_size,
            average_entry_size: if total_entries > 0 {
                total_size as f64 / total_entries as f64
            } else {
                0.0
            },
            storage_efficiency: 1.0, // Mock 100% efficiency
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[async_trait::async_trait]
impl StorageInit for MockRocksDbStorage {
    async fn init(&mut self, _path: Box<dyn AsRef<Path> + Send + Sync>) -> anyhow::Result<()> {
        // Mock initialization: no actual path operations needed
        Ok(())
    }
}

// Custom Hybrid Storage implementation
struct CustomHybridStorage {
    hot_storage: MockRocksDbStorage,
    cold_storage: MockSvdbStorage,
}

impl CustomHybridStorage {
    pub fn new() -> Self {
        Self {
            hot_storage: MockRocksDbStorage::new(),
            cold_storage: MockSvdbStorage::new(),
        }
    }
}

#[async_trait::async_trait]
impl StorageInit for CustomHybridStorage {
    async fn init(&mut self, base_path: Box<dyn AsRef<Path> + Send + Sync>) -> anyhow::Result<()> {
        // For mock storages, we can pass the same path or just unitialize them.
        // Since it's a mock, the actual path doesn't matter.
        let path_buf = base_path.as_ref().as_ref().to_path_buf();
        self.hot_storage.init(Box::new(path_buf.clone())).await?;
        self.cold_storage.init(Box::new(path_buf)).await?;
        Ok(())
    }
}

#[async_trait]
impl Storage for CustomHybridStorage {
    async fn store(&self, data: &[u8]) -> anyhow::Result<Hash> {
        // Store in hot storage first
        let hash = self.hot_storage.store(data).await?;

        // Optionally mirror to cold storage for larger data
        if data.len() > 1024 {
            let _ = self.cold_storage.store(data).await;
        }

        Ok(hash)
    }

    async fn retrieve(&self, hash: &Hash) -> anyhow::Result<Option<Vec<u8>>> {
        // Try hot storage first
        if let Some(data) = self.hot_storage.retrieve(hash).await? {
            return Ok(Some(data));
        }

        // Fallback to cold storage
        self.cold_storage.retrieve(hash).await
    }

    async fn exists(&self, hash: &Hash) -> anyhow::Result<bool> {
        // Check both storages
        let hot_exists = self.hot_storage.exists(hash).await?;
        if hot_exists {
            return Ok(true);
        }

        self.cold_storage.exists(hash).await
    }

    async fn delete(&self, hash: &Hash) -> anyhow::Result<()> {
        // Delete from both storages
        let _ = self.hot_storage.delete(hash).await;
        let _ = self.cold_storage.delete(hash).await;
        Ok(())
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> anyhow::Result<bool> {
        // Verify against available storage
        if self.hot_storage.exists(hash).await? {
            return self.hot_storage.verify(hash, data).await;
        }

        self.cold_storage.verify(hash, data).await
    }

    async fn close(&self) -> anyhow::Result<()> {
        self.hot_storage.close().await?;
        self.cold_storage.close().await?;
        Ok(())
    }

    async fn get_stats(&self) -> anyhow::Result<StorageStats> {
        let hot_stats = self.hot_storage.get_stats().await?;
        let cold_stats = self.cold_storage.get_stats().await?;

        Ok(StorageStats {
            total_entries: hot_stats.total_entries + cold_stats.total_entries,
            total_size_bytes: hot_stats.total_size_bytes + cold_stats.total_size_bytes,
            average_entry_size: {
                let total_entries = hot_stats.total_entries + cold_stats.total_entries;
                let total_size = hot_stats.total_size_bytes + cold_stats.total_size_bytes;
                if total_entries > 0 {
                    total_size as f64 / total_entries as f64
                } else {
                    0.0
                }
            },
            storage_efficiency: (hot_stats.storage_efficiency + cold_stats.storage_efficiency)
                / 2.0,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Hybrid Storage Demo");
    println!("==================");

    // Create temporary directory for demo
    let temp_dir = tempdir()?;
    let base_path = temp_dir.path();

    println!("Using temp directory: {:?}", base_path);

    // Initialize custom hybrid storage
    let mut storage = CustomHybridStorage::new();

    // Initialize storage
    let path_box: Box<dyn AsRef<Path> + Send + Sync> = Box::new(base_path.to_path_buf());
    storage.init(path_box).await?;

    println!("✓ Storage initialized successfully");

    // Test data
    let test_data = b"Hello, Hybrid Storage World!";
    println!("Test data: {:?}", String::from_utf8_lossy(test_data));

    // Store data
    println!("\nStoring data...");
    let hash = storage.store(test_data).await?;
    println!("✓ Data stored with hash: {}", hash.to_hex());

    // Check if data exists
    let exists = storage.exists(&hash).await?;
    println!("✓ Data exists: {}", exists);

    // Retrieve data
    println!("\nRetrieving data...");
    match storage.retrieve(&hash).await? {
        Some(retrieved_data) => {
            println!(
                "✓ Data retrieved: {:?}",
                String::from_utf8_lossy(&retrieved_data)
            );

            // Verify data integrity
            let is_valid = storage.verify(&hash, &retrieved_data).await?;
            println!("✓ Data verification: {}", is_valid);
        }
        None => {
            println!("✗ Data not found");
        }
    }

    // Test with larger data (should trigger cold storage)
    println!("\nTesting with larger data...");
    let large_data = vec![42u8; 2048]; // 2KB data
    let large_hash = storage.store(&large_data).await?;
    println!("✓ Large data stored with hash: {}", large_hash.to_hex());

    // Get storage statistics
    println!("\nStorage statistics:");
    let stats = storage.get_stats().await?;
    println!("Total entries: {}", stats.total_entries);
    println!("Total size: {} bytes", stats.total_size_bytes);
    println!("Average entry size: {:.2} bytes", stats.average_entry_size);
    println!("Storage efficiency: {:.2}", stats.storage_efficiency);

    // Clean up
    storage.delete(&hash).await?;
    storage.delete(&large_hash).await?;
    println!("\n✓ Data cleaned up");

    // Close storage
    storage.close().await?;
    println!("✓ Storage closed");

    println!("\nHybrid Storage Demo completed successfully!");
    Ok(())
}
