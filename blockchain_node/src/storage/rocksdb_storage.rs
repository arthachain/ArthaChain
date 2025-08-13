use crate::storage::{Storage, StorageInit};
use async_trait::async_trait;
use rocksdb::{Options, DB};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// RocksDB storage implementation for on-chain data
#[derive(Debug)]
pub struct RocksDbStorage {
    /// RocksDB database instance
    db: Arc<RwLock<Option<DB>>>,
    /// Path to database
    path: PathBuf,
    /// Configuration options
    write_buffer_size: u64,
    max_open_files: u32,
    compression_enabled: bool,
    block_cache_size: u64,
}

impl Default for RocksDbStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl RocksDbStorage {
    /// Create a new RocksDB storage instance
    pub fn new() -> Self {
        Self {
            db: Arc::new(RwLock::new(None)),
            path: PathBuf::from("data/rocksdb"),
            write_buffer_size: 64 * 1024 * 1024, // 64MB
            max_open_files: 1000,
            compression_enabled: true,
            block_cache_size: 512 * 1024 * 1024, // 512MB
        }
    }

    /// Create a new RocksDB storage with custom path
    pub fn new_with_path(path: &Path) -> anyhow::Result<Self> {
        let mut storage = Self {
            db: Arc::new(RwLock::new(None)),
            path: path.to_path_buf(),
            write_buffer_size: 64 * 1024 * 1024, // 64MB
            max_open_files: 1000,
            compression_enabled: true,
            block_cache_size: 512 * 1024 * 1024, // 512MB
        };

        storage.init_db()?;
        Ok(storage)
    }

    /// Initialize or check database connection
    async fn check_db(&self) -> anyhow::Result<()> {
        let db = self
            .db
            .read()
            .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
        if db.is_none() {
            drop(db); // Release read lock
            self.init_db()?;
        }
        Ok(())
    }

    /// Get a value by key
    pub async fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        if let Ok(db) = self.db.read() {
            if let Some(ref db) = *db {
                return db.get(key).ok().flatten();
            }
        }
        None
    }

    /// Put a key-value pair
    pub async fn put(&self, key: &[u8], value: &[u8]) -> anyhow::Result<()> {
        self.check_db().await?;
        let db = self
            .db
            .read()
            .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
        if let Some(ref db) = *db {
            db.put(key, value)
                .map_err(|e| anyhow::anyhow!("Failed to put to RocksDB: {}", e))?;
        }
        Ok(())
    }

    /// Delete a value by key
    pub async fn delete_key(&self, key: &[u8]) -> anyhow::Result<()> {
        self.check_db().await?;
        let db = self
            .db
            .read()
            .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
        if let Some(ref db) = *db {
            db.delete(key)
                .map_err(|e| anyhow::anyhow!("Failed to delete from RocksDB: {}", e))?;
        }
        Ok(())
    }

    /// Initialize the database
    fn init_db(&self) -> anyhow::Result<()> {
        let mut options = Options::default();
        options.create_if_missing(true);
        options.set_write_buffer_size(self.write_buffer_size as usize);
        options.set_max_open_files(self.max_open_files as i32);

        let db = DB::open(&options, &self.path)
            .map_err(|e| anyhow::anyhow!("Failed to open RocksDB: {}", e))?;

        let mut db_lock = self
            .db
            .write()
            .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
        *db_lock = Some(db);

        Ok(())
    }

    /// Open database with custom path
    pub async fn open<P: AsRef<Path> + Send + Sync>(&mut self, path: P) -> anyhow::Result<()> {
        // Update the path
        self.path = path.as_ref().to_path_buf();

        let mut options = Options::default();
        options.create_if_missing(true);
        options.set_write_buffer_size(self.write_buffer_size as usize);
        options.set_max_open_files(self.max_open_files as i32);

        let db = DB::open(&options, &self.path)
            .map_err(|e| anyhow::anyhow!("Failed to open RocksDB: {}", e))?;

        let mut db_lock = self
            .db
            .write()
            .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
        *db_lock = Some(db);

        Ok(())
    }
}

#[async_trait]
impl Storage for RocksDbStorage {
    async fn get(&self, key: &[u8]) -> crate::storage::Result<Option<Vec<u8>>> {
        // Delegate to the existing get method
        Ok(RocksDbStorage::get(self, key).await)
    }

    async fn put(&self, key: &[u8], value: &[u8]) -> crate::storage::Result<()> {
        // Delegate to the existing put method
        RocksDbStorage::put(self, key, value)
            .await
            .map_err(|e| crate::storage::StorageError::WriteError(e.to_string()))
    }

    async fn delete(&self, key: &[u8]) -> crate::storage::Result<()> {
        self.delete_key(key)
            .await
            .map_err(|e| crate::storage::StorageError::WriteError(e.to_string()))
    }

    async fn exists(&self, key: &[u8]) -> crate::storage::Result<bool> {
        Ok(RocksDbStorage::get(self, key).await.is_some())
    }

    async fn list_keys(&self, _prefix: &[u8]) -> crate::storage::Result<Vec<Vec<u8>>> {
        // RocksDB doesn't have a simple way to list all keys with prefix
        // This would need to be implemented with an iterator
        Ok(Vec::new())
    }

    async fn get_stats(&self) -> crate::storage::Result<crate::storage::StorageStats> {
        Ok(crate::storage::StorageStats::default())
    }

    async fn flush(&self) -> crate::storage::Result<()> {
        // RocksDB auto-flushes, so this is a no-op
        Ok(())
    }

    async fn close(&self) -> crate::storage::Result<()> {
        // RocksDB handles closing automatically on drop
        Ok(())
    }
}

#[async_trait]
impl StorageInit for RocksDbStorage {
    async fn init(&self, _config: &crate::storage::StorageConfig) -> crate::storage::Result<()> {
        Ok(())
    }
}
