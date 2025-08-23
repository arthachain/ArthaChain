use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::Any;

use std::fmt;

// Storage-specific Result type
pub type Result<T> = std::result::Result<T, StorageError>;

// Re-export Hash from crypto for storage modules
pub use crate::crypto::Hash;

// Core storage traits and types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageError {
    NotFound(String),
    WriteError(String),
    ReadError(String),
    ConnectionError(String),
    InvalidData(String),
    Other(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::NotFound(msg) => write!(f, "Not found: {}", msg),
            StorageError::WriteError(msg) => write!(f, "Write error: {}", msg),
            StorageError::ReadError(msg) => write!(f, "Read error: {}", msg),
            StorageError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            StorageError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            StorageError::Other(msg) => write!(f, "Storage error: {}", msg),
        }
    }
}

impl std::error::Error for StorageError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_size: u64,
    pub used_size: u64,
    pub num_entries: u64,
    pub read_operations: u64,
    pub write_operations: u64,
}

impl Default for StorageStats {
    fn default() -> Self {
        Self {
            total_size: 0,
            used_size: 0,
            num_entries: 0,
            read_operations: 0,
            write_operations: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_dir: String,
    pub max_file_size: u64,
    pub cache_size: usize,
    pub enable_compression: bool,
    pub backup_enabled: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: "data".to_string(),
            max_file_size: 1024 * 1024 * 1024, // 1GB
            cache_size: 10000,
            enable_compression: true,
            backup_enabled: true,
        }
    }
}

// Core Storage trait - unified and consistent for ArthaChain
#[async_trait]
pub trait Storage: Send + Sync {
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    async fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;
    async fn delete(&self, key: &[u8]) -> Result<()>;
    async fn exists(&self, key: &[u8]) -> Result<bool>;
    async fn list_keys(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>>;
    async fn get_stats(&self) -> Result<StorageStats>;
    async fn flush(&self) -> Result<()>;
    async fn close(&self) -> Result<()>;
    
    /// Get a reference to the concrete type
    fn as_any(&self) -> &dyn Any;
}

// Storage initialization trait
#[async_trait]
pub trait StorageInit {
    async fn init(&self, config: &StorageConfig) -> Result<()>;
}



// Additional storage types
#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    None,
    Zstd,
    Lz4,
    Gzip,
}

#[derive(Debug, Clone)]
pub struct MemMapOptions {
    pub read_only: bool,
    pub max_size: u64,
    pub huge_tlb: bool,
}

impl Default for MemMapOptions {
    fn default() -> Self {
        Self {
            read_only: false,
            max_size: 1024 * 1024 * 1024, // 1GB
            huge_tlb: false,
        }
    }
}

pub mod blockchain_storage;
pub mod disaster_recovery;
pub mod hybrid_storage;
pub mod memmap_storage;
pub mod replicated_storage;
pub mod rocksdb_storage;
pub mod secure_storage;
pub mod svdb_storage;
pub mod transaction;

// Re-export commonly used types and traits
pub use blockchain_storage::*;
pub use disaster_recovery::*;
pub use hybrid_storage::*;
pub use memmap_storage::*;
pub use replicated_storage::*;
pub use rocksdb_storage::*;
pub use secure_storage::*;
pub use svdb_storage::*;
pub use transaction::*;
