use std::path::Path;
use async_trait::async_trait;
use thiserror::Error;
use crate::types::Hash;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Storage error: {0}")]
    Other(String),
}

impl From<String> for StorageError {
    fn from(s: String) -> Self {
        StorageError::Other(s)
    }
}

pub type Result<T> = std::result::Result<T, StorageError>;

/// Trait for storage operations
#[async_trait]
pub trait Storage: Send + Sync {
    async fn store(&self, data: &[u8]) -> Result<Hash>;
    async fn retrieve(&self, hash: &Hash) -> Result<Option<Vec<u8>>>;
    async fn exists(&self, hash: &Hash) -> Result<bool>;
    async fn delete(&self, hash: &Hash) -> Result<()>;
    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool>;
    async fn close(&self) -> Result<()>;
}

/// Trait for storage initialization
#[async_trait]
pub trait StorageInit: Send + Sync {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()>;
}

// Helper functions for storage operations
pub async fn store_data<S: Storage + ?Sized>(storage: &S, data: &[u8]) -> Result<Hash> {
    storage.store(data).await
}

pub async fn retrieve_data<S: Storage + ?Sized>(storage: &S, hash: &Hash) -> Result<Option<Vec<u8>>> {
    storage.retrieve(hash).await
}

pub async fn exists_data<S: Storage + ?Sized>(storage: &S, hash: &Hash) -> Result<bool> {
    storage.exists(hash).await
}

pub async fn delete_data<S: Storage + ?Sized>(storage: &S, hash: &Hash) -> Result<()> {
    storage.delete(hash).await
}

pub async fn verify_data<S: Storage + ?Sized>(storage: &S, hash: &Hash, data: &[u8]) -> Result<bool> {
    storage.verify(hash, data).await
}

// ... rest of the file remains unchanged ... 