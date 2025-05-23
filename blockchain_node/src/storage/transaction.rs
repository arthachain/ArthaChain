use super::Storage;
use crate::types::Hash;
use anyhow::Result;
use log::debug;

/// Transaction for atomic operations across storage systems
pub struct StorageTransaction<'a> {
    hybrid: &'a dyn Storage,
    operations: Vec<StorageOperation>,
    committed: bool,
}

/// Represents a storage operation
enum StorageOperation {
    Store { data: Vec<u8> },
    Delete { hash: Hash },
}

impl<'a> StorageTransaction<'a> {
    /// Create a new storage transaction
    pub fn new(hybrid: &'a dyn Storage) -> Self {
        Self {
            hybrid,
            operations: Vec::new(),
            committed: false,
        }
    }

    /// Add store operation to transaction
    pub fn store(&mut self, data: &[u8]) -> &mut Self {
        self.operations.push(StorageOperation::Store {
            data: data.to_vec(),
        });
        self
    }

    /// Add delete operation to transaction
    pub fn delete(&mut self, hash: &Hash) -> &mut Self {
        self.operations
            .push(StorageOperation::Delete { hash: hash.clone() });
        self
    }

    /// Commit the transaction
    pub async fn commit(mut self) -> Result<()> {
        // Perform all operations
        for op in &self.operations {
            match op {
                StorageOperation::Store { data } => {
                    self.hybrid.store(data).await?;
                }
                StorageOperation::Delete { hash } => {
                    self.hybrid.delete(hash).await?;
                }
            }
        }

        self.committed = true;
        Ok(())
    }
}

impl<'a> Drop for StorageTransaction<'a> {
    fn drop(&mut self) {
        if !self.committed && !self.operations.is_empty() {
            debug!(
                "Storage transaction dropped without commit ({} operations)",
                self.operations.len()
            );
        }
    }
}
