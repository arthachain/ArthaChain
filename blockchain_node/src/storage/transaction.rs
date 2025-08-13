use crate::types::Hash;

/// Transaction for atomic operations across storage systems
pub struct StorageTransaction {
    operations: Vec<StorageOperation>,
    committed: bool,
}

/// Represents a storage operation
enum StorageOperation {
    Store { data: Vec<u8> },
    Delete { hash: Hash },
}

impl StorageTransaction {
    /// Create a new storage transaction
    pub fn new() -> Self {
        Self {
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
    pub fn delete(&mut self, hash: Hash) -> &mut Self {
        self.operations.push(StorageOperation::Delete { hash });
        self
    }

    /// Get number of operations
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if transaction is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Mark transaction as committed (for testing)
    pub fn mark_committed(&mut self) {
        self.committed = true;
    }
}

impl Drop for StorageTransaction {
    fn drop(&mut self) {
        if !self.committed && !self.operations.is_empty() {
            log::warn!(
                "Storage transaction dropped without commit ({} operations)",
                self.operations.len()
            );
        }
    }
}
