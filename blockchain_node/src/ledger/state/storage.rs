use std::collections::HashMap;
use std::sync::RwLock;
use anyhow::Result;

/// State Storage for persisting blockchain state
#[derive(Debug)]
pub struct StateStorage {
    /// Storage data
    storage: RwLock<HashMap<String, Vec<u8>>>,
}

impl StateStorage {
    /// Create a new state storage
    pub fn new() -> Self {
        Self {
            storage: RwLock::new(HashMap::new()),
        }
    }

    /// Get a value from storage
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let storage = self.storage.read().unwrap();
        Ok(storage.get(key).cloned())
    }

    /// Set a value in storage
    pub fn set(&self, key: &str, value: Vec<u8>) -> Result<()> {
        let mut storage = self.storage.write().unwrap();
        storage.insert(key.to_string(), value);
        Ok(())
    }

    /// Delete a value from storage
    pub fn delete(&self, key: &str) -> Result<()> {
        let mut storage = self.storage.write().unwrap();
        storage.remove(key);
        Ok(())
    }

    /// Check if a key exists
    pub fn has(&self, key: &str) -> Result<bool> {
        let storage = self.storage.read().unwrap();
        Ok(storage.contains_key(key))
    }
} 