use anyhow::Result;
use std::collections::HashMap;
use std::sync::RwLock;

/// State Tree for managing blockchain state
#[derive(Debug)]
pub struct StateTree {
    /// State data
    state: RwLock<HashMap<String, Vec<u8>>>,
}

impl StateTree {
    /// Create a new state tree
    pub fn new() -> Self {
        Self {
            state: RwLock::new(HashMap::new()),
        }
    }

    /// Get a value from the state tree
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let state = self.state.read().unwrap();
        Ok(state.get(key).cloned())
    }

    /// Set a value in the state tree
    pub fn set(&self, key: &str, value: Vec<u8>) -> Result<()> {
        let mut state = self.state.write().unwrap();
        state.insert(key.to_string(), value);
        Ok(())
    }

    /// Delete a value from the state tree
    pub fn delete(&self, key: &str) -> Result<()> {
        let mut state = self.state.write().unwrap();
        state.remove(key);
        Ok(())
    }

    /// Check if a key exists
    pub fn has(&self, key: &str) -> Result<bool> {
        let state = self.state.read().unwrap();
        Ok(state.contains_key(key))
    }
}
