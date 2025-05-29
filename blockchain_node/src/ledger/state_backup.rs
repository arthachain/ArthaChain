use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;
use tokio::sync::RwLock;

use crate::types::{AccountId, BlockHeight};

/// Storage interface for blockchain state
pub trait StateStorage {
    /// Get account balance
    fn get_balance(&self, account: &AccountId) -> Result<u64>;
    
    /// Set account balance
    fn set_balance(&mut self, account: &AccountId, balance: u64) -> Result<()>;
    
    /// Get account nonce
    fn get_nonce(&self, account: &AccountId) -> Result<u64>;
    
    /// Set account nonce
    fn set_nonce(&mut self, account: &AccountId, nonce: u64) -> Result<()>;
    
    /// Get account storage value
    fn get_storage(&self, account: &AccountId, key: &[u8]) -> Result<Option<Vec<u8>>>;
    
    /// Set account storage value
    fn set_storage(&mut self, account: &AccountId, key: &[u8], value: &[u8]) -> Result<()>;
    
    /// Delete account storage value
    fn delete_storage(&mut self, account: &AccountId, key: &[u8]) -> Result<()>;
    
    /// Get account code
    fn get_code(&self, account: &AccountId) -> Result<Option<Vec<u8>>>;
    
    /// Set account code
    fn set_code(&mut self, account: &AccountId, code: &[u8]) -> Result<()>;
    
    /// Commit changes to state
    fn commit(&mut self) -> Result<BlockHeight>;
    
    /// Rollback changes
    fn rollback(&mut self) -> Result<()>;
}

/// Simple in-memory state storage
pub struct MemoryStateStorage {
    /// Account balances
    balances: HashMap<AccountId, u64>,
    /// Account nonces
    nonces: HashMap<AccountId, u64>,
    /// Account storage
    storage: HashMap<AccountId, HashMap<Vec<u8>, Vec<u8>>>,
    /// Account code
    code: HashMap<AccountId, Vec<u8>>,
    /// Current block height
    height: BlockHeight,
}

impl MemoryStateStorage {
    /// Create a new memory state storage
    pub fn new() -> Self {
        Self {
            balances: HashMap::new(),
            nonces: HashMap::new(),
            storage: HashMap::new(),
            code: HashMap::new(),
            height: 0,
        }
    }
}

impl StateStorage for MemoryStateStorage {
    fn get_balance(&self, account: &AccountId) -> Result<u64> {
        Ok(*self.balances.get(account).unwrap_or(&0))
    }
    
    fn set_balance(&mut self, account: &AccountId, balance: u64) -> Result<()> {
        self.balances.insert(account.clone(), balance);
        Ok(())
    }
    
    fn get_nonce(&self, account: &AccountId) -> Result<u64> {
        Ok(*self.nonces.get(account).unwrap_or(&0))
    }
    
    fn set_nonce(&mut self, account: &AccountId, nonce: u64) -> Result<()> {
        self.nonces.insert(account.clone(), nonce);
        Ok(())
    }
    
    fn get_storage(&self, account: &AccountId, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some(account_storage) = self.storage.get(account) {
            Ok(account_storage.get(&key.to_vec()).cloned())
        } else {
            Ok(None)
        }
    }
    
    fn set_storage(&mut self, account: &AccountId, key: &[u8], value: &[u8]) -> Result<()> {
        let account_storage = self.storage.entry(account.clone()).or_insert_with(HashMap::new);
        account_storage.insert(key.to_vec(), value.to_vec());
        Ok(())
    }
    
    fn delete_storage(&mut self, account: &AccountId, key: &[u8]) -> Result<()> {
        if let Some(account_storage) = self.storage.get_mut(account) {
            account_storage.remove(&key.to_vec());
        }
        Ok(())
    }
    
    fn get_code(&self, account: &AccountId) -> Result<Option<Vec<u8>>> {
        Ok(self.code.get(account).cloned())
    }
    
    fn set_code(&mut self, account: &AccountId, code: &[u8]) -> Result<()> {
        self.code.insert(account.clone(), code.to_vec());
        Ok(())
    }
    
    fn commit(&mut self) -> Result<BlockHeight> {
        self.height += 1;
        Ok(self.height)
    }
    
    fn rollback(&mut self) -> Result<()> {
        // In a real implementation, this would revert to the last commit
        // For this simple implementation, we do nothing
        Ok(())
    }
}

/// Merkle tree for state proofs
pub struct StateTree {
    /// Root of the state tree
    root: Vec<u8>,
}

impl StateTree {
    /// Create a new state tree
    pub fn new() -> Self {
        Self {
            root: vec![0; 32], // Empty root hash
        }
    }
    
    /// Get the root hash
    pub fn root_hash(&self) -> &[u8] {
        &self.root
    }
    
    /// Update the tree with a new key-value pair
    pub fn update(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        // In a real implementation, this would update the Merkle tree
        // For this simple implementation, we just return Ok
        Ok(())
    }
    
    /// Generate a proof for a key
    pub fn generate_proof(&self, key: &[u8]) -> Result<Vec<Vec<u8>>> {
        // In a real implementation, this would generate a Merkle proof
        // For this simple implementation, we return an empty proof
        Ok(Vec::new())
    }
    
    /// Verify a proof for a key-value pair
    pub fn verify_proof(
        root_hash: &[u8],
        key: &[u8],
        value: &[u8],
        proof: &[Vec<u8>],
    ) -> Result<bool> {
        // In a real implementation, this would verify the Merkle proof
        // For this simple implementation, we return true
        Ok(true)
    }
}

/// Thread-safe state storage wrapper
pub struct SafeStateStorage {
    /// Internal state storage
    storage: Arc<RwLock<Box<dyn StateStorage + Send + Sync>>>,
}

impl SafeStateStorage {
    /// Create a new safe state storage
    pub fn new(storage: Box<dyn StateStorage + Send + Sync>) -> Self {
        Self {
            storage: Arc::new(RwLock::new(storage)),
        }
    }
    
    /// Get a read-only view of the storage
    pub async fn read(&self) -> tokio::sync::RwLockReadGuard<Box<dyn StateStorage + Send + Sync>> {
        self.storage.read().await
    }
    
    /// Get a writable view of the storage
    pub async fn write(&self) -> tokio::sync::RwLockWriteGuard<Box<dyn StateStorage + Send + Sync>> {
        self.storage.write().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_state_storage() {
        let mut storage = MemoryStateStorage::new();
        
        // Test balance operations
        let account = AccountId::from("test_account");
        storage.set_balance(&account, 100).unwrap();
        assert_eq!(storage.get_balance(&account).unwrap(), 100);
        
        // Test nonce operations
        storage.set_nonce(&account, 5).unwrap();
        assert_eq!(storage.get_nonce(&account).unwrap(), 5);
        
        // Test storage operations
        let key = b"test_key".to_vec();
        let value = b"test_value".to_vec();
        storage.set_storage(&account, &key, &value).unwrap();
        assert_eq!(
            storage.get_storage(&account, &key).unwrap().unwrap(),
            value
        );
        
        // Test code operations
        let code = b"test_code".to_vec();
        storage.set_code(&account, &code).unwrap();
        assert_eq!(storage.get_code(&account).unwrap().unwrap(), code);
        
        // Test commit
        let height = storage.commit().unwrap();
        assert_eq!(height, 1);
    }
} 