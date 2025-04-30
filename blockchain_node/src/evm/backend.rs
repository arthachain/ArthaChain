use crate::storage::{HybridStorage, Storage};
use crate::evm::types::{EvmAddress, EvmError};
use anyhow::{Result, anyhow};
use ethereum_types::{H160, H256, U256};
use primitive_types::Bytes;
use std::sync::Arc;
use log::{debug, error};

/// Adapter between SputnikVM and our storage system
pub struct EvmBackend {
    /// Reference to our hybrid storage
    storage: Arc<HybridStorage>,
    /// Cache for account data
    account_cache: std::collections::HashMap<EvmAddress, EvmAccount>,
    /// Cache for storage data
    storage_cache: std::collections::HashMap<(EvmAddress, H256), H256>,
    /// Cache for contract code
    code_cache: std::collections::HashMap<EvmAddress, Bytes>,
}

/// EVM account structure
#[derive(Clone, Debug, Default)]
pub struct EvmAccount {
    /// Account nonce
    pub nonce: U256,
    /// Account balance
    pub balance: U256,
    /// Account storage root
    pub storage_root: H256,
    /// Account code hash
    pub code_hash: H256,
}

impl EvmBackend {
    /// Create a new EVM backend adapter
    pub fn new(storage: Arc<HybridStorage>) -> Self {
        Self {
            storage,
            account_cache: std::collections::HashMap::new(),
            storage_cache: std::collections::HashMap::new(),
            code_cache: std::collections::HashMap::new(),
        }
    }
    
    /// Get storage key for an EVM account
    fn account_key(address: &EvmAddress) -> String {
        format!("evm:account:{}", hex::encode(address.as_bytes()))
    }
    
    /// Get storage key for EVM contract code
    fn code_key(address: &EvmAddress) -> String {
        format!("evm:code:{}", hex::encode(address.as_bytes()))
    }
    
    /// Get storage key for EVM contract storage
    fn storage_key(address: &EvmAddress, index: &H256) -> String {
        format!("evm:storage:{}:{}", hex::encode(address.as_bytes()), hex::encode(index.as_bytes()))
    }
    
    /// Get account data from storage
    pub async fn get_account(&mut self, address: &EvmAddress) -> Result<EvmAccount, EvmError> {
        // Check cache first
        if let Some(account) = self.account_cache.get(address) {
            return Ok(account.clone());
        }
        
        // Try to get from storage
        let key = Self::account_key(address);
        match self.storage.retrieve(&key).await {
            Ok(data) => {
                // Deserialize account data
                let account: EvmAccount = bincode::deserialize(&data)
                    .map_err(|e| EvmError::StorageError(format!("Failed to deserialize account: {}", e)))?;
                
                // Update cache
                self.account_cache.insert(*address, account.clone());
                Ok(account)
            },
            Err(_) => {
                // Account doesn't exist, create a new empty one
                let account = EvmAccount::default();
                self.account_cache.insert(*address, account.clone());
                Ok(account)
            }
        }
    }
    
    /// Update account data in storage
    pub async fn update_account(&mut self, address: EvmAddress, account: EvmAccount) -> Result<(), EvmError> {
        // Update cache
        self.account_cache.insert(address, account.clone());
        
        // Serialize and store
        let key = Self::account_key(&address);
        let data = bincode::serialize(&account)
            .map_err(|e| EvmError::StorageError(format!("Failed to serialize account: {}", e)))?;
        
        self.storage.store(&key, &data).await
            .map_err(|e| EvmError::StorageError(format!("Failed to store account: {}", e)))?;
        
        Ok(())
    }
    
    /// Get storage value
    pub async fn get_storage(&mut self, address: &EvmAddress, index: &H256) -> Result<H256, EvmError> {
        // Check cache first
        if let Some(value) = self.storage_cache.get(&(*address, *index)) {
            return Ok(*value);
        }
        
        // Try to get from storage
        let key = Self::storage_key(address, index);
        match self.storage.retrieve(&key).await {
            Ok(data) => {
                if data.len() == 32 {
                    let mut value = H256::zero();
                    value.as_bytes_mut().copy_from_slice(&data);
                    
                    // Update cache
                    self.storage_cache.insert((*address, *index), value);
                    Ok(value)
                } else {
                    Err(EvmError::StorageError("Invalid storage value size".to_string()))
                }
            },
            Err(_) => {
                // Storage slot doesn't exist, return zero
                let value = H256::zero();
                self.storage_cache.insert((*address, *index), value);
                Ok(value)
            }
        }
    }
    
    /// Set storage value
    pub async fn set_storage(&mut self, address: &EvmAddress, index: &H256, value: &H256) -> Result<(), EvmError> {
        // Update cache
        self.storage_cache.insert((*address, *index), *value);
        
        // Store in storage
        let key = Self::storage_key(address, index);
        self.storage.store(&key, value.as_bytes()).await
            .map_err(|e| EvmError::StorageError(format!("Failed to store value: {}", e)))?;
        
        Ok(())
    }
    
    /// Get contract code
    pub async fn get_code(&mut self, address: &EvmAddress) -> Result<Bytes, EvmError> {
        // Check cache first
        if let Some(code) = self.code_cache.get(address) {
            return Ok(code.clone());
        }
        
        // Try to get from storage
        let key = Self::code_key(address);
        match self.storage.retrieve(&key).await {
            Ok(data) => {
                let code = Bytes::from(data);
                
                // Update cache
                self.code_cache.insert(*address, code.clone());
                Ok(code)
            },
            Err(_) => {
                // No code, return empty bytes
                let code = Bytes::new();
                self.code_cache.insert(*address, code.clone());
                Ok(code)
            }
        }
    }
    
    /// Set contract code
    pub async fn set_code(&mut self, address: &EvmAddress, code: &[u8]) -> Result<(), EvmError> {
        // Update cache
        self.code_cache.insert(*address, Bytes::from(code.to_vec()));
        
        // Store in storage
        let key = Self::code_key(address);
        self.storage.store(&key, code).await
            .map_err(|e| EvmError::StorageError(format!("Failed to store code: {}", e)))?;
        
        Ok(())
    }
    
    /// Commit changes to storage
    pub async fn commit(&mut self) -> Result<(), EvmError> {
        // This would be called at the end of a transaction execution
        // In a more sophisticated implementation, this would be part of a larger
        // state transition that could be rolled back if needed
        debug!("Committing EVM state changes");
        
        // In a real implementation, we could optimize this to only store changes
        Ok(())
    }
    
    /// Clear caches
    pub fn clear_caches(&mut self) {
        self.account_cache.clear();
        self.storage_cache.clear();
        self.code_cache.clear();
    }
} 