use crate::evm::types::{EvmAddress, EvmError};
use crate::storage::Storage;
use ethereum_types::{H256, U256};
use log::debug;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Adapter between SputnikVM and our storage system
pub struct EvmBackend {
    /// Reference to our storage
    storage: Arc<dyn Storage>,
    /// Cache for account data
    account_cache: std::collections::HashMap<EvmAddress, EvmAccount>,
    /// Cache for storage data
    storage_cache: std::collections::HashMap<(EvmAddress, H256), H256>,
    /// Cache for contract code
    code_cache: std::collections::HashMap<EvmAddress, Vec<u8>>,
}

/// EVM account structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvmAccount {
    /// Account nonce
    pub nonce: u64,
    /// Account balance
    pub balance: U256,
    /// Account storage root
    pub storage_root: H256,
    /// Account code hash
    pub code_hash: H256,
    /// Account code
    pub code: Vec<u8>,
    /// Account storage
    pub storage: HashMap<H256, H256>,
}

impl EvmBackend {
    /// Create a new EVM backend adapter
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self {
            storage,
            account_cache: std::collections::HashMap::new(),
            storage_cache: std::collections::HashMap::new(),
            code_cache: std::collections::HashMap::new(),
        }
    }

    /// Get account data from storage
    pub fn get_account(&self, address: &EvmAddress) -> Result<EvmAccount, EvmError> {
        // Check cache first
        if let Some(account) = self.account_cache.get(address) {
            return Ok(account.clone());
        }

        // Generate storage key for account
        let key = format!("evm:account:{}", hex::encode(address.as_ref()));

        let fetched = futures::executor::block_on(self.storage.get(key.as_bytes()))
            .map_err(|e| EvmError::StorageError(format!("Failed to get account: {}", e)))?;
        match fetched {
            Some(data) => {
                let account: EvmAccount = bincode::deserialize(&data).map_err(|e| {
                    EvmError::StorageError(format!("Failed to deserialize account: {}", e))
                })?;
                Ok(account)
            }
            None => Ok(EvmAccount {
                nonce: 0,
                balance: U256::zero(),
                code: Vec::new(),
                storage: HashMap::new(),
                storage_root: H256::zero(),
                code_hash: H256::zero(),
            }),
        }
    }

    /// Update account data in storage
    pub fn update_account(
        &mut self,
        address: EvmAddress,
        account: EvmAccount,
    ) -> Result<(), EvmError> {
        let key = format!("evm:account:{}", hex::encode(address.as_ref()));
        let data = bincode::serialize(&account)
            .map_err(|e| EvmError::StorageError(format!("Failed to serialize account: {}", e)))?;

        futures::executor::block_on(self.storage.put(key.as_bytes(), &data))
            .map_err(|e| EvmError::StorageError(format!("Failed to store account: {}", e)))?;
        self.account_cache.insert(address, account);
        Ok(())
    }

    /// Get storage value
    pub fn get_storage(&self, address: &EvmAddress, key: H256) -> Result<H256, EvmError> {
        // Check cache first
        if let Some(&value) = self.storage_cache.get(&(*address, key)) {
            return Ok(value);
        }

        let storage_key = format!(
            "evm:storage:{}:{}",
            hex::encode(address.as_ref()),
            hex::encode(key.as_ref())
        );
        let fetched = futures::executor::block_on(self.storage.get(storage_key.as_bytes()))
            .map_err(|e| EvmError::StorageError(format!("Failed to get storage: {}", e)))?;
        match fetched {
            Some(data) => {
                let mut value = H256::zero();
                if data.len() >= 32 {
                    value
                        .as_bytes_mut()
                        .copy_from_slice(&data[data.len() - 32..]);
                } else {
                    // Left-pad with zeros if shorter than 32 bytes
                    let mut padded = vec![0u8; 32];
                    let start = 32 - data.len();
                    padded[start..].copy_from_slice(&data);
                    value.as_bytes_mut().copy_from_slice(&padded);
                }
                Ok(value)
            }
            None => Ok(H256::zero()),
        }
    }

    /// Set storage value
    pub fn set_storage(
        &mut self,
        address: &EvmAddress,
        key: H256,
        value: H256,
    ) -> Result<(), EvmError> {
        let storage_key = format!(
            "evm:storage:{}:{}",
            hex::encode(address.as_ref()),
            hex::encode(key.as_ref())
        );
        futures::executor::block_on(self.storage.put(storage_key.as_bytes(), value.as_ref()))
            .map_err(|e| EvmError::StorageError(format!("Failed to set storage: {}", e)))?;
        self.storage_cache.insert((*address, key), value);
        Ok(())
    }

    /// Get contract code
    pub fn get_code(&self, address: &EvmAddress) -> Result<Vec<u8>, EvmError> {
        // Check cache first
        if let Some(code) = self.code_cache.get(address) {
            return Ok(code.clone());
        }

        let code_key = format!("evm:code:{}", hex::encode(address.as_ref()));
        let fetched = futures::executor::block_on(self.storage.get(code_key.as_bytes()))
            .map_err(|e| EvmError::StorageError(format!("Failed to get code: {}", e)))?;
        match fetched {
            Some(code) => Ok(code),
            None => Ok(Vec::new()),
        }
    }

    /// Set contract code
    pub fn set_code(&mut self, address: &EvmAddress, code: &[u8]) -> Result<(), EvmError> {
        let code_key = format!("evm:code:{}", hex::encode(address.as_ref()));
        futures::executor::block_on(self.storage.put(code_key.as_bytes(), code))
            .map_err(|e| EvmError::StorageError(format!("Failed to set code: {}", e)))?;
        self.code_cache.insert(*address, code.to_vec());
        Ok(())
    }

    /// Commit changes to storage
    pub async fn commit(&mut self) -> Result<(), EvmError> {
        debug!("Committing EVM state changes");
        Ok(())
    }

    /// Clear caches
    pub fn clear_caches(&mut self) {
        self.account_cache.clear();
        self.storage_cache.clear();
        self.code_cache.clear();
    }
}
