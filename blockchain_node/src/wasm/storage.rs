//! WASM contract storage interface
//!
//! Provides storage access for WASM contracts with key-value semantics.
//! Uses a prefixed namespace for each contract to isolate storage between contracts.

use std::sync::Arc;
use parking_lot::RwLock;
use crate::storage::Storage;
use crate::types::Address;
use crate::wasm::types::WasmError;

/// Storage interface for the WASM environment
pub struct WasmStorage {
    /// The underlying storage implementation
    storage: Arc<RwLock<dyn Storage>>,
}

impl WasmStorage {
    /// Create a new WASM storage interface with the given storage backend
    pub fn new(storage: Arc<RwLock<dyn Storage>>) -> Self {
        Self { storage }
    }

    /// Get a storage value for a contract
    pub fn get(&self, contract_address: &Address, key: &[u8]) -> Option<Vec<u8>> {
        let prefixed_key = self.prefixed_key(contract_address, key);
        self.storage.read().get(&prefixed_key)
    }

    /// Set a storage value for a contract
    pub fn set(&self, contract_address: &Address, key: &[u8], value: &[u8]) {
        let prefixed_key = self.prefixed_key(contract_address, key);
        self.storage.write().set(&prefixed_key, value);
    }

    /// Delete a storage value for a contract
    pub fn delete(&self, contract_address: &Address, key: &[u8]) {
        let prefixed_key = self.prefixed_key(contract_address, key);
        self.storage.write().delete(&prefixed_key);
    }

    /// Check if a key exists in storage for a contract
    pub fn has(&self, contract_address: &Address, key: &[u8]) -> bool {
        let prefixed_key = self.prefixed_key(contract_address, key);
        self.storage.read().has(&prefixed_key)
    }

    /// Create a deterministic storage key prefix for a contract to isolate storage
    fn prefixed_key(&self, contract_address: &Address, key: &[u8]) -> Vec<u8> {
        let mut prefixed_key = Vec::with_capacity(contract_address.as_bytes().len() + 1 + key.len());
        prefixed_key.extend_from_slice(contract_address.as_bytes());
        prefixed_key.push(b':');
        prefixed_key.extend_from_slice(key);
        prefixed_key
    }

    /// Get the code for a contract
    pub fn get_code(&self, contract_address: &Address) -> Option<Vec<u8>> {
        let code_key = self.code_key(contract_address);
        self.storage.read().get(&code_key)
    }

    /// Set the code for a contract
    pub fn set_code(&self, contract_address: &Address, code: &[u8]) {
        let code_key = self.code_key(contract_address);
        self.storage.write().set(&code_key, code);
    }

    /// Get the metadata for a contract
    pub fn get_metadata(&self, contract_address: &Address) -> Option<Vec<u8>> {
        let metadata_key = self.metadata_key(contract_address);
        self.storage.read().get(&metadata_key)
    }

    /// Set the metadata for a contract
    pub fn set_metadata(&self, contract_address: &Address, metadata: &[u8]) {
        let metadata_key = self.metadata_key(contract_address);
        self.storage.write().set(&metadata_key, metadata);
    }

    /// Create the key for storing contract code
    fn code_key(&self, contract_address: &Address) -> Vec<u8> {
        let mut code_key = Vec::with_capacity(contract_address.as_bytes().len() + 6);
        code_key.extend_from_slice(b"code:");
        code_key.extend_from_slice(contract_address.as_bytes());
        code_key
    }

    /// Create the key for storing contract metadata
    fn metadata_key(&self, contract_address: &Address) -> Vec<u8> {
        let mut metadata_key = Vec::with_capacity(contract_address.as_bytes().len() + 9);
        metadata_key.extend_from_slice(b"metadata:");
        metadata_key.extend_from_slice(contract_address.as_bytes());
        metadata_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::MemoryStorage;

    #[test]
    fn test_wasm_storage() {
        let storage = Arc::new(MemoryStorage::new());
        let contract_address = Address::from_hex("0x1234567890123456789012345678901234567890").unwrap();
        let wasm_storage = WasmStorage::new(storage);

        // Test write and read
        let key = b"test_key";
        let value = b"test_value";
        wasm_storage.set(&contract_address, key, value);
        let read_value = wasm_storage.get(&contract_address, key).unwrap();
        assert_eq!(read_value, value.to_vec());

        // Test delete
        wasm_storage.delete(&contract_address, key);
        let read_value = wasm_storage.get(&contract_address, key);
        assert_eq!(read_value, None);
    }

    #[test]
    fn test_storage_namespacing() {
        let storage = Arc::new(MemoryStorage::new());
        let contract_address1 = Address::from_hex("0x1234567890123456789012345678901234567890").unwrap();
        let contract_address2 = Address::from_hex("0x0987654321098765432109876543210987654321").unwrap();
        let wasm_storage1 = WasmStorage::new(storage.clone());
        let wasm_storage2 = WasmStorage::new(storage);

        // Write to both storages with the same key
        let key = b"test_key";
        let value1 = b"test_value1";
        let value2 = b"test_value2";
        wasm_storage1.set(&contract_address1, key, value1);
        wasm_storage2.set(&contract_address2, key, value2);

        // Check that they have different values
        let read_value1 = wasm_storage1.get(&contract_address1, key).unwrap();
        let read_value2 = wasm_storage2.get(&contract_address2, key).unwrap();
        assert_eq!(read_value1, value1.to_vec());
        assert_eq!(read_value2, value2.to_vec());
    }
} 