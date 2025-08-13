use crate::types::Address;
use crate::utils::crypto;
use anyhow::Result;
use log::debug;
use std::path::Path;

/// Identity manager for blockchain nodes
pub struct IdentityManager {
    /// Node ID string
    pub node_id: String,
    /// Node address
    pub address: Address,
    /// Private key data
    private_key: Vec<u8>,
}

impl IdentityManager {
    /// Create a new identity manager
    pub fn new(node_id: &str, private_key: Vec<u8>) -> Result<Self> {
        // Generate address from private key
        let address = crypto::derive_address_from_private_key(&private_key)?;

        debug!("Identity created for node {}", node_id);

        Ok(Self {
            node_id: node_id.to_string(),
            address: Address(
                hex::decode(&address)
                    .unwrap_or_default()
                    .try_into()
                    .unwrap_or_default(),
            ),
            private_key,
        })
    }

    /// Load identity from a file
    pub fn load_from_file(node_id: &str, key_path: &Path) -> Result<Self> {
        // Load private key from file
        let private_key = std::fs::read(key_path)?;
        Self::new(node_id, private_key)
    }

    /// Sign data with the identity private key
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>> {
        crypto::sign_data(&self.private_key, data)
    }

    /// Verify signature
    pub fn verify(&self, data: &[u8], signature: &[u8]) -> Result<bool> {
        crypto::verify_signature(&hex::encode(self.address.0), data, signature)
    }
}
