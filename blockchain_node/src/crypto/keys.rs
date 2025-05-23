use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KeyError {
    #[error("Invalid key data")]
    InvalidData,
    #[error("Key generation error")]
    GenerationError,
}

/// Public key for signatures
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicKey(Vec<u8>);

impl PublicKey {
    /// Create a new public key from bytes
    pub fn new(data: Vec<u8>) -> Self {
        PublicKey(data)
    }

    /// Get the raw bytes of the public key
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Private key for signatures
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivateKey(Vec<u8>);

impl PrivateKey {
    /// Create a new private key from bytes
    pub fn new(data: Vec<u8>) -> Self {
        PrivateKey(data)
    }

    /// Get the raw bytes of the private key
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Key pair for signing and verifying
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyPair {
    /// Public key
    pub public: PublicKey,
    /// Private key
    pub private: PrivateKey,
}

impl KeyPair {
    /// Create a new key pair
    pub fn new(public: PublicKey, private: PrivateKey) -> Self {
        Self { public, private }
    }

    /// Generate a new key pair
    pub fn generate() -> Result<Self, KeyError> {
        // Generate ed25519 key
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        // Convert to our key types
        let public = PublicKey::new(verifying_key.to_bytes().to_vec());
        let private = PrivateKey::new(signing_key.to_bytes().to_vec());

        Ok(Self::new(public, private))
    }

    /// Get the public key
    pub fn public_key(&self) -> &PublicKey {
        &self.public
    }

    /// Get the private key
    pub fn private_key(&self) -> &PrivateKey {
        &self.private
    }
}
