use ed25519_dalek::{Signature as Ed25519Signature, Signer, Verifier, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use thiserror::Error;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Hash([u8; 32]);

impl Hash {
    pub fn new(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        Self(hasher.finalize().into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Create a Hash by digesting the provided data
    pub fn digest(data: &[u8]) -> Self {
        Self::new(data)
    }
}

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Invalid key")]
    InvalidKey,
    #[error("Signing error: {0}")]
    SigningError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyPair {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

impl KeyPair {
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    pub fn sign(&self, message: &[u8]) -> Result<Ed25519Signature, CryptoError> {
        Ok(self.signing_key.sign(message))
    }

    pub fn verify(&self, message: &[u8], signature: &Ed25519Signature) -> Result<(), CryptoError> {
        self.verifying_key.verify(message, signature)
            .map_err(|_| CryptoError::InvalidSignature)
    }

    pub fn public_key(&self) -> VerifyingKey {
        self.verifying_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash() {
        let data = b"test data";
        let hash = Hash::new(data);
        assert_eq!(hash.as_bytes().len(), 32);
    }

    #[test]
    fn test_signature() {
        let keypair = KeyPair::generate();
        let message = b"test message";
        let signature = keypair.sign(message).unwrap();
        assert!(keypair.verify(message, &signature).is_ok());
    }
} 