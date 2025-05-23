use thiserror::Error;

pub mod hash;
pub mod keys;
pub mod signature;

// Re-exports
pub use hash::Hash;
pub use keys::{KeyPair, PrivateKey, PublicKey};
pub use signature::{sign, verify, Signature};

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("Invalid signature")]
    InvalidSignature,
    #[error("Invalid key")]
    InvalidKey,
    #[error("Signing error: {0}")]
    SigningError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash() {
        let data = b"test data";
        let hash = Hash::from_data(data);
        assert_eq!(hash.as_bytes().len(), 32);
    }

    #[test]
    fn test_signature() {
        let keypair = KeyPair::generate().unwrap();
        let message = b"test message";
        let signature = sign(message, &keypair).unwrap();
        assert!(verify(message, &signature, &keypair.public).unwrap());
    }
}
