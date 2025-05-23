use ed25519_dalek::{Signature as Ed25519Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::keys::{KeyPair, PublicKey};

#[derive(Debug, Error)]
pub enum SignatureError {
    #[error("Invalid signature data")]
    InvalidData,
    #[error("Invalid key")]
    InvalidKey,
    #[error("Signature verification failed")]
    VerificationFailed,
    #[error("Key generation error")]
    KeyError,
}

/// Signature type used throughout the blockchain
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature(Vec<u8>);

impl Signature {
    /// Create a new signature from bytes
    pub fn new(data: Vec<u8>) -> Self {
        Signature(data)
    }

    /// Get the raw bytes of the signature
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Sign a message using a KeyPair
pub fn sign(message: &[u8], keypair: &KeyPair) -> Result<Signature, SignatureError> {
    // Convert private key bytes to SigningKey
    let signing_key_bytes: [u8; 32] = keypair.private.as_bytes()[..32]
        .try_into()
        .map_err(|_| SignatureError::InvalidKey)?;

    let signing_key = SigningKey::from_bytes(&signing_key_bytes);

    // Sign the message
    let signature = signing_key.sign(message);

    // Return the signature
    Ok(Signature(signature.to_bytes().to_vec()))
}

/// Verify a signature using a PublicKey
pub fn verify(
    message: &[u8],
    signature: &Signature,
    public_key: &PublicKey,
) -> Result<bool, SignatureError> {
    // Convert signature to ed25519 signature
    let sig_bytes: [u8; 64] = signature.as_bytes()[..64]
        .try_into()
        .map_err(|_| SignatureError::InvalidData)?;

    let ed_signature = Ed25519Signature::from_bytes(&sig_bytes);

    // Convert public key to verifying key
    let key_bytes: [u8; 32] = public_key.as_bytes()[..32]
        .try_into()
        .map_err(|_| SignatureError::InvalidKey)?;

    let verifying_key =
        VerifyingKey::from_bytes(&key_bytes).map_err(|_| SignatureError::InvalidKey)?;

    // Verify the signature
    match verifying_key.verify(message, &ed_signature) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}
