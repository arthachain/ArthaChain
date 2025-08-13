use crate::crypto::keys::{PrivateKey, PublicKey};
use anyhow::{anyhow, Result};
use ed25519_dalek::{Signature as Ed25519Signature, Signer, SigningKey, Verifier, VerifyingKey};

use serde::{Deserialize, Serialize};
use std::fmt;

/// Generic signature type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature(Vec<u8>);

impl Signature {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(bytes.to_vec())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl AsRef<[u8]> for Signature {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(&self.0))
    }
}

/// Sign a message using Ed25519
pub fn sign(private_key: &PrivateKey, message: &[u8]) -> Result<Signature> {
    if private_key.as_ref().len() < 32 {
        return Err(anyhow!("Private key too short"));
    }

    let signing_key_bytes: [u8; 32] = private_key.as_ref()[..32]
        .try_into()
        .map_err(|_| anyhow!("Invalid private key length"))?;

    let signing_key = SigningKey::from_bytes(&signing_key_bytes);
    let signature = signing_key.sign(message);

    Ok(Signature::new(signature.to_bytes().to_vec()))
}

/// Verify a signature using Ed25519
pub fn verify(public_key: &PublicKey, message: &[u8], signature: &Signature) -> Result<bool> {
    if signature.as_ref().len() < 64 {
        return Err(anyhow!("Signature too short"));
    }

    if public_key.as_ref().len() < 32 {
        return Err(anyhow!("Public key too short"));
    }

    let sig_bytes: [u8; 64] = signature.as_ref()[..64]
        .try_into()
        .map_err(|_| anyhow!("Invalid signature length"))?;

    let key_bytes: [u8; 32] = public_key.as_ref()[..32]
        .try_into()
        .map_err(|_| anyhow!("Invalid public key length"))?;

    let verifying_key =
        VerifyingKey::from_bytes(&key_bytes).map_err(|e| anyhow!("Invalid public key: {}", e))?;

    let signature = Ed25519Signature::from_bytes(&sig_bytes);

    match verifying_key.verify(message, &signature) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}
