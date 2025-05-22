use crate::types::Address;
use anyhow::{anyhow, Context, Result};
use blake2::{Blake2b512, Digest as Blake2Digest};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use hex;
use rand::{rngs::OsRng, Rng};
use rand_core::RngCore;
use secp256k1::ecdsa::RecoveryId;
use secp256k1::{Message, Secp256k1};
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fmt;
use tiny_keccak::{Hasher, Keccak};

/// Hash representation for cryptographic hashes
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Hash([u8; 32]);

impl Hash {
    /// Create a new hash from raw bytes
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get raw bytes of the hash
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Create a Hash from a byte slice
    /// Returns error if slice length is not 32 bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 32 {
            return Err(anyhow!(
                "Invalid hash length. Expected 32 bytes, got {}",
                bytes.len()
            ));
        }
        let mut hash_bytes = [0u8; 32];
        hash_bytes.copy_from_slice(bytes);
        Ok(Self(hash_bytes))
    }

    /// Create a Hash from a hex string
    pub fn from_hex(hex_str: &str) -> Result<Self> {
        if hex_str.len() != 64 {
            return Err(anyhow!(
                "Invalid hash hex length. Expected 64 chars, got {}",
                hex_str.len()
            ));
        }
        let bytes = hex::decode(hex_str)?;
        Self::from_bytes(&bytes)
    }

    /// Convert to hexadecimal string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl Default for Hash {
    fn default() -> Self {
        Self([0u8; 32])
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl AsRef<[u8]> for Hash {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl std::convert::TryFrom<Vec<u8>> for Hash {
    type Error = anyhow::Error;

    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        Self::from_bytes(&bytes)
    }
}

/// Calculate SHA-256 hash of data
pub fn hash(data: &[u8]) -> Hash {
    let mut hasher = Blake2b512::new();
    hasher.update(data);
    let result = hasher.finalize();
    Hash::new([
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7],
        result[8], result[9], result[10], result[11], result[12], result[13], result[14],
        result[15], result[16], result[17], result[18], result[19], result[20], result[21],
        result[22], result[23], result[24], result[25], result[26], result[27], result[28],
        result[29], result[30], result[31],
    ])
}

/// Generate random bytes
pub fn random_bytes(len: usize) -> Vec<u8> {
    generate_random_bytes(len)
}

/// Generate a random keypair for a new node
pub fn generate_keypair() -> Result<(Vec<u8>, Vec<u8>)> {
    // Generate a random private key
    let mut private_key = [0u8; 32];
    let mut csprng = OsRng {};
    csprng.fill_bytes(&mut private_key);

    // Create signing key and get verifying key
    let signing_key = SigningKey::from_bytes(&private_key);
    let verifying_key = signing_key.verifying_key();

    Ok((
        signing_key.to_bytes().to_vec(),
        verifying_key.to_bytes().to_vec(),
    ))
}

/// Validate signature
pub fn validate_signature(address: &Address, data: &[u8], signature: &[u8]) -> Result<bool> {
    // Convert address to public key bytes
    let mut key_bytes = [0u8; 32];
    key_bytes[..20].copy_from_slice(address.as_bytes());
    key_bytes[20..].fill(0);

    let verifying_key = VerifyingKey::from_bytes(&key_bytes)?;
    let sig = ed25519_dalek::Signature::from_slice(signature)?;
    Ok(verifying_key.verify(data, &sig).is_ok())
}

/// Sign data with a private key
pub fn sign(private_key: &[u8], data: &[u8]) -> Result<Vec<u8>> {
    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(&private_key[..32]);
    let signing_key = SigningKey::from_bytes(&key_bytes);
    let signature = signing_key.sign(data);
    Ok(signature.to_bytes().to_vec())
}

/// Verify a signature with a public key
pub fn verify(public_key: &[u8], data: &[u8], signature: &[u8]) -> Result<bool> {
    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(&public_key[..32]);
    let verifying_key = VerifyingKey::from_bytes(&key_bytes)?;
    let sig = ed25519_dalek::Signature::from_slice(signature)?;
    Ok(verifying_key.verify(data, &sig).is_ok())
}

/// Calculate a BLAKE2b hash of data
pub fn hash_to_bytes(data: &[u8]) -> Vec<u8> {
    let mut hasher = Blake2b512::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Calculate a BLAKE2b hash as a hex string
pub fn hash_hex(data: &[u8]) -> String {
    hex::encode(hash_to_bytes(data))
}

/// Compare a hash with data
pub fn verify_hash(data: &[u8], hash: &[u8]) -> bool {
    let data_hash = hash_to_bytes(data);
    data_hash == hash
}

/// Generate a random 32-byte value for nonce
pub fn generate_random_bytes(size: usize) -> Vec<u8> {
    let mut rng = OsRng {};
    let mut bytes = vec![0u8; size];
    rand::RngCore::fill_bytes(&mut rng, &mut bytes);
    bytes
}

/// Generate a random value from 0 to max-1
pub fn generate_random_number(max: u64) -> u64 {
    let mut rng = OsRng {};
    rng.gen_range(0..max)
}

/// Sign data with an ed25519 private key
pub fn ed25519_sign(private_key_bytes: &[u8], data: &[u8]) -> Result<Vec<u8>> {
    if private_key_bytes.len() != 32 {
        return Err(anyhow!("Invalid private key length"));
    }

    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(private_key_bytes);
    let signing_key = SigningKey::from_bytes(&key_bytes);
    let signature = signing_key.sign(data);
    Ok(signature.to_bytes().to_vec())
}

/// Verify a signature with ed25519
pub fn ed25519_verify(
    public_key_bytes: &[u8],
    data: &[u8],
    signature_bytes: &[u8],
) -> Result<bool> {
    if public_key_bytes.len() != 32 {
        return Err(anyhow!("Invalid public key length"));
    }

    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(public_key_bytes);
    let verifying_key = VerifyingKey::from_bytes(&key_bytes)?;
    let sig = ed25519_dalek::Signature::from_slice(signature_bytes)?;
    Ok(verifying_key.verify(data, &sig).is_ok())
}

/// Derive an address from a private key
pub fn derive_address_from_private_key(private_key_bytes: &[u8]) -> Result<crate::types::Address> {
    if private_key_bytes.len() != 32 {
        return Err(anyhow!("Invalid private key length"));
    }

    // Convert bytes to array for SigningKey
    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(private_key_bytes);

    // Create signing key from private key
    let signing_key = SigningKey::from_bytes(&key_bytes);

    // Get the public key
    let public_key = signing_key.verifying_key().to_bytes();

    // Hash the public key to get the address
    let hash_result = hash(&public_key);
    let address_bytes = hash_result.as_bytes();

    // Create a new Address from the bytes
    let mut addr_bytes = [0u8; 20];
    addr_bytes.copy_from_slice(&address_bytes[0..20]);

    let address = crate::types::Address::new(addr_bytes);

    Ok(address)
}

/// Sign data with the private key
pub fn sign_data(private_key: &[u8], data: &[u8]) -> Result<Vec<u8>> {
    ed25519_sign(private_key, data)
}

/// Verify signature against an address
pub fn verify_signature(public_key: &[u8], message: &[u8], signature: &[u8]) -> Result<bool> {
    let verifying_key = VerifyingKey::from_bytes(
        public_key
            .try_into()
            .map_err(|_| anyhow!("Invalid public key length"))?,
    )?;
    let sig = Signature::from_bytes(
        signature
            .try_into()
            .map_err(|_| anyhow!("Invalid signature length"))?,
    );

    Ok(verifying_key.verify(message, &sig).is_ok())
}

pub fn recover_address_from_signature(
    message: &[u8],
    signature: &[u8],
) -> Result<crate::types::Address> {
    if signature.len() != 65 {
        return Err(anyhow!(
            "Invalid signature length. Expected 65 bytes, got {}",
            signature.len()
        ));
    }

    // Extract the recovery id (the last byte)
    let recovery_id = signature[64];
    let recovery_id_int = recovery_id as i32 - 27; // Adjust for Ethereum's encoding

    if recovery_id_int < 0 || recovery_id_int > 3 {
        return Err(anyhow!("Invalid recovery ID: {}", recovery_id));
    }

    // The actual signature is the first 64 bytes
    let sig_bytes = &signature[0..64];

    // Hash the message using Keccak-256
    let message_hash = keccak256(message);

    // Create the recoverable signature
    let recovery_id =
        RecoveryId::from_i32(recovery_id_int).context("Failed to create recovery ID")?;

    let mut recovered_sig = [0u8; 64];
    recovered_sig.copy_from_slice(sig_bytes);

    let secp = Secp256k1::new();
    let message = Message::from_digest_slice(&message_hash).context("Failed to create message")?;

    // Create a recoverable signature
    let recoverable_sig =
        secp256k1::ecdsa::RecoverableSignature::from_compact(&recovered_sig, recovery_id)
            .context("Failed to create recoverable signature")?;

    // Recover the public key
    let pubkey = secp
        .recover_ecdsa(&message, &recoverable_sig)
        .context("Failed to recover public key")?;

    // Convert to uncompressed form
    let pubkey_serialized = pubkey.serialize_uncompressed();

    // Skip the first byte (0x04 which indicates uncompressed)
    let pubkey_bytes = &pubkey_serialized[1..];

    // Take the keccak hash of the public key and keep the last 20 bytes
    let address_bytes = keccak256(pubkey_bytes);

    // Create a new Address from the last 20 bytes
    let mut addr_bytes = [0u8; 20];
    addr_bytes.copy_from_slice(&address_bytes[12..32]);

    let address = crate::types::Address::new(addr_bytes);

    Ok(address)
}

/// Calculate Keccak-256 hash of data
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut keccak = Keccak::v256();
    let mut result = [0u8; 32];
    keccak.update(data);
    keccak.finalize(&mut result);
    result
}

/// Sign message with private key
pub fn sign_message(private_key: &[u8], message: &[u8]) -> Result<Vec<u8>> {
    let signing_key = SigningKey::from_bytes(
        private_key
            .try_into()
            .map_err(|_| anyhow!("Invalid private key length"))?,
    );
    let signature = signing_key.sign(message);
    Ok(signature.to_bytes().to_vec())
}

/// Verify address signature
pub fn verify_address_signature(
    address: &Address,
    data: &[u8],
    signature_bytes: &[u8],
) -> Result<bool> {
    let mut key_bytes = [0u8; 32];
    key_bytes[..20].copy_from_slice(address.as_bytes());
    key_bytes[20..].fill(0);

    let verifying_key = VerifyingKey::from_bytes(&key_bytes)?;
    let sig = ed25519_dalek::Signature::from_slice(signature_bytes)?;
    Ok(verifying_key.verify(data, &sig).is_ok())
}
