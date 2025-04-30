use std::sync::Arc;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use log::{debug, warn, error};
use sha3::{Digest, Keccak256, Sha3_256};
use ripemd::Ripemd160;
use blake3::Hasher;
use num_bigint::BigUint;
use num_traits::Zero;
use zstd::{encode_all, decode_all};
use k256::{ecdsa::{RecoveryId, Signature}, PublicKey, SecretKey};
use elliptic_curve::sec1::ToEncodedPoint;
use pairing::{bls12_381::{G1, G2, Gt}, Engine};
use blake2::{Blake2b, Digest as Blake2Digest};

use crate::evm::types::{EvmAddress, EvmError, EvmExecutionResult};
use crate::storage::Storage;
use crate::crypto::hash::Hash;

/// Precompiled contract error
#[derive(Debug, Error)]
pub enum PrecompiledError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Out of gas")]
    OutOfGas,
}

/// Precompiled contract type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecompiledType {
    /// ECDSA recovery
    EcdsaRecovery,
    /// SHA256 hash
    Sha256,
    /// RIPEMD160 hash
    Ripemd160,
    /// Identity function
    Identity,
    /// Modular exponentiation
    ModExp,
    /// EC addition
    EcAdd,
    /// EC scalar multiplication
    EcMul,
    /// EC pairing
    EcPairing,
    /// Blake2F compression
    Blake2F,
    /// ZSTD compression
    ZstdCompress,
    /// ZSTD decompression
    ZstdDecompress,
}

/// Precompiled contract configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecompiledConfig {
    /// Contract type
    pub contract_type: PrecompiledType,
    /// Base gas cost
    pub base_gas: u64,
    /// Gas cost per word
    pub word_gas: u64,
    /// Maximum input size
    pub max_input_size: usize,
}

/// Precompiled contract
pub struct PrecompiledContract {
    /// Contract configuration
    config: PrecompiledConfig,
    /// Storage interface
    storage: Arc<dyn Storage>,
}

impl PrecompiledContract {
    /// Create a new precompiled contract
    pub fn new(config: PrecompiledConfig, storage: Arc<dyn Storage>) -> Self {
        Self { config, storage }
    }

    /// Execute the precompiled contract
    pub fn execute(&self, input: &[u8], gas_limit: u64) -> Result<EvmExecutionResult, PrecompiledError> {
        // Check input size
        if input.len() > self.config.max_input_size {
            return Err(PrecompiledError::InvalidInput(
                format!("Input too large: {} > {}", input.len(), self.config.max_input_size)
            ));
        }

        // Calculate gas cost
        let gas_cost = self.calculate_gas_cost(input.len());
        if gas_cost > gas_limit {
            return Err(PrecompiledError::OutOfGas);
        }

        // Execute based on contract type
        let (output, gas_used) = match self.config.contract_type {
            PrecompiledType::EcdsaRecovery => self.execute_ecdsa_recovery(input)?,
            PrecompiledType::Sha256 => self.execute_sha256(input)?,
            PrecompiledType::Ripemd160 => self.execute_ripemd160(input)?,
            PrecompiledType::Identity => self.execute_identity(input)?,
            PrecompiledType::ModExp => self.execute_modexp(input)?,
            PrecompiledType::EcAdd => self.execute_ecadd(input)?,
            PrecompiledType::EcMul => self.execute_ecmul(input)?,
            PrecompiledType::EcPairing => self.execute_ecpairing(input)?,
            PrecompiledType::Blake2F => self.execute_blake2f(input)?,
            PrecompiledType::ZstdCompress => self.execute_zstd_compress(input)?,
            PrecompiledType::ZstdDecompress => self.execute_zstd_decompress(input)?,
        };

        Ok(EvmExecutionResult {
            success: true,
            gas_used,
            return_data: output,
            contract_address: None,
            logs: vec![],
            error: None,
        })
    }

    /// Calculate gas cost
    fn calculate_gas_cost(&self, input_size: usize) -> u64 {
        let words = (input_size + 31) / 32;
        self.config.base_gas + (words as u64 * self.config.word_gas)
    }

    /// Execute ECDSA recovery
    fn execute_ecdsa_recovery(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        if input.len() != 128 {
            return Err(PrecompiledError::InvalidInput("Invalid input length for ECDSA recovery".to_string()));
        }

        let hash = &input[0..32];
        let v = input[32];
        let r = &input[33..65];
        let s = &input[65..97];

        let recovery_id = RecoveryId::from_byte(v).ok_or_else(|| 
            PrecompiledError::InvalidInput("Invalid recovery ID".to_string())
        )?;

        let signature = Signature::from_scalars(
            BigUint::from_bytes_be(r).into(),
            BigUint::from_bytes_be(s).into()
        ).map_err(|e| PrecompiledError::ExecutionFailed(format!("Invalid signature: {}", e)))?;

        let public_key = PublicKey::recover_from_msg(
            hash,
            &signature,
            recovery_id
        ).map_err(|e| PrecompiledError::ExecutionFailed(format!("Recovery failed: {}", e)))?;

        let mut output = vec![0u8; 32];
        output[12..].copy_from_slice(&public_key.to_encoded_point(false).as_bytes()[1..]);

        Ok((output, self.calculate_gas_cost(input.len())))
    }

    /// Execute SHA256 hash
    fn execute_sha256(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        let mut hasher = Sha3_256::new();
        hasher.update(input);
        let result = hasher.finalize();
        
        Ok((result.to_vec(), self.calculate_gas_cost(input.len())))
    }

    /// Execute RIPEMD160 hash
    fn execute_ripemd160(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        let mut hasher = Ripemd160::new();
        hasher.update(input);
        let result = hasher.finalize();
        
        Ok((result.to_vec(), self.calculate_gas_cost(input.len())))
    }

    /// Execute identity function
    fn execute_identity(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        Ok((input.to_vec(), self.calculate_gas_cost(input.len())))
    }

    /// Execute modular exponentiation
    fn execute_modexp(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        if input.len() < 96 {
            return Err(PrecompiledError::InvalidInput("Input too short for ModExp".to_string()));
        }

        let base_len = BigUint::from_bytes_be(&input[0..32]).to_usize().unwrap_or(0);
        let exp_len = BigUint::from_bytes_be(&input[32..64]).to_usize().unwrap_or(0);
        let mod_len = BigUint::from_bytes_be(&input[64..96]).to_usize().unwrap_or(0);

        if input.len() < 96 + base_len + exp_len + mod_len {
            return Err(PrecompiledError::InvalidInput("Input too short for ModExp parameters".to_string()));
        }

        let base = BigUint::from_bytes_be(&input[96..96+base_len]);
        let exp = BigUint::from_bytes_be(&input[96+base_len..96+base_len+exp_len]);
        let modulus = BigUint::from_bytes_be(&input[96+base_len+exp_len..96+base_len+exp_len+mod_len]);

        if modulus.is_zero() {
            return Err(PrecompiledError::InvalidInput("Modulus cannot be zero".to_string()));
        }

        let result = base.modpow(&exp, &modulus);
        let mut output = vec![0u8; mod_len];
        let result_bytes = result.to_bytes_be();
        output[mod_len - result_bytes.len()..].copy_from_slice(&result_bytes);

        Ok((output, self.calculate_gas_cost(input.len())))
    }

    /// Execute EC addition
    fn execute_ecadd(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        if input.len() != 128 {
            return Err(PrecompiledError::InvalidInput("Invalid input length for EC addition".to_string()));
        }

        let p1 = G1::from_bytes(&input[0..64])
            .map_err(|e| PrecompiledError::InvalidInput(format!("Invalid point 1: {}", e)))?;
        let p2 = G1::from_bytes(&input[64..128])
            .map_err(|e| PrecompiledError::InvalidInput(format!("Invalid point 2: {}", e)))?;

        let result = p1 + p2;
        let mut output = vec![0u8; 64];
        output.copy_from_slice(&result.to_bytes());

        Ok((output, self.calculate_gas_cost(input.len())))
    }

    /// Execute EC scalar multiplication
    fn execute_ecmul(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        if input.len() != 96 {
            return Err(PrecompiledError::InvalidInput("Invalid input length for EC multiplication".to_string()));
        }

        let point = G1::from_bytes(&input[0..64])
            .map_err(|e| PrecompiledError::InvalidInput(format!("Invalid point: {}", e)))?;
        let scalar = BigUint::from_bytes_be(&input[64..96]);

        let result = point * scalar;
        let mut output = vec![0u8; 64];
        output.copy_from_slice(&result.to_bytes());

        Ok((output, self.calculate_gas_cost(input.len())))
    }

    /// Execute EC pairing
    fn execute_ecpairing(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        if input.len() % 192 != 0 {
            return Err(PrecompiledError::InvalidInput("Invalid input length for EC pairing".to_string()));
        }

        let mut result = Gt::one();
        for i in 0..input.len() / 192 {
            let g1 = G1::from_bytes(&input[i*192..i*192+64])
                .map_err(|e| PrecompiledError::InvalidInput(format!("Invalid G1 point: {}", e)))?;
            let g2 = G2::from_bytes(&input[i*192+64..i*192+192])
                .map_err(|e| PrecompiledError::InvalidInput(format!("Invalid G2 point: {}", e)))?;

            result = result * pairing::bls12_381::pairing(g1, g2);
        }

        let mut output = vec![0u8; 32];
        output[0] = if result == Gt::one() { 1 } else { 0 };

        Ok((output, self.calculate_gas_cost(input.len())))
    }

    /// Execute Blake2F compression
    fn execute_blake2f(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        if input.len() != 213 {
            return Err(PrecompiledError::InvalidInput("Invalid input length for Blake2F".to_string()));
        }

        let rounds = u32::from_be_bytes(input[0..4].try_into().unwrap());
        let h = &input[4..68];
        let m = &input[68..196];
        let t = &input[196..212];
        let f = input[212] != 0;

        let mut hasher = Blake2b::new();
        hasher.update(h);
        hasher.update(m);
        hasher.update(t);
        if f {
            hasher.update(&[1]);
        }

        let result = hasher.finalize();
        Ok((result.to_vec(), self.calculate_gas_cost(input.len())))
    }

    /// Execute ZSTD compression
    fn execute_zstd_compress(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        let compressed = encode_all(input, 0)
            .map_err(|e| PrecompiledError::ExecutionFailed(format!("ZSTD compression failed: {}", e)))?;
        
        Ok((compressed, self.calculate_gas_cost(input.len())))
    }

    /// Execute ZSTD decompression
    fn execute_zstd_decompress(&self, input: &[u8]) -> Result<(Vec<u8>, u64), PrecompiledError> {
        let decompressed = decode_all(input)
            .map_err(|e| PrecompiledError::ExecutionFailed(format!("ZSTD decompression failed: {}", e)))?;
        
        Ok((decompressed, self.calculate_gas_cost(input.len())))
    }
}

/// Precompiled contract registry
pub struct PrecompiledRegistry {
    /// Registered contracts
    contracts: Vec<PrecompiledContract>,
}

impl PrecompiledRegistry {
    /// Create a new precompiled registry
    pub fn new() -> Self {
        Self {
            contracts: Vec::new(),
        }
    }

    /// Register a precompiled contract
    pub fn register_contract(&mut self, contract: PrecompiledContract) {
        self.contracts.push(contract);
    }

    /// Get contract by address
    pub fn get_contract(&self, address: &EvmAddress) -> Option<&PrecompiledContract> {
        self.contracts.iter()
            .find(|c| c.config.contract_type == self.address_to_type(address))
    }

    /// Convert address to contract type
    fn address_to_type(&self, address: &EvmAddress) -> PrecompiledType {
        match address.as_bytes()[19] {
            1 => PrecompiledType::EcdsaRecovery,
            2 => PrecompiledType::Sha256,
            3 => PrecompiledType::Ripemd160,
            4 => PrecompiledType::Identity,
            5 => PrecompiledType::ModExp,
            6 => PrecompiledType::EcAdd,
            7 => PrecompiledType::EcMul,
            8 => PrecompiledType::EcPairing,
            9 => PrecompiledType::Blake2F,
            10 => PrecompiledType::ZstdCompress,
            11 => PrecompiledType::ZstdDecompress,
            _ => PrecompiledType::Identity,
        }
    }
}

/// Default precompiled contract configurations
pub fn default_configs() -> Vec<PrecompiledConfig> {
    vec![
        PrecompiledConfig {
            contract_type: PrecompiledType::EcdsaRecovery,
            base_gas: 3000,
            word_gas: 0,
            max_input_size: 128,
        },
        PrecompiledConfig {
            contract_type: PrecompiledType::Sha256,
            base_gas: 60,
            word_gas: 12,
            max_input_size: 1024,
        },
        PrecompiledConfig {
            contract_type: PrecompiledType::Ripemd160,
            base_gas: 600,
            word_gas: 120,
            max_input_size: 1024,
        },
        PrecompiledConfig {
            contract_type: PrecompiledType::Identity,
            base_gas: 15,
            word_gas: 3,
            max_input_size: 1024,
        },
        PrecompiledConfig {
            contract_type: PrecompiledType::ModExp,
            base_gas: 0,
            word_gas: 0,
            max_input_size: 1024,
        },
        PrecompiledConfig {
            contract_type: PrecompiledType::EcAdd,
            base_gas: 500,
            word_gas: 0,
            max_input_size: 128,
        },
        PrecompiledConfig {
            contract_type: PrecompiledType::EcMul,
            base_gas: 40000,
            word_gas: 0,
            max_input_size: 128,
        },
        PrecompiledConfig {
            contract_type: PrecompiledType::EcPairing,
            base_gas: 100000,
            word_gas: 0,
            max_input_size: 1024,
        },
        PrecompiledConfig {
            contract_type: PrecompiledType::Blake2F,
            base_gas: 0,
            word_gas: 0,
            max_input_size: 1024,
        },
        PrecompiledConfig {
            contract_type: PrecompiledType::ZstdCompress,
            base_gas: 0,
            word_gas: 0,
            max_input_size: 1024,
        },
        PrecompiledConfig {
            contract_type: PrecompiledType::ZstdDecompress,
            base_gas: 0,
            word_gas: 0,
            max_input_size: 1024,
        },
    ]
} 