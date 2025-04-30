use crate::evm::backend::{EvmBackend, EvmAccount};
use crate::evm::types::{EvmAddress, EvmTransaction, EvmExecutionResult, EvmError, EvmConfig, EvmLog};
use crate::evm::precompiles::init_precompiles;
use crate::storage::HybridStorage;
use ethereum_types::{H160, H256, U256};
use std::sync::Arc;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use log::{debug, info, warn, error};
use std::collections::HashMap;

#[cfg(feature = "evm-runtime")]
use evm_runtime::{ExitReason, ExitSucceed, ExitError, Config as EvmRuntimeConfig};

/// EVM Runtime for executing Solidity smart contracts
pub struct EvmRuntime {
    /// Backend adapter to our storage system
    backend: EvmBackend,
    /// Configuration for the EVM
    config: EvmConfig,
    /// Block number for current execution context
    block_number: u64,
    /// Block timestamp for current execution context
    block_timestamp: u64,
    /// Block gas limit
    gas_limit: u64,
    /// Logs from the current execution
    logs: Vec<EvmLog>,
}

impl EvmRuntime {
    /// Create a new EVM runtime
    pub fn new(storage: Arc<HybridStorage>, config: EvmConfig) -> Self {
        Self {
            backend: EvmBackend::new(storage),
            config,
            block_number: 0,
            block_timestamp: 0,
            gas_limit: config.default_gas_limit,
            logs: Vec::new(),
        }
    }
    
    /// Set the current block context
    pub fn set_block_context(&mut self, number: u64, timestamp: u64) {
        self.block_number = number;
        self.block_timestamp = timestamp;
    }
    
    /// Execute a transaction
    pub async fn execute(&mut self, tx: EvmTransaction) -> Result<EvmExecutionResult, EvmError> {
        info!("Executing EVM transaction: {:?}", tx);
        
        // Validate transaction
        self.validate_transaction(&tx)?;
        
        // Execute transaction based on type (call or create)
        let result = match tx.to {
            Some(to) => self.execute_call(tx.from, to, tx.value, tx.gas_limit.as_u64(), tx.data).await?,
            None => self.execute_create(tx.from, tx.value, tx.gas_limit.as_u64(), tx.data).await?,
        };
        
        // Update sender account (nonce and balance)
        let mut sender_account = self.backend.get_account(&tx.from).await?;
        
        // Increment nonce
        let nonce = sender_account.nonce.as_u64() + 1;
        sender_account.nonce = U256::from(nonce);
        
        // Deduct gas cost (gas_used * gas_price)
        let gas_cost = U256::from(result.gas_used) * tx.gas_price;
        if sender_account.balance >= gas_cost {
            sender_account.balance -= gas_cost;
        } else {
            warn!("Account doesn't have enough balance to pay for gas");
            sender_account.balance = U256::zero();
        }
        
        // Update sender account
        self.backend.update_account(tx.from, sender_account).await?;
        
        // Commit changes
        self.backend.commit().await?;
        
        Ok(result)
    }
    
    /// Execute a contract call
    async fn execute_call(
        &mut self,
        sender: EvmAddress,
        target: EvmAddress,
        value: U256,
        gas_limit: u64,
        data: Vec<u8>,
    ) -> Result<EvmExecutionResult, EvmError> {
        debug!("Executing call to {:?}", target);
        
        // Check if this is a precompiled contract
        if let Some(precompile) = self.config.precompiles.get(&target) {
            debug!("Executing precompiled contract at {:?}", target);
            
            // Call the precompiled contract
            let (return_data, gas_used) = precompile(&data, gas_limit)?;
            
            return Ok(EvmExecutionResult {
                success: true,
                gas_used,
                return_data,
                contract_address: None,
                logs: Vec::new(),
                error: None,
            });
        }
        
        // This is not a precompiled contract, execute in the EVM
        #[cfg(feature = "evm-runtime")]
        {
            // Set up EVM runtime configuration
            let config = EvmRuntimeConfig::london();
            
            // Create a schema for the EVM runtime from our backend
            // This is a simplified example - in a real implementation, this would be a proper adapter
            let storage_fn = |address: H160, key: H256| -> H256 {
                // This would call backend.get_storage in a real async implementation
                H256::zero()
            };
            
            let mut evm = evm_runtime::Machine::new(
                &config,
                &storage_fn,
                sender,
                target,
                value,
                data.clone(),
                gas_limit,
            );
            
            // Execute the call
            let (exit_reason, result_data, gas_used) = evm.execute();
            
            match exit_reason {
                ExitReason::Succeed(succeed) => {
                    debug!("Call succeeded: {:?}", succeed);
                    
                    Ok(EvmExecutionResult {
                        success: true,
                        gas_used,
                        return_data: result_data,
                        contract_address: None,
                        logs: self.logs.clone(), // Get logs from EVM execution
                        error: None,
                    })
                },
                ExitReason::Error(error) => {
                    warn!("Call error: {:?}", error);
                    
                    Ok(EvmExecutionResult {
                        success: false,
                        gas_used,
                        return_data: result_data,
                        contract_address: None,
                        logs: Vec::new(),
                        error: Some(format!("EVM error: {:?}", error)),
                    })
                },
                ExitReason::Revert(revert) => {
                    warn!("Call reverted: {:?}", revert);
                    
                    Ok(EvmExecutionResult {
                        success: false,
                        gas_used,
                        return_data: result_data,
                        contract_address: None,
                        logs: Vec::new(),
                        error: Some("Transaction reverted".to_string()),
                    })
                },
                ExitReason::Fatal(fatal) => {
                    error!("Call fatal error: {:?}", fatal);
                    
                    Ok(EvmExecutionResult {
                        success: false,
                        gas_used,
                        return_data: result_data,
                        contract_address: None,
                        logs: Vec::new(),
                        error: Some(format!("Fatal error: {:?}", fatal)),
                    })
                },
            }
        }
        
        #[cfg(not(feature = "evm-runtime"))]
        {
            // Placeholder implementation when EVM runtime is not enabled
            warn!("EVM runtime is not enabled (feature flag 'evm-runtime' is not set)");
            
            Ok(EvmExecutionResult {
                success: false,
                gas_used: 0,
                return_data: Vec::new(),
                contract_address: None,
                logs: Vec::new(),
                error: Some("EVM runtime is not enabled".to_string()),
            })
        }
    }
    
    /// Execute contract creation
    async fn execute_create(
        &mut self,
        sender: EvmAddress,
        value: U256,
        gas_limit: u64,
        code: Vec<u8>,
    ) -> Result<EvmExecutionResult, EvmError> {
        debug!("Executing contract creation");
        
        // Generate new contract address (simplified - in real implementation this would use keccak256(rlp([sender, nonce])))
        // This is a placeholder implementation
        let sender_account = self.backend.get_account(&sender).await?;
        let nonce = sender_account.nonce.as_u64();
        
        // Generate contract address (this is a simplified version)
        let mut hasher = sha3::Keccak256::new();
        hasher.update(sender.as_bytes());
        hasher.update(&nonce.to_be_bytes());
        let hash_result = hasher.finalize();
        
        let mut address_bytes = [0u8; 20];
        address_bytes.copy_from_slice(&hash_result[12..32]);
        let contract_address = H160::from(address_bytes);
        
        debug!("New contract address: {:?}", contract_address);
        
        // Execute the contract creation code
        #[cfg(feature = "evm-runtime")]
        {
            // Set up EVM runtime configuration
            let config = EvmRuntimeConfig::london();
            
            // Create a schema for the EVM runtime from our backend
            // This is a simplified example - in a real implementation, this would be a proper adapter
            let storage_fn = |address: H160, key: H256| -> H256 {
                // This would call backend.get_storage in a real async implementation
                H256::zero()
            };
            
            let mut evm = evm_runtime::Machine::new(
                &config,
                &storage_fn,
                sender,
                H160::zero(), // For create, target is zero address
                value,
                code.clone(),
                gas_limit,
            );
            
            // Execute the creation code
            let (exit_reason, result_data, gas_used) = evm.execute();
            
            match exit_reason {
                ExitReason::Succeed(succeed) => {
                    debug!("Contract creation succeeded: {:?}", succeed);
                    
                    // Store the contract code
                    self.backend.set_code(&contract_address, &result_data).await?;
                    
                    // Create the contract account
                    let contract_account = EvmAccount {
                        nonce: U256::zero(),
                        balance: value,
                        storage_root: H256::zero(),
                        code_hash: H256::zero(), // This would be keccak256(code) in a real implementation
                    };
                    
                    // Store the account
                    self.backend.update_account(contract_address, contract_account).await?;
                    
                    Ok(EvmExecutionResult {
                        success: true,
                        gas_used,
                        return_data: result_data.clone(),
                        contract_address: Some(contract_address),
                        logs: self.logs.clone(), // Get logs from EVM execution
                        error: None,
                    })
                },
                ExitReason::Error(error) => {
                    warn!("Contract creation error: {:?}", error);
                    
                    Ok(EvmExecutionResult {
                        success: false,
                        gas_used,
                        return_data: result_data,
                        contract_address: None,
                        logs: Vec::new(),
                        error: Some(format!("EVM error: {:?}", error)),
                    })
                },
                ExitReason::Revert(revert) => {
                    warn!("Contract creation reverted: {:?}", revert);
                    
                    Ok(EvmExecutionResult {
                        success: false,
                        gas_used,
                        return_data: result_data,
                        contract_address: None,
                        logs: Vec::new(),
                        error: Some("Transaction reverted".to_string()),
                    })
                },
                ExitReason::Fatal(fatal) => {
                    error!("Contract creation fatal error: {:?}", fatal);
                    
                    Ok(EvmExecutionResult {
                        success: false,
                        gas_used,
                        return_data: result_data,
                        contract_address: None,
                        logs: Vec::new(),
                        error: Some(format!("Fatal error: {:?}", fatal)),
                    })
                },
            }
        }
        
        #[cfg(not(feature = "evm-runtime"))]
        {
            // Placeholder implementation when EVM runtime is not enabled
            warn!("EVM runtime is not enabled (feature flag 'evm-runtime' is not set)");
            
            Ok(EvmExecutionResult {
                success: false,
                gas_used: 0,
                return_data: Vec::new(),
                contract_address: None,
                logs: Vec::new(),
                error: Some("EVM runtime is not enabled".to_string()),
            })
        }
    }
    
    /// Validate a transaction before execution
    fn validate_transaction(&mut self, tx: &EvmTransaction) -> Result<(), EvmError> {
        // Check for invalid transaction
        if tx.gas_limit.as_u64() == 0 {
            return Err(EvmError::InvalidTransaction("Gas limit cannot be zero".to_string()));
        }
        
        if tx.gas_limit.as_u64() > self.gas_limit {
            return Err(EvmError::InvalidTransaction(format!("Gas limit {} exceeds block gas limit {}", tx.gas_limit.as_u64(), self.gas_limit)));
        }
        
        // More validation logic would go here...
        
        Ok(())
    }
    
    /// Clear cached data
    pub fn clear_cache(&mut self) {
        self.backend.clear_caches();
        self.logs.clear();
    }
    
    /// Get the current gas limit
    pub fn get_gas_limit(&self) -> u64 {
        self.gas_limit
    }
    
    /// Set the gas limit
    pub fn set_gas_limit(&mut self, gas_limit: u64) {
        self.gas_limit = gas_limit;
    }
} 