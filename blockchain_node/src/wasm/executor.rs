//! WASM contract executor
//!
//! Provides the execution environment for WebAssembly smart contracts.
//! Integrates storage, runtime, and context for contract execution.

use std::sync::Arc;
use std::collections::HashMap;

use crate::storage::Storage;
use crate::ledger::Ledger;
use crate::wasm::runtime::WasmRuntime;
use crate::wasm::storage::WasmStorage;
use crate::wasm::types::{
    WasmContractAddress, CallContext, CallParams, CallResult, WasmError,
    WasmTransaction, WasmExecutionResult, WasmLog
};
use crate::types::{Block, Transaction, Address};

/// Transaction executor for WASM smart contracts
pub struct WasmExecutor {
    /// Storage for persisting contract state
    storage: Arc<dyn Storage>,
    /// Runtime for executing contracts
    runtime: WasmRuntime,
    /// Ledger for reading blockchain state
    ledger: Arc<Ledger>,
    /// Cached contract metadata
    contract_cache: HashMap<WasmContractAddress, ContractInfo>,
}

/// Cached contract information
struct ContractInfo {
    /// Contract bytecode hash
    bytecode_hash: [u8; 32],
    /// Number of total calls to this contract (for metrics)
    call_count: u64,
    /// Average gas used per call
    avg_gas_used: u64,
}

impl WasmExecutor {
    /// Create a new WasmExecutor
    pub fn new(storage: Arc<dyn Storage>, ledger: Arc<Ledger>) -> Self {
        let runtime = WasmRuntime::new(storage.clone());
        
        Self {
            storage,
            runtime,
            ledger,
            contract_cache: HashMap::new(),
        }
    }
    
    /// Deploy a new contract transaction
    pub fn deploy(
        &mut self, 
        transaction: &WasmTransaction
    ) -> Result<WasmExecutionResult, WasmError> {
        // Verify the transaction signature
        self.verify_transaction(transaction)?;
        
        // Get the deployer
        let deployer = transaction.get_sender();
        
        // Get the nonce
        let nonce = self.ledger.get_account_nonce(&deployer)?;
        
        // Get the contract bytecode
        let bytecode = match &transaction.data {
            Some(data) => data,
            None => return Err(WasmError::ValidationError("Missing contract bytecode".to_string())),
        };
        
        // Deploy the contract
        let contract_address = self.runtime.deploy_contract(
            bytecode, 
            &deployer, 
            nonce,
            transaction.constructor_args.as_deref()
        )?;
        
        // Calculate bytecode hash for caching
        let bytecode_hash = self.calculate_bytecode_hash(bytecode);
        
        // Add to contract cache
        self.contract_cache.insert(contract_address.clone(), ContractInfo {
            bytecode_hash,
            call_count: 0,
            avg_gas_used: 0,
        });
        
        // Return success with the contract address
        Ok(WasmExecutionResult::deployment_success(
            contract_address,
            transaction.gas_limit - transaction.gas_used,
            vec![]
        ))
    }
    
    /// Execute a contract transaction
    pub fn execute(
        &mut self, 
        transaction: &WasmTransaction
    ) -> Result<WasmExecutionResult, WasmError> {
        // Verify the transaction signature
        self.verify_transaction(transaction)?;
        
        // Get the sender
        let sender = transaction.get_sender();
        
        // Get the contract address
        let contract_address = match &transaction.to {
            Some(to) => to.clone(),
            None => return Err(WasmError::ValidationError("Missing contract address".to_string())),
        };
        
        // Check if the contract exists
        let contract_storage = WasmStorage::new(self.storage.clone(), &contract_address);
        if !contract_storage.contract_exists() {
            return Err(WasmError::ValidationError(
                format!("Contract does not exist: {}", contract_address)
            ));
        }
        
        // Get the current block for context
        let current_block = self.ledger.get_latest_block()?;
        
        // Create the call context
        let context = CallContext {
            contract_address: contract_address.clone(),
            caller: sender,
            value: transaction.value.unwrap_or(0),
            gas_limit: self.config.gas_limit,
            block_height: current_block.height,
            block_timestamp: current_block.timestamp,
        };
        
        // Create the call parameters
        let params = CallParams {
            function: transaction.function.clone().unwrap_or_else(|| "main".to_string()),
            arguments: transaction.function_args.clone().unwrap_or_default(),
            gas_limit: transaction.gas_limit,
        };
        
        // Execute the contract
        let result = self.runtime.execute_contract(
            &contract_address,
            &context,
            &params
        )?;
        
        // Update contract metrics
        if let Some(contract_info) = self.contract_cache.get_mut(&contract_address) {
            contract_info.call_count += 1;
            
            // Update average gas used
            let total_gas = contract_info.avg_gas_used * (contract_info.call_count - 1);
            let new_avg = (total_gas + result.gas_used) / contract_info.call_count;
            contract_info.avg_gas_used = new_avg;
        }
        
        // Convert to execution result
        if result.succeeded {
            Ok(WasmExecutionResult::call_success(
                result.data,
                result.gas_used,
                Vec::new() // In a full implementation, would collect logs from execution
            ))
        } else {
            Ok(WasmExecutionResult::call_failure(
                result.error.unwrap_or_else(|| "Unknown execution error".to_string()),
                result.gas_used,
                Vec::new()
            ))
        }
    }
    
    /// Verify a transaction signature
    fn verify_transaction(&self, transaction: &WasmTransaction) -> Result<(), WasmError> {
        // In a real implementation, you would verify the signature here
        // For now, we just return success
        Ok(())
    }
    
    /// Calculate a hash of the contract bytecode
    fn calculate_bytecode_hash(&self, bytecode: &[u8]) -> [u8; 32] {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(bytecode);
        *hasher.finalize().as_bytes()
    }
    
    /// Check if a contract exists
    pub fn contract_exists(&self, address: &WasmContractAddress) -> bool {
        let contract_storage = WasmStorage::new(self.storage.clone(), address);
        contract_storage.contract_exists()
    }
    
    /// Get contract metrics
    pub fn get_contract_metrics(&self, address: &WasmContractAddress) -> Option<(u64, u64)> {
        self.contract_cache.get(address).map(|info| {
            (info.call_count, info.avg_gas_used)
        })
    }
    
    /// Read contract storage (view function, doesn't modify state)
    pub fn read_contract_storage(
        &self,
        contract_address: &WasmContractAddress,
        key: &[u8]
    ) -> Result<Option<Vec<u8>>, WasmError> {
        let contract_storage = WasmStorage::new(self.storage.clone(), contract_address);
        contract_storage.read(key)
            .map_err(|e| WasmError::StorageError(e.to_string()))
    }
    
    /// Execute a contract view function (doesn't modify state)
    pub fn execute_view(
        &self,
        contract_address: &WasmContractAddress,
        function: &str,
        args: &[u8],
        caller: &Address
    ) -> Result<CallResult, WasmError> {
        // Check if the contract exists
        let contract_storage = WasmStorage::new(self.storage.clone(), contract_address);
        if !contract_storage.contract_exists() {
            return Err(WasmError::ValidationError(
                format!("Contract does not exist: {}", contract_address)
            ));
        }
        
        // Get the current block for context
        let current_block = self.ledger.get_latest_block()?;
        
        // Create the call context
        let context = CallContext {
            contract_address: contract_address.clone(),
            caller: caller.clone(),
            value: 0, // View functions can't receive value
            gas_limit: self.config.gas_limit,
            block_height: current_block.height,
            block_timestamp: current_block.timestamp,
        };
        
        // Create the call parameters with a standard gas limit for view functions
        let params = CallParams {
            function: function.to_string(),
            arguments: args.to_vec(),
            gas_limit: 1_000_000, // Standard gas limit for view functions
        };
        
        // Execute the contract in a cloned runtime to avoid state changes
        let mut runtime_clone = self.runtime.clone();
        runtime_clone.execute_contract(
            contract_address,
            &context,
            &params
        )
    }
    
    /// Add this compatibility method to get the latest block
    pub fn get_latest_block(&self) -> Result<Block, WasmError> {
        self.ledger.get_latest_block()
            .map_err(|e| WasmError::Internal(format!("Failed to get latest block: {}", e)))
    }
    
    /// Add this compatibility method to get account nonce
    pub fn get_account_nonce(&self, address: &Address) -> Result<u64, WasmError> {
        self.ledger.get_account_nonce(address)
            .map_err(|e| WasmError::Internal(format!("Failed to get account nonce: {}", e)))
    }
} 