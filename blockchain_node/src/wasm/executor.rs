//! WASM contract executor
//!
//! Provides the execution environment for WebAssembly smart contracts.
//! Integrates storage, runtime, and context for contract execution.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ledger::state::State;
use crate::ledger::Ledger;
use crate::storage::Storage;
use crate::types::{Address, Block};
use crate::wasm::runtime::WasmRuntime;
use crate::wasm::runtime::{WasmConfig, WasmEnv};
use crate::wasm::storage::WasmStorage;
use crate::wasm::types::{
    CallContext, CallParams, CallResult, WasmContractAddress, WasmError, WasmExecutionResult,
    WasmTransaction,
};

use log::debug;
use wasmtime::Val;

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

/// Smart contract executor
pub struct ContractExecutor {
    /// WASM runtime
    runtime: WasmRuntime,
    /// Contract cache
    contract_cache: RwLock<HashMap<String, wasmtime::Module>>,
    /// Configuration
    config: WasmConfig,
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
        transaction: &WasmTransaction,
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
            None => {
                return Err(WasmError::ValidationError(
                    "Missing contract bytecode".to_string(),
                ))
            }
        };

        // Deploy the contract
        let contract_address = self.runtime.deploy_contract(
            bytecode,
            &deployer,
            nonce,
            transaction.constructor_args.as_deref(),
        )?;

        // Calculate bytecode hash for caching
        let bytecode_hash = self.calculate_bytecode_hash(bytecode);

        // Add to contract cache
        self.contract_cache.insert(
            contract_address.clone(),
            ContractInfo {
                bytecode_hash,
                call_count: 0,
                avg_gas_used: 0,
            },
        );

        // Return success with the contract address
        Ok(WasmExecutionResult::deployment_success(
            contract_address,
            transaction.gas_limit - transaction.gas_used,
            vec![],
        ))
    }

    /// Execute a contract transaction
    pub fn execute(
        &mut self,
        transaction: &WasmTransaction,
    ) -> Result<WasmExecutionResult, WasmError> {
        // Verify the transaction signature
        self.verify_transaction(transaction)?;

        // Get the sender
        let sender = transaction.get_sender();

        // Get the contract address
        let contract_address = match &transaction.to {
            Some(to) => to.clone(),
            None => {
                return Err(WasmError::ValidationError(
                    "Missing contract address".to_string(),
                ))
            }
        };

        // Check if the contract exists
        let contract_storage = WasmStorage::new(self.storage.clone(), &contract_address);
        if !contract_storage.contract_exists() {
            return Err(WasmError::ValidationError(format!(
                "Contract does not exist: {}",
                contract_address
            )));
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
            function: transaction
                .function
                .clone()
                .unwrap_or_else(|| "main".to_string()),
            arguments: transaction.function_args.clone().unwrap_or_default(),
            gas_limit: transaction.gas_limit,
        };

        // Execute the contract
        let result = self
            .runtime
            .execute_contract(&contract_address, &context, &params)?;

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
                Vec::new(), // In a full implementation, would collect logs from execution
            ))
        } else {
            Ok(WasmExecutionResult::call_failure(
                result
                    .error
                    .unwrap_or_else(|| "Unknown execution error".to_string()),
                result.gas_used,
                Vec::new(),
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
        self.contract_cache
            .get(address)
            .map(|info| (info.call_count, info.avg_gas_used))
    }

    /// Read contract storage (view function, doesn't modify state)
    pub fn read_contract_storage(
        &self,
        contract_address: &WasmContractAddress,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, WasmError> {
        let contract_storage = WasmStorage::new(self.storage.clone(), contract_address);
        contract_storage
            .read(key)
            .map_err(|e| WasmError::StorageError(e.to_string()))
    }

    /// Execute a contract view function (doesn't modify state)
    pub fn execute_view(
        &self,
        contract_address: &WasmContractAddress,
        function: &str,
        args: &[u8],
        caller: &Address,
    ) -> Result<CallResult, WasmError> {
        // Check if the contract exists
        let contract_storage = WasmStorage::new(self.storage.clone(), contract_address);
        if !contract_storage.contract_exists() {
            return Err(WasmError::ValidationError(format!(
                "Contract does not exist: {}",
                contract_address
            )));
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
        runtime_clone.execute_contract(contract_address, &context, &params)
    }

    /// Add this compatibility method to get the latest block
    pub fn get_latest_block(&self) -> Result<Block, WasmError> {
        self.ledger
            .get_latest_block()
            .map_err(|e| WasmError::Internal(format!("Failed to get latest block: {}", e)))
    }

    /// Add this compatibility method to get account nonce
    pub fn get_account_nonce(&self, address: &Address) -> Result<u64, WasmError> {
        self.ledger
            .get_account_nonce(address)
            .map_err(|e| WasmError::Internal(format!("Failed to get account nonce: {}", e)))
    }
}

impl ContractExecutor {
    /// Create a new contract executor
    pub fn new(config: WasmConfig) -> Result<Self> {
        let runtime = WasmRuntime::new(config.clone())?;
        Ok(Self {
            runtime,
            contract_cache: RwLock::new(HashMap::new()),
            config,
        })
    }

    /// Deploy a contract
    pub fn deploy_contract(
        &self,
        contract_code: &[u8],
        sender: &str,
        gas_limit: u64,
    ) -> Result<String> {
        debug!("Deploying contract from sender: {}", sender);

        // Validate WASM bytecode
        crate::wasm::validate_wasm_bytecode(contract_code)?;

        // Compile the contract
        let module = self.runtime.compile(contract_code)?;

        // Generate contract address (hash of code + sender + timestamp)
        let mut hasher = sha2::Sha256::new();
        use sha2::Digest;
        hasher.update(contract_code);
        hasher.update(sender.as_bytes());
        hasher.update(
            &std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                .to_be_bytes(),
        );
        let contract_address = hex::encode(hasher.finalize());

        // Store in cache
        let mut cache = self.contract_cache.write().unwrap();
        cache.insert(contract_address.clone(), module);

        // Call constructor if it exists
        // In a real implementation, we would check if the contract has a constructor
        // and call it with the deploy arguments

        debug!("Contract deployed at address: {}", contract_address);
        Ok(contract_address)
    }

    /// Execute a contract call
    pub fn execute_contract(
        &self,
        contract_address: &str,
        contract_code: &[u8],
        call_data: &[u8],
        sender: &str,
        value: u64,
        gas_limit: u64,
    ) -> Result<WasmExecutionResult> {
        debug!(
            "Executing contract call to {} from {}",
            contract_address, sender
        );

        // Get module from cache or compile
        let module = {
            let cache = self.contract_cache.read().unwrap();
            match cache.get(contract_address) {
                Some(module) => module.clone(),
                None => {
                    // Not in cache, compile it
                    drop(cache); // Release read lock
                    let module = self.runtime.compile(contract_code)?;
                    let mut cache = self.contract_cache.write().unwrap();
                    cache.insert(contract_address.to_string(), module.clone());
                    module
                }
            }
        };

        // Create execution environment
        let state = Arc::new(State::new(&crate::config::Config::default())?);
        let mut env = WasmEnv::new(
            state.clone(),
            gas_limit,
            sender,
            contract_address,
            value,
            call_data.to_vec(),
        );

        // Parse call data to determine function and arguments
        // In a real implementation, this would decode the ABI-encoded call data
        // For simplicity, we'll assume the first 4 bytes are the function selector
        // and the rest are parameters
        if call_data.len() < 4 {
            return Ok(WasmExecutionResult::failure(
                "Invalid call data".to_string(),
                0,
                Vec::new(),
            ));
        }

        let selector = &call_data[0..4];
        let args = &call_data[4..];

        // Map selector to function name
        // In a real implementation, this would use a proper ABI mapping
        let function_name = match selector {
            [0, 0, 0, 0] => "main",
            [1, 0, 0, 0] => "transfer",
            [2, 0, 0, 0] => "balance_of",
            [3, 0, 0, 0] => "total_supply",
            _ => "main", // Default to main
        };

        // Convert arguments to WASM values
        // In a real implementation, this would properly parse based on the function signature
        let wasm_args = if args.is_empty() {
            vec![]
        } else if args.len() >= 4 {
            let value = i32::from_le_bytes([args[0], args[1], args[2], args[3]]);
            vec![Val::I32(value)]
        } else {
            vec![Val::I32(0)]
        };

        // Execute the function
        let result = match self
            .runtime
            .execute(&module, &mut env, function_name, &wasm_args)
        {
            Ok(values) => {
                // Extract return value if any
                let return_data = if values.is_empty() {
                    None
                } else {
                    match &values[0] {
                        Val::I32(val) => Some(val.to_le_bytes().to_vec()),
                        Val::I64(val) => Some(val.to_le_bytes().to_vec()),
                        Val::F32(val) => Some(val.to_le_bytes().to_vec()),
                        Val::F64(val) => Some(val.to_le_bytes().to_vec()),
                        _ => None,
                    }
                };

                WasmExecutionResult::success(return_data, env.gas_meter.gas_used(), env.logs)
            }
            Err(e) => WasmExecutionResult::failure(
                format!("Execution error: {}", e),
                env.gas_meter.gas_used(),
                env.logs,
            ),
        };

        debug!(
            "Contract execution completed, gas used: {}",
            result.gas_used
        );
        Ok(result)
    }

    /// Update configuration
    pub fn update_config(&mut self, config: WasmConfig) {
        self.config = config;
    }

    /// Clear contract cache
    pub fn clear_cache(&self) {
        let mut cache = self.contract_cache.write().unwrap();
        cache.clear();
    }

    /// Get number of cached contracts
    pub fn cached_contracts_count(&self) -> usize {
        let cache = self.contract_cache.read().unwrap();
        cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_deployment() {
        // Create executor
        let config = WasmConfig::default();
        let executor = ContractExecutor::new(config).unwrap();

        // Simple Wasm module (wat format)
        let wat = r#"
            (module
                (func $main (export "main") (result i32)
                    i32.const 42
                )
            )
        "#;

        // Convert to wasm
        let wasm = wat::parse_str(wat).unwrap();

        // Deploy contract
        let result = executor.deploy_contract(&wasm, "test_sender", 1000000);
        assert!(result.is_ok());

        let contract_address = result.unwrap();
        assert!(!contract_address.is_empty());

        // Verify contract is cached
        assert_eq!(executor.cached_contracts_count(), 1);
    }

    #[test]
    fn test_contract_execution() {
        // Create executor
        let config = WasmConfig::default();
        let executor = ContractExecutor::new(config).unwrap();

        // Simple Wasm module (wat format) that returns 42
        let wat = r#"
            (module
                (func $main (export "main") (result i32)
                    i32.const 42
                )
            )
        "#;

        // Convert to wasm
        let wasm = wat::parse_str(wat).unwrap();

        // Deploy contract
        let contract_address = executor
            .deploy_contract(&wasm, "test_sender", 1000000)
            .unwrap();

        // Execute contract
        let call_data = vec![0, 0, 0, 0]; // Selector for "main"
        let result = executor
            .execute_contract(
                &contract_address,
                &wasm,
                &call_data,
                "test_sender",
                0,
                1000000,
            )
            .unwrap();

        // Verify success
        assert!(result.success);
        assert!(result.return_data.is_some());

        // Check return value (should be 42)
        let return_data = result.return_data.unwrap();
        assert_eq!(return_data.len(), 4); // i32 is 4 bytes
        let value = i32::from_le_bytes([
            return_data[0],
            return_data[1],
            return_data[2],
            return_data[3],
        ]);
        assert_eq!(value, 42);
    }
}
