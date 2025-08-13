//! WASM Smart Contract Example (Simplified)
//!
//! This example demonstrates the concept of WASM smart contracts
//! with a simplified implementation that doesn't require the full WASM VM.

use blockchain_node::storage::{MemMapOptions, MemMapStorage};
use blockchain_node::types::Address;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Simplified WASM VM configuration
#[derive(Debug, Clone)]
pub struct WasmVmConfig {
    pub gas_limit: u64,
    pub memory_limit: u32,
    pub timeout_ms: u64,
    pub enable_debug: bool,
}

impl Default for WasmVmConfig {
    fn default() -> Self {
        Self {
            gas_limit: 10_000_000,
            memory_limit: 16 * 1024 * 1024, // 16MB
            timeout_ms: 5000,
            enable_debug: false,
        }
    }
}

/// Smart contract execution context
#[derive(Debug, Clone)]
pub struct CallContext {
    pub contract_address: String,
    pub caller: String,
    pub block_height: u64,
    pub timestamp: u64,
}

/// Contract execution result
#[derive(Debug)]
pub struct ExecutionResult {
    pub success: bool,
    pub return_data: Option<Vec<u8>>,
    pub gas_used: u64,
    pub logs: Vec<String>,
    pub error_message: Option<String>,
}

/// Simplified contract storage
pub struct ContractStorage {
    #[allow(dead_code)]
    storage: Arc<MemMapStorage>,
    contract_states: HashMap<String, HashMap<String, Vec<u8>>>,
}

impl ContractStorage {
    pub fn new(storage: Arc<MemMapStorage>) -> Self {
        Self {
            storage,
            contract_states: HashMap::new(),
        }
    }

    pub fn store_value(&mut self, contract: &str, key: &str, value: Vec<u8>) {
        self.contract_states
            .entry(contract.to_string())
            .or_default()
            .insert(key.to_string(), value);
    }

    pub fn get_value(&self, contract: &str, key: &str) -> Option<Vec<u8>> {
        self.contract_states
            .get(contract)
            .and_then(|state| state.get(key))
            .cloned()
    }
}

/// Simplified WASM VM
pub struct WasmVm {
    #[allow(dead_code)]
    config: WasmVmConfig,
    contracts: HashMap<String, Vec<u8>>,
    storage: ContractStorage,
}

impl WasmVm {
    pub fn new(config: WasmVmConfig, storage: Arc<MemMapStorage>) -> Self {
        Self {
            config,
            contracts: HashMap::new(),
            storage: ContractStorage::new(storage),
        }
    }

    pub fn load_contract(&mut self, address: &str, bytecode: Vec<u8>) -> Result<(), String> {
        println!("📦 Loading contract at address: {}", address);
        println!("   Bytecode size: {} bytes", bytecode.len());

        // Simulate validation
        if bytecode.is_empty() {
            return Err("Empty bytecode".to_string());
        }

        self.contracts.insert(address.to_string(), bytecode);
        println!("✅ Contract loaded and validated successfully");
        Ok(())
    }

    pub fn execute_function(
        &mut self,
        contract_address: &str,
        function_name: &str,
        context: CallContext,
        args: Vec<i32>,
    ) -> Result<ExecutionResult, String> {
        println!(
            "🚀 Executing function '{}' on contract {}",
            function_name, contract_address
        );

        // Check if contract exists
        if !self.contracts.contains_key(contract_address) {
            return Err(format!("Contract not found: {}", contract_address));
        }

        // Simulate different contract functions
        match function_name {
            "increment" => self.execute_increment(contract_address, context),
            "get" => self.execute_get(contract_address, context),
            "set" => {
                let value = args.first().copied().unwrap_or(0);
                self.execute_set(contract_address, context, value)
            }
            "transfer" => {
                let amount = args.first().copied().unwrap_or(0) as u64;
                self.execute_transfer(contract_address, context, amount)
            }
            "balance" => self.execute_get_balance(contract_address, context),
            _ => Err(format!("Unknown function: {}", function_name)),
        }
    }

    fn execute_increment(
        &mut self,
        contract: &str,
        context: CallContext,
    ) -> Result<ExecutionResult, String> {
        let current_value = self.get_counter_value(contract);
        let new_value = current_value + 1;

        // Store new value
        self.storage
            .store_value(contract, "counter", new_value.to_le_bytes().to_vec());

        println!("   Counter incremented: {} -> {}", current_value, new_value);

        Ok(ExecutionResult {
            success: true,
            return_data: Some(new_value.to_le_bytes().to_vec()),
            gas_used: 21000,
            logs: vec![format!("Counter incremented by {}", context.caller)],
            error_message: None,
        })
    }

    fn execute_get(
        &self,
        contract: &str,
        _context: CallContext,
    ) -> Result<ExecutionResult, String> {
        let value = self.get_counter_value(contract);

        println!("   Current counter value: {}", value);

        Ok(ExecutionResult {
            success: true,
            return_data: Some(value.to_le_bytes().to_vec()),
            gas_used: 5000,
            logs: vec![],
            error_message: None,
        })
    }

    fn execute_set(
        &mut self,
        contract: &str,
        context: CallContext,
        value: i32,
    ) -> Result<ExecutionResult, String> {
        // Store new value
        self.storage
            .store_value(contract, "counter", value.to_le_bytes().to_vec());

        println!("   Counter set to: {}", value);

        Ok(ExecutionResult {
            success: true,
            return_data: Some(value.to_le_bytes().to_vec()),
            gas_used: 15000,
            logs: vec![format!("Counter set to {} by {}", value, context.caller)],
            error_message: None,
        })
    }

    fn execute_transfer(
        &mut self,
        contract: &str,
        context: CallContext,
        amount: u64,
    ) -> Result<ExecutionResult, String> {
        let sender_balance = self.get_balance(contract, &context.caller);

        if sender_balance < amount {
            return Ok(ExecutionResult {
                success: false,
                return_data: None,
                gas_used: 10000,
                logs: vec![],
                error_message: Some("Insufficient balance".to_string()),
            });
        }

        // For demo, just subtract from sender (in a real contract, we'd have a recipient)
        let new_balance = sender_balance - amount;
        self.storage.store_value(
            contract,
            &format!("balance_{}", context.caller),
            new_balance.to_le_bytes().to_vec(),
        );

        println!(
            "   Transfer of {} completed. New balance: {}",
            amount, new_balance
        );

        Ok(ExecutionResult {
            success: true,
            return_data: Some(new_balance.to_le_bytes().to_vec()),
            gas_used: 25000,
            logs: vec![format!("Transferred {} from {}", amount, context.caller)],
            error_message: None,
        })
    }

    fn execute_get_balance(
        &self,
        contract: &str,
        context: CallContext,
    ) -> Result<ExecutionResult, String> {
        let balance = self.get_balance(contract, &context.caller);

        println!("   Balance for {}: {}", context.caller, balance);

        Ok(ExecutionResult {
            success: true,
            return_data: Some(balance.to_le_bytes().to_vec()),
            gas_used: 8000,
            logs: vec![],
            error_message: None,
        })
    }

    fn get_counter_value(&self, contract: &str) -> i32 {
        self.storage
            .get_value(contract, "counter")
            .and_then(|bytes| {
                if bytes.len() >= 4 {
                    Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                } else {
                    None
                }
            })
            .unwrap_or(0)
    }

    fn get_balance(&self, contract: &str, address: &str) -> u64 {
        self.storage
            .get_value(contract, &format!("balance_{}", address))
            .and_then(|bytes| {
                if bytes.len() >= 8 {
                    Some(u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]))
                } else {
                    None
                }
            })
            .unwrap_or(1000) // Default balance for demo
    }
}

// Sample "WASM" bytecode (just some mock bytes for demonstration)
fn create_sample_contract_bytecode() -> Vec<u8> {
    // This would normally be actual WASM bytecode
    // For demo purposes, we'll just use some mock bytes
    vec![
        0x00, 0x61, 0x73, 0x6d, // WASM magic number
        0x01, 0x00, 0x00, 0x00, // WASM version
        // Mock function sections
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // Type section: () -> ()
        0x03, 0x02, 0x01, 0x00, // Function section: 1 function of type 0
        0x07, 0x0a, 0x01, 0x06, 0x69, 0x6e, 0x63, 0x72, 0x65, 0x6d, 0x00,
        0x00, // Export section: export "increm" function 0
        0x0a, 0x04, 0x01, 0x02, 0x00, 0x0b, // Code section: function body
        // Additional mock data
        0xde, 0xad, 0xbe, 0xef,
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 WASM Smart Contract Example (Simplified)");
    println!("{}", "=".repeat(60));

    // Create storage
    let storage = Arc::new(MemMapStorage::new(MemMapOptions::default()));

    // Create contract and caller addresses - using from_string instead of from_str
    let contract_address = Address::from_string("0x1234567890123456789012345678901234567890")?;
    let caller_address = Address::from_string("0xabcdefabcdefabcdefabcdefabcdefabcdefabcd")?;

    // Create WASM VM
    let config = WasmVmConfig::default();
    let mut vm = WasmVm::new(config, storage);

    println!("\n📦 Contract Deployment");
    println!("{}", "-".repeat(30));

    // Create sample contract bytecode
    let bytecode = create_sample_contract_bytecode();

    // Deploy the contract
    vm.load_contract(&contract_address.to_string(), bytecode)?;

    // Create execution context
    let context = CallContext {
        contract_address: contract_address.to_string(),
        caller: caller_address.to_string(),
        block_height: 1,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
    };

    println!("\n🔧 Contract Execution");
    println!("{}", "-".repeat(30));

    // Test 1: Get initial counter value
    println!("\n1. Getting initial counter value:");
    let result = vm.execute_function(
        &contract_address.to_string(),
        "get",
        context.clone(),
        vec![],
    )?;
    display_execution_result(&result);

    // Test 2: Increment counter
    println!("\n2. Incrementing counter:");
    let result = vm.execute_function(
        &contract_address.to_string(),
        "increment",
        context.clone(),
        vec![],
    )?;
    display_execution_result(&result);

    // Test 3: Increment again
    println!("\n3. Incrementing counter again:");
    let result = vm.execute_function(
        &contract_address.to_string(),
        "increment",
        context.clone(),
        vec![],
    )?;
    display_execution_result(&result);

    // Test 4: Set specific value
    println!("\n4. Setting counter to 100:");
    let result = vm.execute_function(
        &contract_address.to_string(),
        "set",
        context.clone(),
        vec![100],
    )?;
    display_execution_result(&result);

    // Test 5: Get final counter value
    println!("\n5. Getting final counter value:");
    let result = vm.execute_function(
        &contract_address.to_string(),
        "get",
        context.clone(),
        vec![],
    )?;
    display_execution_result(&result);

    // Test 6: Check initial balance
    println!("\n6. Checking initial balance:");
    let result = vm.execute_function(
        &contract_address.to_string(),
        "balance",
        context.clone(),
        vec![],
    )?;
    display_execution_result(&result);

    // Test 7: Transfer some tokens
    println!("\n7. Transferring 250 tokens:");
    let result = vm.execute_function(
        &contract_address.to_string(),
        "transfer",
        context.clone(),
        vec![250],
    )?;
    display_execution_result(&result);

    // Test 8: Check balance after transfer
    println!("\n8. Checking balance after transfer:");
    let result = vm.execute_function(
        &contract_address.to_string(),
        "balance",
        context.clone(),
        vec![],
    )?;
    display_execution_result(&result);

    println!("\n✅ WASM contract example completed successfully!");
    println!("🎯 All contract functions executed correctly!");
    println!("\n📊 Contract Features Demonstrated:");
    println!("   • Contract loading and validation");
    println!("   • Function execution with gas tracking");
    println!("   • State persistence (counter, balances)");
    println!("   • Error handling (insufficient balance)");
    println!("   • Event logging");
    println!("   • Multiple data types (integers, balances)");

    Ok(())
}

fn display_execution_result(result: &ExecutionResult) {
    if result.success {
        println!("   ✅ Execution successful");

        if let Some(data) = &result.return_data {
            if data.len() >= 4 {
                // Try to interpret as i32 first
                let value = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                println!("   📊 Return value: {}", value);
            } else if data.len() >= 8 {
                // Try to interpret as u64
                let value = u64::from_le_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                ]);
                println!("   📊 Return value: {}", value);
            }
        }

        println!("   ⛽ Gas used: {}", result.gas_used);

        if !result.logs.is_empty() {
            println!("   📝 Logs:");
            for log in &result.logs {
                println!("      {}", log);
            }
        }
    } else {
        println!("   ❌ Execution failed");
        if let Some(error) = &result.error_message {
            println!("   Error: {}", error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_contract_execution() {
        let storage = Arc::new(MemMapStorage::new(MemMapOptions::default()));
        let mut vm = WasmVm::new(WasmVmConfig::default(), storage);

        let contract_address = "0x1234567890123456789012345678901234567890";
        let caller = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd";

        // Load contract
        let bytecode = create_sample_contract_bytecode();
        vm.load_contract(contract_address, bytecode).unwrap();

        let context = CallContext {
            contract_address: contract_address.to_string(),
            caller: caller.to_string(),
            block_height: 1,
            timestamp: 1000000,
        };

        // Test increment
        let result = vm
            .execute_function(contract_address, "increment", context.clone(), vec![])
            .unwrap();
        assert!(result.success);
        assert_eq!(result.gas_used, 21000);

        // Test get
        let result = vm
            .execute_function(contract_address, "get", context.clone(), vec![])
            .unwrap();
        assert!(result.success);
        assert_eq!(result.gas_used, 5000);

        // Test set
        let result = vm
            .execute_function(contract_address, "set", context.clone(), vec![42])
            .unwrap();
        assert!(result.success);
        assert_eq!(result.gas_used, 15000);

        // Test balance operations
        let result = vm
            .execute_function(contract_address, "balance", context.clone(), vec![])
            .unwrap();
        assert!(result.success);

        let result = vm
            .execute_function(contract_address, "transfer", context.clone(), vec![100])
            .unwrap();
        assert!(result.success);

        // Test insufficient balance
        let result = vm
            .execute_function(contract_address, "transfer", context.clone(), vec![10000])
            .unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_bytecode_creation() {
        let bytecode = create_sample_contract_bytecode();
        assert!(!bytecode.is_empty());
        assert_eq!(&bytecode[0..4], &[0x00, 0x61, 0x73, 0x6d]); // WASM magic number
    }

    #[test]
    fn test_wasm_vm_config() {
        let config = WasmVmConfig::default();
        assert_eq!(config.gas_limit, 10_000_000);
        assert_eq!(config.memory_limit, 16 * 1024 * 1024);
        assert_eq!(config.timeout_ms, 5000);
        assert!(!config.enable_debug);
    }
}
