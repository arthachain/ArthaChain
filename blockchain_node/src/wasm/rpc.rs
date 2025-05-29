//! WASM RPC Service
//!
//! Provides API endpoints for interacting with WebAssembly smart contracts.

use axum::{
    extract::{Json, Path, State},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::types::Address;
use crate::wasm::executor::WasmExecutor;
use crate::wasm::types::{WasmContractAddress, WasmExecutionResult, WasmTransaction};

/// WASM RPC service state
pub struct WasmRpcService {
    /// Executor for WASM contracts
    executor: Arc<RwLock<WasmExecutor>>,
}

/// Contract code deployment request
#[derive(Debug, Serialize, Deserialize)]
pub struct DeployRequest {
    /// Contract bytecode (hex encoded)
    pub bytecode: String,
    /// Constructor arguments (hex encoded)
    pub constructor_args: Option<String>,
    /// Deployer address
    pub deployer: String,
    /// Gas limit
    pub gas_limit: u64,
}

/// Contract function call request
#[derive(Debug, Serialize, Deserialize)]
pub struct CallRequest {
    /// Contract address
    pub contract_address: String,
    /// Function name
    pub function: String,
    /// Function arguments (hex encoded)
    pub args: Option<String>,
    /// Caller address
    pub caller: String,
    /// Value to send
    pub value: Option<u64>,
    /// Gas limit
    pub gas_limit: u64,
}

/// Contract view function call request
#[derive(Debug, Serialize, Deserialize)]
pub struct ViewRequest {
    /// Contract address
    pub contract_address: String,
    /// Function name
    pub function: String,
    /// Function arguments (hex encoded)
    pub args: Option<String>,
    /// Caller address
    pub caller: String,
}

/// Contract storage read request
#[derive(Debug, Serialize, Deserialize)]
pub struct StorageReadRequest {
    /// Contract address
    pub contract_address: String,
    /// Storage key (hex encoded)
    pub key: String,
}

/// Contract information response
#[derive(Debug, Serialize, Deserialize)]
pub struct ContractInfoResponse {
    /// Contract address
    pub address: String,
    /// Whether the contract exists
    pub exists: bool,
    /// Total call count
    pub call_count: Option<u64>,
    /// Average gas used
    pub avg_gas_used: Option<u64>,
}

/// Storage read response
#[derive(Debug, Serialize, Deserialize)]
pub struct StorageReadResponse {
    /// Contract address
    pub contract_address: String,
    /// Storage key (hex encoded)
    pub key: String,
    /// Storage value (hex encoded)
    pub value: Option<String>,
}

impl WasmRpcService {
    /// Create a new WASM RPC service
    pub fn new(executor: Arc<RwLock<WasmExecutor>>) -> Self {
        Self { executor }
    }

    /// Create the API router
    pub fn router(self) -> Router {
        Router::new()
            .route("/wasm/deploy", post(Self::deploy_contract))
            .route("/wasm/call", post(Self::call_contract))
            .route("/wasm/view", post(Self::view_contract))
            .route("/wasm/storage", post(Self::read_storage))
            .route("/wasm/contract/:address", get(Self::get_contract_info))
            .with_state(Arc::new(self))
    }

    /// Deploy a new contract
    async fn deploy_contract(
        State(state): State<Arc<Self>>,
        Json(request): Json<DeployRequest>,
    ) -> Result<Json<WasmExecutionResult>, String> {
        // Decode bytecode from hex
        let bytecode =
            hex::decode(&request.bytecode).map_err(|e| format!("Invalid bytecode hex: {}", e))?;

        // Decode constructor args if provided
        let constructor_args = if let Some(args_hex) = request.constructor_args {
            Some(
                hex::decode(&args_hex)
                    .map_err(|e| format!("Invalid constructor args hex: {}", e))?,
            )
        } else {
            None
        };

        // Parse deployer address
        let deployer = Address::from_string(&request.deployer)
            .map_err(|_| format!("Invalid deployer address: {}", request.deployer))?;

        // Create a transaction
        let transaction = WasmTransaction::new_deployment(
            deployer,
            bytecode,
            constructor_args,
            request.gas_limit,
        );

        // Execute the transaction
        let mut executor = state.executor.write().await;
        executor
            .deploy(&transaction)
            .map(Json)
            .map_err(|e| format!("Deployment failed: {}", e))
    }

    /// Call a contract function
    async fn call_contract(
        State(state): State<Arc<Self>>,
        Json(request): Json<CallRequest>,
    ) -> Result<Json<WasmExecutionResult>, String> {
        // Parse contract address
        let contract_address = WasmContractAddress::from_string(&request.contract_address);

        // Parse caller address
        let caller = Address::from_string(&request.caller)
            .map_err(|_| format!("Invalid caller address: {}", request.caller))?;

        // Decode function args if provided
        let function_args = if let Some(args_hex) = request.args {
            Some(hex::decode(&args_hex).map_err(|e| format!("Invalid function args hex: {}", e))?)
        } else {
            None
        };

        // Create a transaction
        let transaction = WasmTransaction::new_call(
            caller,
            contract_address,
            request.function,
            function_args,
            request.value,
            request.gas_limit,
        );

        // Execute the transaction
        let mut executor = state.executor.write().await;
        executor
            .execute(&transaction)
            .map(Json)
            .map_err(|e| format!("Execution failed: {}", e))
    }

    /// Call a contract view function
    async fn view_contract(
        State(state): State<Arc<Self>>,
        Json(request): Json<ViewRequest>,
    ) -> Result<Json<serde_json::Value>, String> {
        // Parse contract address
        let contract_address = WasmContractAddress::from_string(&request.contract_address);

        // Parse caller address
        let caller = Address::from_string(&request.caller)
            .map_err(|_| format!("Invalid caller address: {}", request.caller))?;

        // Decode function args if provided
        let args = if let Some(args_hex) = request.args {
            hex::decode(&args_hex).map_err(|e| format!("Invalid function args hex: {}", e))?
        } else {
            Vec::new()
        };

        // Execute view call
        let executor = state.executor.read().await;
        let result = executor
            .execute_view(&contract_address, &request.function, &args, &caller)
            .map_err(|e| format!("View execution failed: {}", e))?;

        // Format result
        let result_json = if result.succeeded {
            if let Some(data) = result.data {
                // Try to parse as JSON
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&data) {
                    json
                } else {
                    // Fallback to hex encoding
                    serde_json::json!({
                        "data": format!("0x{}", hex::encode(&data)),
                        "gas_used": result.gas_used
                    })
                }
            } else {
                serde_json::json!({
                    "gas_used": result.gas_used
                })
            }
        } else {
            serde_json::json!({
                "error": result.error.unwrap_or_else(|| "Unknown error".to_string()),
                "gas_used": result.gas_used
            })
        };

        Ok(Json(result_json))
    }

    /// Read contract storage
    async fn read_storage(
        State(state): State<Arc<Self>>,
        Json(request): Json<StorageReadRequest>,
    ) -> Result<Json<StorageReadResponse>, String> {
        // Parse contract address
        let contract_address = WasmContractAddress::from_string(&request.contract_address);

        // Decode key from hex
        let key = hex::decode(&request.key).map_err(|e| format!("Invalid key hex: {}", e))?;

        // Read from storage
        let executor = state.executor.read().await;
        let value = executor
            .read_contract_storage(&contract_address, &key)
            .map_err(|e| format!("Storage read failed: {}", e))?;

        // Format result
        let value_hex = value.map(|v| format!("0x{}", hex::encode(&v)));

        Ok(Json(StorageReadResponse {
            contract_address: contract_address.to_string(),
            key: format!("0x{}", hex::encode(&key)),
            value: value_hex,
        }))
    }

    /// Get contract information
    async fn get_contract_info(
        State(state): State<Arc<Self>>,
        Path(address): Path<String>,
    ) -> Result<Json<ContractInfoResponse>, String> {
        // Parse contract address
        let contract_address = WasmContractAddress::from_string(&address);

        // Check if contract exists
        let executor = state.executor.read().await;
        let exists = executor.contract_exists(&contract_address);

        // Get metrics if available
        let (call_count, avg_gas_used) = if exists {
            if let Some((count, gas)) = executor.get_contract_metrics(&contract_address) {
                (Some(count), Some(gas))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Format result
        Ok(Json(ContractInfoResponse {
            address: contract_address.to_string(),
            exists,
            call_count,
            avg_gas_used,
        }))
    }
}

impl Into<Router> for WasmRpcService {
    fn into(self) -> Router {
        self.router()
    }
}
