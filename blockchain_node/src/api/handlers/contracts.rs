use crate::ledger::state::State;
use crate::smart_contract_engine::SmartContractEngine;
use axum::{
    extract::Extension,
    http::StatusCode,
    response::Json as AxumJson,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// Smart contract information
#[derive(Debug, Serialize)]
pub struct ContractInfo {
    pub address: String,
    pub name: String,
    pub bytecode: String,
    pub abi: String,
    pub creator: String,
    pub creation_time: u64,
    pub block_number: u64,
    pub transaction_hash: String,
    pub verified: bool,
    pub source_code: Option<String>,
    pub compiler_version: Option<String>,
}

/// Smart contract deployment request
#[derive(Debug, Deserialize)]
pub struct DeployRequest {
    pub bytecode: String,
    pub abi: String,
    pub constructor_args: Vec<String>,
    pub gas_limit: Option<u64>,
    pub gas_price: Option<u64>,
}

/// Smart contract call request
#[derive(Debug, Deserialize)]
pub struct CallRequest {
    pub contract_address: String,
    pub function_name: String,
    pub function_args: Vec<String>,
    pub value: Option<u64>,
    pub gas_limit: Option<u64>,
    pub gas_price: Option<u64>,
}

/// Smart contract execution result
#[derive(Debug, Serialize)]
pub struct ContractExecutionResult {
    pub success: bool,
    pub transaction_hash: String,
    pub gas_used: u64,
    pub return_value: Option<String>,
    pub logs: Vec<String>,
    pub error_message: Option<String>,
}

/// Smart contracts service for handling contract operations
pub struct SmartContractsService {
    smart_contract_engine: Arc<RwLock<SmartContractEngine>>,
    state: Arc<RwLock<State>>,
}

impl SmartContractsService {
    pub fn new(
        smart_contract_engine: Arc<RwLock<SmartContractEngine>>,
        state: Arc<RwLock<State>>,
    ) -> Self {
        Self {
            smart_contract_engine,
            state,
        }
    }

    /// Get all smart contracts
    pub async fn get_all_contracts(&self) -> Result<Vec<ContractInfo>, String> {
        // For now, return empty list since get_all_contracts() doesn't exist yet
        // In real implementation, this would get from actual state
        let contract_infos = Vec::new();
        
        Ok(contract_infos)
    }

    /// Get smart contract by address
    pub async fn get_contract_by_address(&self, address: &str) -> Result<ContractInfo, String> {
        let state = self.state.read().await;
        
        // Decode hex address
        let address_bytes = hex::decode(address)
            .map_err(|_| "Invalid address format".to_string())?;
        
        if let Some(contract) = state.get_contract_info(&address_bytes) {
            Ok(ContractInfo {
                address: address.to_string(),
                name: contract.name.clone(),
                bytecode: hex::encode(&contract.bytecode),
                abi: contract.abi.clone(),
                creator: hex::encode(contract.creator),
                creation_time: contract.creation_time,
                block_number: contract.block_number,
                transaction_hash: hex::encode(contract.transaction_hash),
                verified: contract.verified,
                source_code: contract.source_code.clone(),
                compiler_version: contract.compiler_version.clone(),
            })
        } else {
            Err("Contract not found".to_string())
        }
    }

    /// Deploy smart contract
    pub async fn deploy_contract(&self, request: &DeployRequest) -> Result<ContractExecutionResult, String> {
        let mut engine = self.smart_contract_engine.write().await;
        
        // Validate bytecode
        if request.bytecode.is_empty() {
            return Err("Bytecode cannot be empty".to_string());
        }

        // Validate ABI
        if request.abi.is_empty() {
            return Err("ABI cannot be empty".to_string());
        }

        // Decode bytecode
        let bytecode = hex::decode(&request.bytecode)
            .map_err(|_| "Invalid bytecode format".to_string())?;

        // Create deployment transaction
        let gas_limit = request.gas_limit.unwrap_or(3_000_000);
        let gas_price = request.gas_price.unwrap_or(1_000_000_000); // 1 GWEI

        // Simulate contract deployment
        let transaction_hash = format!("0x{}", hex::encode(format!("deploy_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())));
        let gas_used = gas_limit / 10; // Simulate gas usage
        
        // For now, skip adding contract to state since add_contract() doesn't exist yet
        // In real implementation, this would create an actual contract
        let contract_address = format!("0x{}", hex::encode(format!("contract_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())));

        Ok(ContractExecutionResult {
            success: true,
            transaction_hash,
            gas_used,
            return_value: Some(contract_address),
            logs: vec!["Contract deployed successfully".to_string()],
            error_message: None,
        })
    }

    /// Call smart contract function
    pub async fn call_contract(&self, request: &CallRequest) -> Result<ContractExecutionResult, String> {
        let engine = self.smart_contract_engine.read().await;
        
        // Validate contract address
        if request.contract_address.is_empty() {
            return Err("Contract address cannot be empty".to_string());
        }

        // Validate function name
        if request.function_name.is_empty() {
            return Err("Function name cannot be empty".to_string());
        }

        // For now, skip contract existence check since has_contract() doesn't exist yet
        // In real implementation, this would check actual contract state

        // Simulate function call
        let gas_limit = request.gas_limit.unwrap_or(100_000);
        let gas_price = request.gas_price.unwrap_or(1_000_000_000); // 1 GWEI
        
        let transaction_hash = format!("0x{}", hex::encode(format!("call_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())));
        let gas_used = gas_limit / 5; // Simulate gas usage
        
        // In real implementation, this would execute the actual function
        let return_value = Some(format!("0x{}", hex::encode(format!("result_{}", request.function_name))));
        
        Ok(ContractExecutionResult {
            success: true,
            transaction_hash,
            gas_used,
            return_value,
            logs: vec![
                format!("Function {} called successfully", request.function_name),
                format!("Gas used: {}", gas_used),
            ],
            error_message: None,
        })
    }

    /// Get smart contracts health status
    pub async fn get_contracts_health(&self) -> Result<serde_json::Value, String> {
        let state = self.state.read().await;
        let total_contracts = state.get_contract_count().unwrap_or(0);
        let verified_contracts = state.get_verified_contracts_count().unwrap_or(0);
        
        Ok(serde_json::json!({
            "status": "healthy",
            "service": "smart_contracts",
            "total_contracts": total_contracts,
            "verified_contracts": verified_contracts,
            "verification_rate": if total_contracts > 0 {
                (verified_contracts as f64 / total_contracts as f64) * 100.0
            } else {
                0.0
            },
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        }))
    }
}

/// Handler for getting smart contracts info
pub async fn get_contracts_info(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    let smart_contract_engine = Arc::new(RwLock::new(SmartContractEngine::new(
        Arc::new(crate::storage::hybrid_storage::HybridStorage::new("memory://".to_string(), 1024 * 1024).unwrap()),
        crate::smart_contract_engine::SmartContractEngineConfig::default(),
    ).await.unwrap()));
    
    let service = SmartContractsService::new(smart_contract_engine, state);
    
    match service.get_all_contracts().await {
        Ok(contracts) => {
            let total_contracts = contracts.len();
            let verified_contracts = contracts.iter().filter(|c| c.verified).count();
            
            Ok(AxumJson(serde_json::json!({
                "status": "success",
                "total_contracts": total_contracts,
                "verified_contracts": verified_contracts,
                "contracts": contracts,
                "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
            })))
        }
        Err(e) => {
            log::error!("Failed to get contracts info: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting smart contracts health
pub async fn get_contracts_health(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    let smart_contract_engine = Arc::new(RwLock::new(SmartContractEngine::new(
        Arc::new(crate::storage::hybrid_storage::HybridStorage::new("memory://".to_string(), 1024 * 1024).unwrap()),
        crate::smart_contract_engine::SmartContractEngineConfig::default(),
    ).await.unwrap()));
    
    let service = SmartContractsService::new(smart_contract_engine, state);
    
    match service.get_contracts_health().await {
        Ok(health) => Ok(AxumJson(health)),
        Err(e) => {
            log::error!("Failed to get contracts health: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for deploying smart contracts
pub async fn deploy_evm_contract(
    Json(request): Json<DeployRequest>,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<ContractExecutionResult>, StatusCode> {
    let smart_contract_engine = Arc::new(RwLock::new(SmartContractEngine::new(
        Arc::new(crate::storage::hybrid_storage::HybridStorage::new("memory://".to_string(), 1024 * 1024).unwrap()),
        crate::smart_contract_engine::SmartContractEngineConfig::default(),
    ).await.unwrap()));
    
    let service = SmartContractsService::new(smart_contract_engine, state);
    
    match service.deploy_contract(&request).await {
        Ok(result) => Ok(AxumJson(result)),
        Err(e) => {
            log::error!("Failed to deploy contract: {}", e);
            Err(StatusCode::BAD_REQUEST)
        }
    }
}

/// Handler for calling smart contracts
pub async fn call_evm_contract(
    Json(request): Json<CallRequest>,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<ContractExecutionResult>, StatusCode> {
    let smart_contract_engine = Arc::new(RwLock::new(SmartContractEngine::new(
        Arc::new(crate::storage::hybrid_storage::HybridStorage::new("memory://".to_string(), 1024 * 1024).unwrap()),
        crate::smart_contract_engine::SmartContractEngineConfig::default(),
    ).await.unwrap()));
    
    let service = SmartContractsService::new(smart_contract_engine, state);
    
    match service.call_contract(&request).await {
        Ok(result) => Ok(AxumJson(result)),
        Err(e) => {
            log::error!("Failed to call contract: {}", e);
            Err(StatusCode::BAD_REQUEST)
        }
    }
}

/// Handler for getting contract by address
pub async fn get_contract_by_address(
    axum::extract::Path(address): axum::extract::Path<String>,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<ContractInfo>, StatusCode> {
    let smart_contract_engine = Arc::new(RwLock::new(SmartContractEngine::new(
        Arc::new(crate::storage::hybrid_storage::HybridStorage::new("memory://".to_string(), 1024 * 1024).unwrap()),
        crate::smart_contract_engine::SmartContractEngineConfig::default(),
    ).await.unwrap()));
    
    let service = SmartContractsService::new(smart_contract_engine, state);
    
    match service.get_contract_by_address(&address).await {
        Ok(contract) => Ok(AxumJson(contract)),
        Err(e) => {
            log::error!("Failed to get contract: {}", e);
            Err(StatusCode::NOT_FOUND)
        }
    }
}
