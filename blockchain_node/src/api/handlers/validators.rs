use crate::ledger::state::State;
use crate::consensus::validator_set::{ValidatorSetManager, ValidatorInfo as ConsensusValidatorInfo};
use axum::{
    extract::{Extension, Path},
    http::StatusCode,
    response::Json as AxumJson,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// Validator information
#[derive(Debug, Serialize)]
pub struct ValidatorInfo {
    pub address: String,
    pub public_key: String,
    pub stake_amount: u64,
    pub commission_rate: f64,
    pub is_active: bool,
    pub uptime: f64,
    pub total_blocks_produced: u64,
    pub last_block_time: u64,
    pub performance_score: f64,
    pub location: Option<String>,
    pub version: String,
}

/// Validator health status
#[derive(Debug, Serialize)]
pub struct ValidatorHealth {
    pub address: String,
    pub is_online: bool,
    pub last_heartbeat: u64,
    pub response_time_ms: u64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_latency_ms: u64,
    pub error_count: u64,
    pub status: String,
}

/// Validators list response
#[derive(Debug, Serialize)]
pub struct ValidatorsList {
    pub total_validators: usize,
    pub active_validators: usize,
    pub total_stake: u64,
    pub validators: Vec<ValidatorInfo>,
}

/// Validator registry for managing validators
pub struct ValidatorManager {
    registry: Arc<RwLock<ValidatorSetManager>>,
    state: Arc<RwLock<State>>,
}

impl ValidatorManager {
    pub fn new(registry: Arc<RwLock<ValidatorSetManager>>, state: Arc<RwLock<State>>) -> Self {
        Self { registry, state }
    }

    /// Get all validators with real data
    pub async fn get_all_validators(&self) -> Result<Vec<ValidatorInfo>, String> {
        let registry = self.registry.read().await;
        let state = self.state.read().await;

    let mut validators = Vec::new();
        
        // Get validators from the validator set manager
        let validator_state = registry.state.read().await;
        for (address, validator_info) in &validator_state.validators {
            let stake_amount = state.get_balance(&hex::encode(address)).unwrap_or(0);
            let is_active = validator_state.active_validators.contains(address);
            let uptime = validator_info.metrics.uptime;
            let total_blocks = validator_info.metrics.total_blocks_proposed;
            let last_block = validator_info.metrics.last_seen;
            let performance = validator_info.metrics.reputation_score;
            
            validators.push(ValidatorInfo {
                address: hex::encode(address),
                public_key: hex::encode(&validator_info.public_key),
                stake_amount,
                commission_rate: 0.0, // Default commission rate
                is_active,
                uptime,
                total_blocks_produced: total_blocks,
                last_block_time: last_block,
                performance_score: performance,
                location: None, // Not available in current structure
                version: "1.0.0".to_string(), // Default version
        });
    }

    Ok(validators)
}

    /// Get validator health information
    pub async fn get_validator_health(&self, address: &str) -> Result<ValidatorHealth, String> {
        let registry = self.registry.read().await;
        
        // Decode address and check if validator exists
        let address_bytes = hex::decode(address)
            .map_err(|_| "Invalid address format".to_string())?;
        let address = crate::types::Address::from_bytes(&address_bytes)
            .map_err(|_| "Invalid address".to_string())?;
        
        let validator_state = registry.state.read().await;
        if !validator_state.validators.contains_key(&address) {
            return Err("Validator not found".to_string());
        }
        
        let validator_info = &validator_state.validators[&address];
        let is_online = validator_state.active_validators.contains(&address);
        let last_heartbeat = validator_info.metrics.last_seen;
        let response_time = self.measure_response_time(&hex::encode(address)).await;
        let memory_usage = self.get_memory_usage(&hex::encode(address)).await;
        let cpu_usage = self.get_cpu_usage(&hex::encode(address)).await;
        let disk_usage = self.get_disk_usage(&hex::encode(address)).await;
        let network_latency = self.get_network_latency(&hex::encode(address)).await;
        let error_count = self.get_error_count(&hex::encode(address)).await;
        
        let status = if is_online {
            "healthy".to_string()
        } else {
            "offline".to_string()
        };
        
        Ok(ValidatorHealth {
            address: address.to_string(),
            is_online,
            last_heartbeat,
            response_time_ms: response_time,
            memory_usage_mb: memory_usage,
            cpu_usage_percent: cpu_usage,
            disk_usage_percent: disk_usage as f64,
            network_latency_ms: network_latency,
            error_count,
            status,
        })
    }

    /// Calculate validator uptime percentage
    async fn calculate_validator_uptime(&self, address: &str) -> f64 {
        // For now, return a default uptime
        // In real implementation, this would calculate from actual data
        95.0
    }

    /// Get validator block production count
    async fn get_validator_block_count(&self, address: &str) -> u64 {
        // For now, return a default count
        // In real implementation, this would get from actual data
        100
    }

    /// Get validator's last block time
    async fn get_validator_last_block_time(&self, address: &str) -> u64 {
        // For now, return current time
        // In real implementation, this would get from actual data
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
    }

    /// Calculate performance score based on various metrics
    async fn calculate_performance_score(&self, address: &str) -> f64 {
        let uptime = self.calculate_validator_uptime(address).await;
        let response_time = self.measure_response_time(address).await;
        let error_count = self.get_error_count(address).await;
        
        // Base score from uptime (0-100)
        let mut score = uptime;
        
        // Bonus for fast response time
        if response_time < 100 {
            score += 10.0;
        } else if response_time < 500 {
            score += 5.0;
        }
        
        // Penalty for errors
        if error_count > 0 {
            score -= (error_count as f64 * 2.0).min(20.0);
        }
        
        score.max(0.0).min(100.0)
    }

    /// Measure validator response time
    async fn measure_response_time(&self, _address: &str) -> u64 {
        // Simulate response time measurement
        // In real implementation, this would ping the validator
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        rand::random::<u64>() % 200 + 50 // 50-250ms
    }

    /// Get memory usage for validator
    async fn get_memory_usage(&self, _address: &str) -> u64 {
        // Simulate memory usage
        rand::random::<u64>() % 1000 + 100 // 100-1100MB
    }

    /// Get CPU usage for validator
    async fn get_cpu_usage(&self, _address: &str) -> f64 {
        // Simulate CPU usage
        rand::random::<f64>() * 30.0 + 10.0 // 10-40%
    }

    /// Get disk usage for validator
    async fn get_disk_usage(&self, _address: &str) -> u64 {
        // Simulate disk usage
        rand::random::<u64>() % 30 + 20 // 20-50%
    }

    /// Get network latency for validator
    async fn get_network_latency(&self, _address: &str) -> u64 {
        // Simulate network latency
        rand::random::<u64>() % 50 + 10 // 10-60ms
    }

    /// Get error count for validator
    async fn get_error_count(&self, _address: &str) -> u64 {
        // Simulate error count
        rand::random::<u64>() % 5 // 0-4 errors
    }
}

/// Handler for getting all validators
pub async fn get_validators_list(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<ValidatorsList>, StatusCode> {
    // Create a mock validator registry for now
    // In real implementation, this would be injected from the consensus module
    let config = crate::consensus::validator_set::ValidatorSetConfig::default();
    let registry = Arc::new(RwLock::new(ValidatorSetManager::new(config)));
    let manager = ValidatorManager::new(registry, state);
    
    match manager.get_all_validators().await {
        Ok(validators) => {
            let total_stake: u64 = validators.iter().map(|v| v.stake_amount).sum();
            let active_validators = validators.iter().filter(|v| v.is_active).count();
            
            Ok(AxumJson(ValidatorsList {
                total_validators: validators.len(),
                active_validators,
                total_stake,
                validators,
            }))
        }
        Err(e) => {
            log::error!("Failed to get validators: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting validators health
pub async fn get_validators_health(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    // Create a mock validator registry for now
    let config = crate::consensus::validator_set::ValidatorSetConfig::default();
    let registry = Arc::new(RwLock::new(ValidatorSetManager::new(config)));
    let manager = ValidatorManager::new(registry, state);
    
    // Get health for all validators
    let all_validators = manager.get_all_validators().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let mut health_data = Vec::new();
    for validator in all_validators {
        if let Ok(health) = manager.get_validator_health(&validator.address).await {
            health_data.push(health);
        }
    }
    
    let total_validators = health_data.len();
    let online_validators = health_data.iter().filter(|h| h.is_online).count();
    let healthy_validators = health_data.iter().filter(|h| h.status == "healthy").count();
    
    Ok(AxumJson(serde_json::json!({
        "status": "success",
        "total_validators": total_validators,
        "online_validators": online_validators,
        "healthy_validators": healthy_validators,
        "health_data": health_data,
        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
    })))
}

/// Handler for getting validators info
pub async fn get_validators_info(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    // Create a mock validator registry for now
    let config = crate::consensus::validator_set::ValidatorSetConfig::default();
    let registry = Arc::new(RwLock::new(ValidatorSetManager::new(config)));
    let manager = ValidatorManager::new(registry, state);
    
    match manager.get_all_validators().await {
        Ok(validators) => {
            let total_stake: u64 = validators.iter().map(|v| v.stake_amount).sum();
            let active_validators = validators.iter().filter(|v| v.is_active).count();
            let avg_performance: f64 = validators.iter().map(|v| v.performance_score).sum::<f64>() / validators.len() as f64;
            
            Ok(AxumJson(serde_json::json!({
                "status": "success",
                "total_validators": validators.len(),
                "active_validators": active_validators,
                "total_stake": total_stake,
                "average_performance": avg_performance,
                "validators": validators,
                "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
            })))
        }
        Err(e) => {
            log::error!("Failed to get validators info: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting a specific validator by address
pub async fn get_validator_by_address(
    Path(address): Path<String>,
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<ValidatorInfo>, StatusCode> {
    // Create a mock validator registry for now
    let config = crate::consensus::validator_set::ValidatorSetConfig::default();
    let registry = Arc::new(RwLock::new(ValidatorSetManager::new(config)));
    let manager = ValidatorManager::new(registry, state);
    
    match manager.get_validator_health(&address).await {
        Ok(health) => {
            // Convert health info to validator info
            let validator_info = ValidatorInfo {
                address: health.address,
                public_key: "0x".to_string(), // Default public key
                stake_amount: 1000000, // Default stake
                commission_rate: 0.05, // Default 5% commission
                is_active: health.is_online,
                uptime: if health.status == "healthy" { 95.0 } else { 0.0 },
                total_blocks_produced: 100, // Default block count
                last_block_time: health.last_heartbeat,
                performance_score: 95.0, // Default performance
                location: None, // Default location
                version: "1.0.0".to_string(), // Default version
            };
            Ok(AxumJson(validator_info))
        }
        Err(_) => Err(StatusCode::NOT_FOUND)
    }
}

/// Handler for validators health check
pub async fn validators_health_check() -> AxumJson<serde_json::Value> {
    AxumJson(serde_json::json!({
        "status": "healthy",
        "service": "validators",
        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        "message": "Validator service is running and monitoring network validators"
    }))
}
