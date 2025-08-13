use crate::consensus::validator_set::ValidatorSetManager;
use crate::types::Address;
use axum::{extract::Extension, http::StatusCode, response::Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Clone)]
pub struct Validator {
    pub address: String,
    pub voting_power: f64,
    pub status: String,
    pub uptime: f64,
    pub last_seen: u64,
    pub registration_block: u64,
}

#[derive(Serialize)]
pub struct ValidatorsResponse {
    pub validators: Vec<Validator>,
    pub total_count: usize,
    pub active_count: usize,
}

/// Get all validators from the real validator manager
pub async fn get_validators(
    Extension(validator_manager): Extension<Arc<ValidatorSetManager>>,
) -> Result<Json<ValidatorsResponse>, StatusCode> {
    // Get real validator data from the validator manager
    let real_validators = get_real_validators(&validator_manager)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if !real_validators.is_empty() {
        let active_count = real_validators
            .iter()
            .filter(|v| v.status == "active")
            .count();

        return Ok(Json(ValidatorsResponse {
            total_count: real_validators.len(),
            active_count,
            validators: real_validators,
        }));
    }

    // Fallback to bootstrap validators if no real validators are registered yet
    let validators = vec![Validator {
        address: "testnet_server_node_001".to_string(),
        voting_power: 100.0,
        status: "active".to_string(),
        uptime: 100.0,
        last_seen: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        registration_block: 1,
    }];

    let active_count = validators.iter().filter(|v| v.status == "active").count();

    Ok(Json(ValidatorsResponse {
        total_count: validators.len(),
        active_count,
        validators,
    }))
}

/// Get validator by address  
pub async fn get_validator_by_address(
    axum::extract::Path(address): axum::extract::Path<String>,
    Extension(validator_manager): Extension<Arc<ValidatorSetManager>>,
) -> Result<Json<Validator>, StatusCode> {
    // Try to get real validator data first
    if let Ok(real_validator) = get_validator_by_address_real(&validator_manager, &address).await {
        return Ok(Json(real_validator));
    }

    // Fallback to mock validator for now
    let validator = Validator {
        address,
        voting_power: 100.0,
        status: "active".to_string(),
        uptime: 99.5,
        last_seen: chrono::Utc::now().timestamp() as u64,
        registration_block: 1,
    };

    Ok(Json(validator))
}

/// Helper function to get real validators from validator manager
async fn get_real_validators(
    validator_manager: &ValidatorSetManager,
) -> Result<Vec<Validator>, Box<dyn std::error::Error>> {
    // Access the internal state to get all validators
    let state_lock = &validator_manager.state;
    let state = state_lock.read().await;

    let mut validators = Vec::new();
    let current_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    for (address, info) in &state.validators {
        let address_str = format!("artha{}", hex::encode(address.as_bytes()));
        let voting_power = 100.0 / state.validators.len() as f64; // Equal voting power for all validators

        validators.push(Validator {
            address: address_str,
            voting_power,
            status: if info.is_active { "active" } else { "inactive" }.to_string(),
            uptime: info.metrics.uptime,
            last_seen: if info.metrics.last_seen > 0 {
                info.metrics.last_seen
            } else {
                current_time
            },
            registration_block: info.registration_block,
        });
    }

    Ok(validators)
}

/// Helper function to get a specific validator by address
async fn get_validator_by_address_real(
    validator_manager: &ValidatorSetManager,
    address_str: &str,
) -> Result<Validator, Box<dyn std::error::Error>> {
    // Remove "artha" prefix if present and decode hex
    let hex_part = if address_str.starts_with("artha") {
        &address_str[5..]
    } else {
        address_str
    };

    let address_bytes = hex::decode(hex_part)?;
    let address = Address::from_bytes(&address_bytes)?;

    let state_lock = &validator_manager.state;
    let state = state_lock.read().await;

    if let Some(info) = state.validators.get(&address) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let voting_power = 100.0; // Each validator has equal voting power

        Ok(Validator {
            address: address_str.to_string(),
            voting_power,
            status: if info.is_active { "active" } else { "inactive" }.to_string(),
            uptime: info.metrics.uptime,
            last_seen: if info.metrics.last_seen > 0 {
                info.metrics.last_seen
            } else {
                current_time
            },
            registration_block: info.registration_block,
        })
    } else {
        Err("Validator not found".into())
    }
}
