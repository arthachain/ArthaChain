// AI Engine modules will be implemented here
pub mod security;
pub mod device_health;
pub mod user_identification;
pub mod data_chunking;
pub mod fraud_detection;
pub mod explainability;

use std::sync::{Arc, Mutex};
use anyhow::{Result, Context};
use log::{info, warn, error};
use crate::config::Config;
use crate::ledger::state::State;
use std::path::PathBuf;
use tokio::sync::RwLock;

/// Central AI Engine that integrates all AI modules
pub struct AIEngine {
    /// Device Health AI module
    pub device_health: device_health::DeviceHealthAI,
    /// User Identification AI module
    pub user_identification: user_identification::UserIdentificationAI,
    /// Data Chunking AI module
    pub data_chunking: data_chunking::DataChunkingAI,
    /// Fraud Detection AI module
    pub fraud_detection: fraud_detection::FraudDetectionAI,
    /// Security AI module
    pub security_ai: Option<security::SecurityAI>,
    /// Whether the AI Engine is running
    running: Arc<Mutex<bool>>,
    /// AI models directory
    models_dir: PathBuf,
    /// Config reference
    config: Config,
}

impl AIEngine {
    /// Create a new AI Engine instance
    pub fn new(config: Config) -> Self {
        // Set default models directory if not provided
        let models_dir = if config.ai_model_dir.as_os_str().is_empty() {
            PathBuf::from("./models")
        } else {
            config.ai_model_dir.clone()
        };
        
        Self {
            device_health: device_health::DeviceHealthAI::new(&config),
            user_identification: user_identification::UserIdentificationAI::new(&config),
            data_chunking: data_chunking::DataChunkingAI::new(&config),
            fraud_detection: fraud_detection::FraudDetectionAI::new(&config),
            security_ai: None, // Will be initialized when state is available
            running: Arc::new(Mutex::new(false)),
            models_dir,
            config,
        }
    }
    
    /// Initialize the Security AI module with blockchain state
    pub fn init_security_ai(&mut self, state: Arc<RwLock<State>>) -> Result<()> {
        // Create the SecurityAI instance
        let security = security::SecurityAI::new(self.config.clone(), state)
            .context("Failed to initialize SecurityAI")?;
        
        self.security_ai = Some(security);
        Ok(())
    }
    
    /// Start the AI Engine and all its modules
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Ok(());
        }
        
        // Start each AI module
        if let Err(e) = self.device_health.start().await {
            error!("Failed to start Device Health AI: {}", e);
        }
        
        // Start security AI if initialized
        if let Some(security) = &mut self.security_ai {
            if let Err(e) = security.start().await {
                error!("Failed to start Security AI: {}", e);
            }
        } else {
            warn!("Security AI not initialized, skipping start");
        }
        
        // Other modules don't require explicit starting but could be enhanced
        
        *running = true;
        info!("AI Engine started successfully");
        Ok(())
    }
    
    /// Stop the AI Engine and all its modules
    pub fn stop(&self) {
        let mut running = self.running.lock().unwrap();
        if !*running {
            return;
        }
        
        // Stop each AI module
        self.device_health.stop();
        
        // Security AI is stopped via its JoinHandle
        
        *running = false;
        info!("AI Engine stopped");
    }
    
    /// Update AI models for all modules
    pub async fn update_models(&mut self) -> Result<()> {
        info!("Updating AI models for all modules");
        
        // Ensure models directory exists
        if !self.models_dir.exists() {
            std::fs::create_dir_all(&self.models_dir)?;
        }
        
        // Update each AI module's model
        let device_health_model = self.models_dir.join("device_health_model.bin");
        if let Err(e) = self.device_health.update_model(device_health_model.to_str().unwrap()).await {
            warn!("Failed to update Device Health AI model: {}", e);
        }
        
        let user_id_model = self.models_dir.join("user_identification_model.bin");
        if let Err(e) = self.user_identification.update_model(user_id_model.to_str().unwrap()).await {
            warn!("Failed to update User Identification AI model: {}", e);
        }
        
        let data_chunking_model = self.models_dir.join("data_chunking_model.bin");
        if let Err(e) = self.data_chunking.update_model(data_chunking_model.to_str().unwrap()).await {
            warn!("Failed to update Data Chunking AI model: {}", e);
        }
        
        let fraud_detection_model = self.models_dir.join("fraud_detection_model.bin");
        if let Err(e) = self.fraud_detection.update_model(fraud_detection_model.to_str().unwrap()).await {
            warn!("Failed to update Fraud Detection AI model: {}", e);
        }
        
        // Security AI has its own model reload mechanism
        
        info!("AI model updates completed");
        Ok(())
    }
    
    /// Get participation eligibility and weight for a node
    pub fn get_participation_info(&self, node_id: &str) -> (bool, f32) {
        // Check device health
        let device_eligible = self.device_health.is_eligible_for_validation();
        let device_weight = self.device_health.get_participation_weight();
        
        // Check security score
        let security_weight = match self.fraud_detection.get_security_score(node_id) {
            Some(score) => score.overall_score,
            None => 0.7, // Default if no score exists
        };
        
        // Check if banned
        let is_banned = self.fraud_detection.is_banned(node_id);
        
        // Calculate eligibility and weight
        let is_eligible = device_eligible && !is_banned;
        let weight = device_weight * security_weight;
        
        (is_eligible, weight)
    }
    
    /// Get an overall SVCP reputation score for a node (0.0-1.0)
    pub async fn get_svcp_score(&self, node_id: &str) -> f32 {
        // Combine scores from different AI modules for SVCP
        let device_score = self.device_health.get_score();
        
        // Get security score or default
        let security_score = match self.fraud_detection.get_security_score(node_id) {
            Some(score) => score.overall_score,
            None => 0.7,
        };
        
        // Get SecurityAI score if available
        let ai_security_score = if let Some(security) = &self.security_ai {
            // Create a default metrics object for evaluation
            let metrics = security::NodeMetrics {
                device_health: security::DeviceHealthMetrics {
                    cpu_usage: 50.0,
                    memory_usage: 50.0,
                    disk_available: 10 * 1024 * 1024 * 1024, // 10 GB
                    num_cores: 4,
                    uptime: 3600,  // 1 hour
                    os_info: "Linux".to_string(),
                    avg_response_time: 100.0,
                    dropped_connections: 0,
                    temperature: Some(45.0),
                },
                network: security::NetworkMetrics {
                    bandwidth_usage: 1024 * 1024, // 1 MB/s
                    latency: 100.0,
                    packet_loss: 0.01,
                    connection_stability: 0.95,
                    peer_count: 10,
                    geo_consistency: 0.9,
                    p2p_score: 0.85,
                    sync_status: 1.0,
                },
                storage: security::StorageMetrics {
                    storage_provided: 100 * 1024 * 1024 * 1024, // 100 GB
                    storage_utilization: 0.7,
                    retrieval_success_rate: 0.95,
                    avg_retrieval_time: 200.0,
                    redundancy_level: 3.0,
                    integrity_violations: 0,
                    storage_uptime: 0.99,
                    storage_growth_rate: 10 * 1024 * 1024, // 10 MB/day
                },
                engagement: security::EngagementMetrics {
                    validation_participation: 0.9,
                    transaction_frequency: 100.0,
                    participation_time: 86400 * 7, // 7 days
                    community_contribution: 0.5,
                    governance_participation: 0.5,
                    staking_percentage: 0.01,
                    referrals: 0,
                    social_verification: 0.8,
                },
                ai_behavior: security::AIBehaviorMetrics {
                    anomaly_score: 0.1,
                    risk_assessment: 0.9,
                    fraud_probability: 0.05,
                    threat_level: 0.1,
                    pattern_consistency: 0.05,
                    sybil_probability: 0.05,
                    historical_reliability: 0.9,
                    identity_verification: 0.8,
                },
            };
            
            match security.evaluate_node(node_id, &metrics).await {
                Ok(score) => score.overall_score,
                Err(_) => 0.7,
            }
        } else {
            0.7
        };
        
        // Weighted combination
        let device_weight = 0.3;
        let security_weight = 0.3;
        let ai_security_weight = 0.4;
        
        (device_score.overall_score * device_weight) + 
        (security_score * security_weight) +
        (ai_security_score * ai_security_weight)
    }
}

/// Create a default AI Engine with manual configuration
pub fn create_default_ai_engine() -> AIEngine {
    let mut config = Config::default();
    
    // Set network configuration
    config.network.p2p_port = 7000;
    config.network.max_peers = 50;
    config.network.bootstrap_nodes = vec![
        "127.0.0.1:7001".to_string(),
        "127.0.0.1:7002".to_string(),
        "127.0.0.1:7003".to_string(),
    ];
    
    // Set API configuration
    config.api.port = 8080;
    config.api.host = "127.0.0.1".to_string();
    config.api.address = "127.0.0.1".to_string();
    config.api.enabled = true;
    config.api.cors_domains = vec!["*".to_string()];
    config.api.allow_origin = vec!["*".to_string()];
    config.api.max_request_body_size = 10 * 1024 * 1024; // 10MB
    config.api.max_connections = 100;
    config.api.enable_websocket = false;
    config.api.enable_graphql = false;
    
    // Set sharding configuration
    config.sharding.shard_count = 4;
    config.sharding.shard_id = 0;
    config.sharding.enabled = true;
    
    // Create and return the AI Engine
    AIEngine::new(config)
}

#[cfg(test)]
mod data_chunking_tests;

// Re-export key components
pub use data_chunking::ChunkingConfig; 