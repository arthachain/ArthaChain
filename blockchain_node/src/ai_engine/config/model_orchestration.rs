use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSchedulerConfig {
    /// Device health check interval
    pub device_health_interval: Duration,
    /// User identity update interval
    pub identity_update_interval: Duration,
    /// Data chunking refresh interval
    pub chunking_refresh_interval: Duration,
    /// Security model update interval
    pub security_update_interval: Duration,
    /// Fraud detection training interval
    pub fraud_detection_training_interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFailoverSettings {
    /// Whether to enable rule-based fallback
    pub enable_fallback: bool,
    /// Memory threshold for failover (percentage)
    pub memory_threshold: f32,
    /// CPU load threshold for failover (0-100)
    pub cpu_threshold: f32,
    /// Disk usage threshold for failover (percentage)
    pub disk_threshold: f32,
    /// Maximum inference time before failover (ms)
    pub max_inference_time: u64,
}

#[derive(Debug, Clone)]
pub struct ModelOrchestrationConfig {
    pub enabled_components: Vec<String>,
    pub scheduler: ModelSchedulerConfig,
    pub failover: ModelFailoverSettings,
}

impl Default for ModelSchedulerConfig {
    fn default() -> Self {
        Self {
            device_health_interval: Duration::from_secs(60),
            identity_update_interval: Duration::from_secs(300),
            chunking_refresh_interval: Duration::from_secs(600),
            security_update_interval: Duration::from_secs(120),
            fraud_detection_training_interval: Duration::from_secs(3600),
        }
    }
}

impl Default for ModelOrchestrationConfig {
    fn default() -> Self {
        Self {
            enabled_components: vec![
                "device_health".to_string(),
                "user_identity".to_string(),
                "data_chunking".to_string(),
                "security".to_string(),
                "fraud_detection".to_string(),
            ],
            scheduler: ModelSchedulerConfig::default(),
            failover: ModelFailoverSettings::default(),
        }
    }
}

impl Default for ModelFailoverSettings {
    fn default() -> Self {
        Self {
            enable_fallback: true,
            memory_threshold: 80.0,   // 80%
            cpu_threshold: 80.0,      // 80%
            disk_threshold: 80.0,     // 80%
            max_inference_time: 1000, // 1 second
        }
    }
}
