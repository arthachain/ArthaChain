use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for model failover behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFailoverConfig {
    /// Memory usage threshold in bytes before failover
    pub memory_threshold: u64,
    /// CPU usage threshold (0.0-1.0) before failover
    pub cpu_threshold: f32,
    /// Disk usage threshold in bytes before failover
    pub disk_threshold: u64,
    /// Whether to enable automatic failover
    pub auto_failover: bool,
    /// Minimum time between failovers in seconds
    pub min_failover_interval: u64,
    /// Number of retry attempts before permanent failover
    pub retry_attempts: u32,
    /// Duration to wait between retry attempts
    #[serde(with = "serde_duration")]
    pub backoff_duration: Duration,
    /// Name of the fallback model to use
    pub fallback_model: String,
}

impl Default for ModelFailoverConfig {
    fn default() -> Self {
        Self {
            memory_threshold: 1024 * 1024 * 1024 * 8, // 8GB
            cpu_threshold: 0.8,                       // 80% CPU
            disk_threshold: 1024 * 1024 * 1024 * 50,  // 50GB
            auto_failover: true,
            min_failover_interval: 300,               // 5 minutes
            retry_attempts: 3,
            backoff_duration: Duration::from_secs(5),
            fallback_model: "fallback".to_string(),
        }
    }
}

mod serde_duration {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
} 