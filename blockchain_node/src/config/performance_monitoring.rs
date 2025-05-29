use crate::ai_engine::performance_monitor::{
    AiOptimizationConfig, LoggingConfig, MonitoringConfig, QuantumMonitoringConfig,
};
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::path::PathBuf;

/// Configuration for the performance monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    /// Path to AI models directory
    pub models_path: PathBuf,
    /// Visualization settings
    pub visualization: VisualizationConfig,
    /// Alert settings
    pub alerts: AlertConfig,
}

/// Configuration for performance data visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Enable dashboard
    pub dashboard_enabled: bool,
    /// Dashboard port
    pub dashboard_port: u16,
    /// Prometheus integration
    pub prometheus_enabled: bool,
    /// Prometheus endpoint
    pub prometheus_endpoint: String,
    /// Grafana configuration
    pub grafana: Option<GrafanaConfig>,
    /// Quantum visualization
    pub quantum_visualization: bool,
}

/// Grafana configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrafanaConfig {
    /// Grafana URL
    pub url: String,
    /// API key
    pub api_key: String,
    /// Dashboard ID
    pub dashboard_id: String,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Notification email
    pub email: Option<String>,
    /// Webhook URL
    pub webhook_url: Option<String>,
    /// Minimum severity for notifications (1-5)
    pub min_severity: u8,
    /// Alert for quantum vulnerabilities
    pub quantum_alerts: bool,
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring: MonitoringConfig {
                enabled: true,
                sampling_interval_ms: 5000,
                retention_period_days: 30,
                ai_optimization: AiOptimizationConfig {
                    enabled: true,
                    model_path: "models/performance_predictor.onnx".to_string(),
                    prediction_interval_ms: 60000,
                    confidence_threshold: 0.7,
                },
                quantum_monitoring: QuantumMonitoringConfig {
                    enabled: true,
                    simulation_metrics: true,
                    security_level_monitoring: true,
                },
                logging: LoggingConfig {
                    level: "info".to_string(),
                    output_dir: "/var/log/blockchain/performance".to_string(),
                    stdout: true,
                },
            },
            models_path: PathBuf::from("models"),
            visualization: VisualizationConfig {
                dashboard_enabled: true,
                dashboard_port: 8090,
                prometheus_enabled: true,
                prometheus_endpoint: "http://localhost:9090".to_string(),
                grafana: Some(GrafanaConfig {
                    url: "http://localhost:3000".to_string(),
                    api_key: "".to_string(),
                    dashboard_id: "blockchain-performance".to_string(),
                }),
                quantum_visualization: true,
            },
            alerts: AlertConfig {
                enabled: true,
                email: None,
                webhook_url: None,
                min_severity: 3,
                quantum_alerts: true,
            },
        }
    }
}

/// Create performance monitoring configuration from file
pub fn load_performance_config(
    config_path: &str,
) -> Result<PerformanceMonitoringConfig, std::io::Error> {
    let config_str = std::fs::read_to_string(config_path)?;
    let config: PerformanceMonitoringConfig = serde_yaml::from_str(&config_str)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(config)
}

/// Save performance monitoring configuration to file
pub fn save_performance_config(
    config: &PerformanceMonitoringConfig,
    config_path: &str,
) -> Result<(), std::io::Error> {
    let config_str = serde_yaml::to_string(config)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(config_path, config_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_serialization() {
        let config = PerformanceMonitoringConfig::default();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();

        save_performance_config(&config, path).unwrap();
        let loaded_config = load_performance_config(path).unwrap();

        assert_eq!(config.enabled, loaded_config.enabled);
        assert_eq!(
            config.monitoring.sampling_interval_ms,
            loaded_config.monitoring.sampling_interval_ms
        );
        assert_eq!(
            config.visualization.dashboard_port,
            loaded_config.visualization.dashboard_port
        );
    }
}
