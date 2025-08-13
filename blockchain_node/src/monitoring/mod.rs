pub mod advanced_alerting;
pub mod alerting;
pub mod health_check;
pub mod metrics_collector;

use anyhow::Result;
use log::{error, info, warn};
use prometheus::{Counter, Encoder, Gauge, Histogram, HistogramOpts, Registry, TextEncoder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Once};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

// Import ComponentHealth from health_check module
use health_check::ComponentHealth;

/// Global metrics registry
lazy_static::lazy_static! {
    pub static ref METRICS_REGISTRY: Registry = Registry::new();

    // System metrics
    pub static ref NODE_UPTIME: Gauge = Gauge::new("node_uptime_seconds", "Node uptime in seconds").unwrap();
    pub static ref CPU_USAGE: Gauge = Gauge::new("cpu_usage_percent", "CPU usage percentage").unwrap();
    pub static ref MEMORY_USAGE: Gauge = Gauge::new("memory_usage_bytes", "Memory usage in bytes").unwrap();
    pub static ref DISK_USAGE: Gauge = Gauge::new("disk_usage_bytes", "Disk usage in bytes").unwrap();

    // Blockchain metrics
    pub static ref BLOCK_HEIGHT: Gauge = Gauge::new("blockchain_height", "Current blockchain height").unwrap();
    pub static ref BLOCK_PROCESSING_TIME: Histogram = Histogram::with_opts(
        HistogramOpts::new("block_processing_time_seconds", "Block processing time in seconds")
    ).unwrap();
    pub static ref TRANSACTION_COUNT: Counter = Counter::new("transaction_total", "Total transactions processed").unwrap();
    pub static ref TRANSACTION_PROCESSING_TIME: Histogram = Histogram::with_opts(
        HistogramOpts::new("transaction_processing_time_seconds", "Transaction processing time in seconds")
    ).unwrap();

    // Network metrics
    pub static ref PEER_COUNT: Gauge = Gauge::new("network_peers", "Number of connected peers").unwrap();
    pub static ref NETWORK_BYTES_IN: Counter = Counter::new("network_bytes_in_total", "Total bytes received").unwrap();
    pub static ref NETWORK_BYTES_OUT: Counter = Counter::new("network_bytes_out_total", "Total bytes sent").unwrap();
    pub static ref NETWORK_LATENCY: Histogram = Histogram::with_opts(
        HistogramOpts::new("network_latency_seconds", "Network latency in seconds")
    ).unwrap();

    // Storage metrics
    pub static ref STORAGE_READS: Counter = Counter::new("storage_reads_total", "Total storage read operations").unwrap();
    pub static ref STORAGE_WRITES: Counter = Counter::new("storage_writes_total", "Total storage write operations").unwrap();
    pub static ref STORAGE_LATENCY: Histogram = Histogram::with_opts(
        HistogramOpts::new("storage_latency_seconds", "Storage operation latency in seconds")
    ).unwrap();
    pub static ref STORAGE_SIZE: Gauge = Gauge::new("storage_size_bytes", "Total storage size in bytes").unwrap();

    // Consensus metrics
    pub static ref CONSENSUS_ROUNDS: Counter = Counter::new("consensus_rounds_total", "Total consensus rounds").unwrap();
    pub static ref CONSENSUS_FAILURES: Counter = Counter::new("consensus_failures_total", "Total consensus failures").unwrap();
    pub static ref LEADER_ELECTIONS: Counter = Counter::new("leader_elections_total", "Total leader elections").unwrap();
    pub static ref BYZANTINE_FAULTS: Counter = Counter::new("byzantine_faults_total", "Total Byzantine faults detected").unwrap();

    // AI metrics
    pub static ref AI_MODEL_PREDICTIONS: Counter = Counter::new("ai_model_predictions_total", "Total AI model predictions").unwrap();
    pub static ref AI_MODEL_ACCURACY: Gauge = Gauge::new("ai_model_accuracy", "AI model accuracy score").unwrap();
    pub static ref AI_PROCESSING_TIME: Histogram = Histogram::with_opts(
        HistogramOpts::new("ai_processing_time_seconds", "AI processing time in seconds")
    ).unwrap();
}

// Use a global Once to ensure metrics are only initialized once
static INIT_METRICS: Once = Once::new();

/// Initialize all metrics
pub fn init_metrics() -> Result<()> {
    INIT_METRICS.call_once(|| {
        // Register system metrics
        let _ = METRICS_REGISTRY.register(Box::new(NODE_UPTIME.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(CPU_USAGE.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(MEMORY_USAGE.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(DISK_USAGE.clone()));

        // Register blockchain metrics
        let _ = METRICS_REGISTRY.register(Box::new(BLOCK_HEIGHT.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(BLOCK_PROCESSING_TIME.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(TRANSACTION_COUNT.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(TRANSACTION_PROCESSING_TIME.clone()));

        // Register network metrics
        let _ = METRICS_REGISTRY.register(Box::new(PEER_COUNT.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(NETWORK_BYTES_IN.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(NETWORK_BYTES_OUT.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(NETWORK_LATENCY.clone()));

        // Register storage metrics
        let _ = METRICS_REGISTRY.register(Box::new(STORAGE_READS.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(STORAGE_WRITES.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(STORAGE_LATENCY.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(STORAGE_SIZE.clone()));

        // Register consensus metrics
        let _ = METRICS_REGISTRY.register(Box::new(CONSENSUS_ROUNDS.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(CONSENSUS_FAILURES.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(LEADER_ELECTIONS.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(BYZANTINE_FAULTS.clone()));

        // Register AI metrics
        let _ = METRICS_REGISTRY.register(Box::new(AI_MODEL_PREDICTIONS.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(AI_MODEL_ACCURACY.clone()));
        let _ = METRICS_REGISTRY.register(Box::new(AI_PROCESSING_TIME.clone()));
    });

    info!("Metrics initialized successfully");
    Ok(())
}

/// Export metrics in Prometheus format
pub fn export_metrics() -> Result<String> {
    let encoder = TextEncoder::new();
    let metric_families = METRICS_REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    Ok(String::from_utf8(buffer)?)
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub metrics_enabled: bool,
    /// Metrics collection interval
    pub metrics_interval_secs: u64,
    /// Enable health checks
    pub health_checks_enabled: bool,
    /// Health check interval
    pub health_check_interval_secs: u64,
    /// Enable alerting
    pub alerting_enabled: bool,
    /// Alert check interval
    pub alert_check_interval_secs: u64,
    /// Prometheus endpoint
    pub prometheus_endpoint: String,
    /// Grafana endpoint
    pub grafana_endpoint: Option<String>,
    /// Loki endpoint for logs
    pub loki_endpoint: Option<String>,
    /// Alert webhook URL
    pub alert_webhook_url: Option<String>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_enabled: true,
            metrics_interval_secs: 10,
            health_checks_enabled: true,
            health_check_interval_secs: 30,
            alerting_enabled: true,
            alert_check_interval_secs: 60,
            prometheus_endpoint: "0.0.0.0:9090".to_string(),
            grafana_endpoint: Some("http://localhost:3000".to_string()),
            loki_endpoint: Some("http://localhost:3100".to_string()),
            alert_webhook_url: None,
        }
    }
}

/// System health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall system health
    pub overall_health: SystemHealth,
    /// Component health statuses
    pub components: HashMap<String, ComponentHealth>,
    /// System metrics
    pub metrics: SystemMetrics,
    /// Active alerts
    pub active_alerts: Vec<Alert>,
    /// Report timestamp
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemHealth {
    Healthy,
    Degraded,
    Critical,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Disk usage in bytes
    pub disk_usage: u64,
    /// Network bandwidth usage
    pub network_bandwidth: u64,
    /// Active connections
    pub active_connections: usize,
    /// Transaction throughput (TPS)
    pub transaction_throughput: f64,
    /// Block production rate
    pub block_production_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert message
    pub message: String,
    /// Component affected
    pub component: String,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Additional context
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    SystemResource,
    NetworkIssue,
    StorageIssue,
    ConsensusFailure,
    SecurityThreat,
    PerformanceDegradation,
    ComponentFailure,
}

/// Monitoring service
pub struct MonitoringService {
    /// Configuration
    config: MonitoringConfig,
    /// Health checker
    health_checker: Arc<health_check::HealthChecker>,
    /// Metrics collector
    metrics_collector: Arc<metrics_collector::MetricsCollector>,
    /// Alert manager
    alert_manager: Arc<alerting::AlertManager>,
    /// Start time for uptime calculation
    start_time: Instant,
}

impl MonitoringService {
    /// Create new monitoring service
    pub fn new(config: MonitoringConfig) -> Result<Self> {
        // Initialize metrics
        init_metrics()?;

        let service = Self {
            config: config.clone(),
            health_checker: Arc::new(health_check::HealthChecker::new()),
            metrics_collector: Arc::new(metrics_collector::MetricsCollector::new()),
            alert_manager: Arc::new(alerting::AlertManager::new(config.clone())),
            start_time: Instant::now(),
        };

        Ok(service)
    }

    /// Start monitoring service
    pub async fn start(&self) -> Result<()> {
        // Start health checker
        if self.config.health_checks_enabled {
            self.health_checker
                .start(self.config.health_check_interval_secs)
                .await?;
        }

        // Start metrics collector
        if self.config.metrics_enabled {
            self.metrics_collector
                .start(self.config.metrics_interval_secs)
                .await?;
        }

        // Start alert manager
        if self.config.alerting_enabled {
            self.alert_manager
                .start(self.config.alert_check_interval_secs)
                .await?;
        }

        // Update uptime metric
        let uptime_updater = self.start_time;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                NODE_UPTIME.set(uptime_updater.elapsed().as_secs_f64());
            }
        });

        info!("Monitoring service started");
        Ok(())
    }

    /// Get health report
    pub async fn get_health_report(&self) -> Result<HealthReport> {
        let components = self.health_checker.check_all_components().await?;
        let metrics = self.metrics_collector.get_current_metrics().await?;
        let active_alerts = self.alert_manager.get_active_alerts().await;

        // Determine overall health
        let overall_health = Self::calculate_overall_health(&components);

        Ok(HealthReport {
            overall_health,
            components,
            metrics,
            active_alerts,
            timestamp: SystemTime::now(),
        })
    }

    /// Calculate overall system health
    fn calculate_overall_health(components: &HashMap<String, ComponentHealth>) -> SystemHealth {
        let unhealthy_count = components.values().filter(|c| !c.is_healthy).count();
        let total_count = components.len();

        if unhealthy_count == 0 {
            SystemHealth::Healthy
        } else if unhealthy_count < total_count / 4 {
            SystemHealth::Degraded
        } else if unhealthy_count < total_count / 2 {
            SystemHealth::Critical
        } else {
            SystemHealth::Failed
        }
    }

    /// Export Prometheus metrics
    pub fn export_prometheus_metrics(&self) -> Result<String> {
        export_metrics()
    }

    /// Register component for health checking
    pub async fn register_component(
        &self,
        name: String,
        checker: Box<dyn health_check::ComponentChecker>,
    ) -> Result<()> {
        self.health_checker.register_component(name, checker).await
    }

    /// Add alert rule
    pub async fn add_alert_rule(&self, rule: alerting::AlertRule) -> Result<()> {
        self.alert_manager.add_rule(rule).await
    }

    /// Stop monitoring service
    pub async fn stop(&self) {
        self.health_checker.stop().await;
        self.metrics_collector.stop().await;
        self.alert_manager.stop().await;
        info!("Monitoring service stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_initialization() {
        init_metrics().unwrap();

        // Test metric updates
        BLOCK_HEIGHT.set(100.0);
        PEER_COUNT.set(10.0);
        TRANSACTION_COUNT.inc();

        // Export metrics
        let metrics = export_metrics().unwrap();
        assert!(metrics.contains("blockchain_height 100"));
        assert!(metrics.contains("network_peers 10"));
    }

    #[tokio::test]
    async fn test_monitoring_service() {
        let config = MonitoringConfig::default();
        let service = MonitoringService::new(config).unwrap();

        // Start service
        service.start().await.unwrap();

        // Get health report
        let report = service.get_health_report().await.unwrap();
        assert!(matches!(report.overall_health, SystemHealth::Healthy));

        // Stop service
        service.stop().await;
    }
}

// Re-export commonly used types
pub use health_check::HealthChecker;
pub use metrics_collector::MetricsCollector;
