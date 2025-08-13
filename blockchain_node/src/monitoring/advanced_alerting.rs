use crate::consensus::consensus_manager::{ConsensusManager, ConsensusState};
use crate::monitoring::health_check::{ComponentHealth, HealthChecker};
use crate::monitoring::metrics_collector::MetricsCollector;
use crate::network::partition_healer::NetworkPartitionHealer;
use crate::storage::disaster_recovery::DisasterRecoveryManager;
use anyhow::{anyhow, Result};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::time::interval;

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    /// System health alerts
    SystemHealth,
    /// Consensus alerts
    Consensus,
    /// Storage alerts
    Storage,
    /// Network alerts
    Network,
    /// Security alerts
    Security,
    /// Performance alerts
    Performance,
    /// Custom alert
    Custom(String),
}

/// Alert rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Severity
    pub severity: AlertSeverity,
    /// Condition to check
    pub condition: AlertCondition,
    /// Cooldown period between alerts (seconds)
    pub cooldown_secs: u64,
    /// Enable rule
    pub enabled: bool,
    /// Escalation policy
    pub escalation_policy: Option<String>,
}

/// Alert conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Metric threshold
    MetricThreshold {
        metric: String,
        operator: ComparisonOperator,
        threshold: f64,
        duration_secs: u64,
    },
    /// Component health check
    ComponentHealth {
        component: String,
        expected_state: String,
    },
    /// Consensus state
    ConsensusState { expected_state: String },
    /// Storage integrity
    StorageIntegrity { corruption_detected: bool },
    /// Network partition
    NetworkPartition { max_partitions: usize },
    /// Custom condition
    Custom { expression: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Alert notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Rule that triggered the alert
    pub rule_id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Additional context
    pub context: HashMap<String, String>,
    /// Trigger time
    pub triggered_at: SystemTime,
    /// Alert status
    pub status: AlertStatus,
    /// Last notification time
    pub last_notified: Option<SystemTime>,
    /// Acknowledgment info
    pub acknowledged_by: Option<String>,
    /// Acknowledgment time
    pub acknowledged_at: Option<SystemTime>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
    Suppressed,
}

/// Escalation policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    /// Policy ID
    pub id: String,
    /// Policy name
    pub name: String,
    /// Escalation steps
    pub steps: Vec<EscalationStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationStep {
    /// Step delay in seconds
    pub delay_secs: u64,
    /// Notification targets
    pub targets: Vec<NotificationTarget>,
}

/// Notification targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationTarget {
    /// Webhook URL
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    /// Email address
    Email { address: String },
    /// Slack channel
    Slack {
        webhook_url: String,
        channel: String,
    },
    /// Discord webhook
    Discord { webhook_url: String },
    /// SMS (requires external service)
    SMS { phone_number: String },
    /// PagerDuty integration
    PagerDuty { integration_key: String },
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting system
    pub enabled: bool,
    /// Check interval in seconds
    pub check_interval_secs: u64,
    /// Default notification targets
    pub default_targets: Vec<NotificationTarget>,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Escalation policies
    pub escalation_policies: HashMap<String, EscalationPolicy>,
    /// Global alert suppression
    pub suppression_enabled: bool,
    /// Maximum alerts per minute
    pub max_alerts_per_minute: u32,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval_secs: 10,
            default_targets: vec![],
            rules: vec![
                // Critical system health alert
                AlertRule {
                    id: "system_health_critical".to_string(),
                    name: "System Health Critical".to_string(),
                    alert_type: AlertType::SystemHealth,
                    severity: AlertSeverity::Critical,
                    condition: AlertCondition::ComponentHealth {
                        component: "overall".to_string(),
                        expected_state: "healthy".to_string(),
                    },
                    cooldown_secs: 300, // 5 minutes
                    enabled: true,
                    escalation_policy: Some("critical".to_string()),
                },
                // Consensus failure alert
                AlertRule {
                    id: "consensus_failure".to_string(),
                    name: "Consensus Failure".to_string(),
                    alert_type: AlertType::Consensus,
                    severity: AlertSeverity::Emergency,
                    condition: AlertCondition::ConsensusState {
                        expected_state: "emergency".to_string(),
                    },
                    cooldown_secs: 60, // 1 minute
                    enabled: true,
                    escalation_policy: Some("emergency".to_string()),
                },
                // Storage corruption alert
                AlertRule {
                    id: "storage_corruption".to_string(),
                    name: "Storage Corruption Detected".to_string(),
                    alert_type: AlertType::Storage,
                    severity: AlertSeverity::Critical,
                    condition: AlertCondition::StorageIntegrity {
                        corruption_detected: true,
                    },
                    cooldown_secs: 300, // 5 minutes
                    enabled: true,
                    escalation_policy: Some("critical".to_string()),
                },
                // Network partition alert
                AlertRule {
                    id: "network_partition".to_string(),
                    name: "Network Partition Detected".to_string(),
                    alert_type: AlertType::Network,
                    severity: AlertSeverity::Warning,
                    condition: AlertCondition::NetworkPartition { max_partitions: 0 },
                    cooldown_secs: 180, // 3 minutes
                    enabled: true,
                    escalation_policy: Some("standard".to_string()),
                },
                // High CPU usage alert
                AlertRule {
                    id: "high_cpu".to_string(),
                    name: "High CPU Usage".to_string(),
                    alert_type: AlertType::Performance,
                    severity: AlertSeverity::Warning,
                    condition: AlertCondition::MetricThreshold {
                        metric: "cpu_usage_percent".to_string(),
                        operator: ComparisonOperator::GreaterThan,
                        threshold: 80.0,
                        duration_secs: 300, // 5 minutes
                    },
                    cooldown_secs: 600, // 10 minutes
                    enabled: true,
                    escalation_policy: None,
                },
            ],
            escalation_policies: {
                let mut policies = HashMap::new();

                // Standard escalation
                policies.insert(
                    "standard".to_string(),
                    EscalationPolicy {
                        id: "standard".to_string(),
                        name: "Standard Escalation".to_string(),
                        steps: vec![
                            EscalationStep {
                                delay_secs: 0,
                                targets: vec![], // Would be configured with actual targets
                            },
                            EscalationStep {
                                delay_secs: 300, // 5 minutes
                                targets: vec![], // Escalate to higher level
                            },
                        ],
                    },
                );

                // Critical escalation
                policies.insert(
                    "critical".to_string(),
                    EscalationPolicy {
                        id: "critical".to_string(),
                        name: "Critical Escalation".to_string(),
                        steps: vec![
                            EscalationStep {
                                delay_secs: 0,
                                targets: vec![], // Immediate notification
                            },
                            EscalationStep {
                                delay_secs: 120, // 2 minutes
                                targets: vec![], // Quick escalation
                            },
                        ],
                    },
                );

                // Emergency escalation
                policies.insert(
                    "emergency".to_string(),
                    EscalationPolicy {
                        id: "emergency".to_string(),
                        name: "Emergency Escalation".to_string(),
                        steps: vec![EscalationStep {
                            delay_secs: 0,
                            targets: vec![], // All hands on deck
                        }],
                    },
                );

                policies
            },
            suppression_enabled: false,
            max_alerts_per_minute: 10,
        }
    }
}

/// Advanced alerting system
pub struct AdvancedAlertingSystem {
    /// Configuration
    config: AlertingConfig,
    /// Health checker
    health_checker: Arc<HealthChecker>,
    /// Metrics collector
    metrics_collector: Arc<MetricsCollector>,
    /// Consensus manager
    consensus_manager: Arc<ConsensusManager>,
    /// Disaster recovery manager
    disaster_recovery: Arc<DisasterRecoveryManager>,
    /// Network partition healer
    partition_healer: Arc<NetworkPartitionHealer>,
    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, Alert>>>,
    /// Alert history
    alert_history: Arc<RwLock<VecDeque<Alert>>>,
    /// Rule cooldowns
    rule_cooldowns: Arc<RwLock<HashMap<String, Instant>>>,
    /// Alert rate limiter
    alert_rate_limiter: Arc<RwLock<VecDeque<Instant>>>,
    /// Event broadcaster
    event_sender: broadcast::Sender<Alert>,
    /// Monitoring task handle
    monitor_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl AdvancedAlertingSystem {
    /// Get metric value from SystemMetrics
    fn get_metric_value(metrics: &crate::monitoring::SystemMetrics, metric: &str) -> Option<f64> {
        match metric {
            "cpu_usage" => Some(metrics.cpu_usage),
            "memory_usage" => Some(metrics.memory_usage as f64),
            "disk_usage" => Some(metrics.disk_usage as f64),
            "active_connections" => Some(metrics.active_connections as f64),
            "transaction_throughput" => Some(metrics.transaction_throughput),
            "block_production_rate" => Some(metrics.block_production_rate),
            _ => None,
        }
    }

    /// Create new alerting system
    pub fn new(
        config: AlertingConfig,
        health_checker: Arc<HealthChecker>,
        metrics_collector: Arc<MetricsCollector>,
        consensus_manager: Arc<ConsensusManager>,
        disaster_recovery: Arc<DisasterRecoveryManager>,
        partition_healer: Arc<NetworkPartitionHealer>,
    ) -> Self {
        let (event_sender, _) = broadcast::channel(1000);

        Self {
            config,
            health_checker,
            metrics_collector,
            consensus_manager,
            disaster_recovery,
            partition_healer,
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            rule_cooldowns: Arc::new(RwLock::new(HashMap::new())),
            alert_rate_limiter: Arc::new(RwLock::new(VecDeque::new())),
            event_sender,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Start alerting system
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Alerting system is disabled");
            return Ok(());
        }

        info!("Starting advanced alerting system");

        // Start monitoring task
        self.start_monitoring().await?;

        info!("Advanced alerting system started");
        Ok(())
    }

    /// Start monitoring task
    async fn start_monitoring(&self) -> Result<()> {
        let config = self.config.clone();
        let health_checker = self.health_checker.clone();
        let metrics_collector = self.metrics_collector.clone();
        let consensus_manager = self.consensus_manager.clone();
        let disaster_recovery = self.disaster_recovery.clone();
        let partition_healer = self.partition_healer.clone();
        let active_alerts = self.active_alerts.clone();
        let alert_history = self.alert_history.clone();
        let rule_cooldowns = self.rule_cooldowns.clone();
        let alert_rate_limiter = self.alert_rate_limiter.clone();
        let event_sender = self.event_sender.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.check_interval_secs));

            loop {
                interval.tick().await;

                // Check all alert rules
                for rule in &config.rules {
                    if !rule.enabled {
                        continue;
                    }

                    // Check if rule is in cooldown
                    if Self::is_rule_in_cooldown(&rule_cooldowns, &rule.id, rule.cooldown_secs)
                        .await
                    {
                        continue;
                    }

                    // Evaluate rule condition
                    let triggered = match Self::evaluate_condition(
                        &rule.condition,
                        &health_checker,
                        &metrics_collector,
                        &consensus_manager,
                        &disaster_recovery,
                        &partition_healer,
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(e) => {
                            error!("Failed to evaluate rule {}: {}", rule.id, e);
                            continue;
                        }
                    };

                    if triggered {
                        // Check rate limiting
                        if Self::is_rate_limited(&alert_rate_limiter, config.max_alerts_per_minute)
                            .await
                        {
                            warn!(
                                "Alert rate limit exceeded, suppressing alert for rule: {}",
                                rule.id
                            );
                            continue;
                        }

                        // Create and fire alert
                        let alert = Self::create_alert(rule).await;

                        // Add to active alerts
                        active_alerts
                            .write()
                            .await
                            .insert(alert.id.clone(), alert.clone());

                        // Add to history
                        let mut history = alert_history.write().await;
                        history.push_back(alert.clone());

                        // Keep history size reasonable
                        while history.len() > 1000 {
                            history.pop_front();
                        }

                        // Set cooldown
                        rule_cooldowns
                            .write()
                            .await
                            .insert(rule.id.clone(), Instant::now());

                        // Send notification
                        if let Err(e) = Self::send_notification(&alert, &config).await {
                            error!("Failed to send notification for alert {}: {}", alert.id, e);
                        }

                        // Broadcast event
                        let _ = event_sender.send(alert);

                        info!("Alert triggered: {} - {}", rule.name, rule.id);
                    }
                }

                // Clean up resolved alerts
                Self::cleanup_resolved_alerts(&active_alerts).await;
            }
        });

        *self.monitor_handle.lock().await = Some(handle);
        Ok(())
    }

    /// Check if rule is in cooldown
    async fn is_rule_in_cooldown(
        rule_cooldowns: &Arc<RwLock<HashMap<String, Instant>>>,
        rule_id: &str,
        cooldown_secs: u64,
    ) -> bool {
        if let Some(last_triggered) = rule_cooldowns.read().await.get(rule_id) {
            last_triggered.elapsed().as_secs() < cooldown_secs
        } else {
            false
        }
    }

    /// Check if rate limited
    async fn is_rate_limited(
        alert_rate_limiter: &Arc<RwLock<VecDeque<Instant>>>,
        max_per_minute: u32,
    ) -> bool {
        let now = Instant::now();
        let mut alerts = alert_rate_limiter.write().await;

        // Remove alerts older than 1 minute
        while let Some(&front) = alerts.front() {
            if now.duration_since(front).as_secs() >= 60 {
                alerts.pop_front();
            } else {
                break;
            }
        }

        // Check if we've exceeded the limit
        if alerts.len() >= max_per_minute as usize {
            return true;
        }

        // Add current alert
        alerts.push_back(now);
        false
    }

    /// Evaluate alert condition
    async fn evaluate_condition(
        condition: &AlertCondition,
        health_checker: &Arc<HealthChecker>,
        metrics_collector: &Arc<MetricsCollector>,
        consensus_manager: &Arc<ConsensusManager>,
        disaster_recovery: &Arc<DisasterRecoveryManager>,
        partition_healer: &Arc<NetworkPartitionHealer>,
    ) -> Result<bool> {
        match condition {
            AlertCondition::MetricThreshold {
                metric,
                operator,
                threshold,
                duration_secs: _,
            } => {
                let metrics = metrics_collector.get_current_metrics().await;
                if let Ok(metrics_data) = metrics {
                    if let Some(value) = Self::get_metric_value(&metrics_data, metric) {
                        Ok(Self::compare_value(value, operator, *threshold))
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            AlertCondition::ComponentHealth {
                component,
                expected_state,
            } => {
                let health_results = health_checker.check_all_components().await;
                if let Ok(health_data) = health_results {
                    if let Some(health) = health_data.get(component) {
                        let is_healthy = health.is_healthy;
                        Ok(expected_state == "healthy" && !is_healthy)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            AlertCondition::ConsensusState { expected_state } => {
                let state = consensus_manager.get_state().await;
                let state_str = format!("{:?}", state).to_lowercase();
                Ok(state_str == expected_state.to_lowercase())
            }
            AlertCondition::StorageIntegrity {
                corruption_detected,
            } => {
                let integrity_ok = disaster_recovery.check_storage_integrity().await?;
                Ok(*corruption_detected && !integrity_ok)
            }
            AlertCondition::NetworkPartition { max_partitions } => {
                let partitions = partition_healer.get_partitions().await;
                Ok(partitions.len() > *max_partitions)
            }
            AlertCondition::Custom { expression: _ } => {
                // Custom conditions would require a rule engine
                // For now, return false
                Ok(false)
            }
        }
    }

    /// Compare metric value with threshold
    fn compare_value(value: f64, operator: &ComparisonOperator, threshold: f64) -> bool {
        match operator {
            ComparisonOperator::GreaterThan => value > threshold,
            ComparisonOperator::LessThan => value < threshold,
            ComparisonOperator::Equals => (value - threshold).abs() < f64::EPSILON,
            ComparisonOperator::NotEquals => (value - threshold).abs() >= f64::EPSILON,
            ComparisonOperator::GreaterThanOrEqual => value >= threshold,
            ComparisonOperator::LessThanOrEqual => value <= threshold,
        }
    }

    /// Create alert from rule
    async fn create_alert(rule: &AlertRule) -> Alert {
        Alert {
            id: uuid::Uuid::new_v4().to_string(),
            rule_id: rule.id.clone(),
            alert_type: rule.alert_type.clone(),
            severity: rule.severity.clone(),
            message: format!("Alert: {}", rule.name),
            context: HashMap::new(),
            triggered_at: SystemTime::now(),
            status: AlertStatus::Active,
            last_notified: None,
            acknowledged_by: None,
            acknowledged_at: None,
        }
    }

    /// Send notification for alert
    async fn send_notification(alert: &Alert, config: &AlertingConfig) -> Result<()> {
        // Send to default targets first
        for target in &config.default_targets {
            if let Err(e) = Self::send_to_target(alert, target).await {
                error!("Failed to send to default target: {}", e);
            }
        }

        // Handle escalation policy if configured
        // This would be implemented based on the escalation policy logic

        Ok(())
    }

    /// Send alert to specific target
    async fn send_to_target(alert: &Alert, target: &NotificationTarget) -> Result<()> {
        match target {
            NotificationTarget::Webhook { url, headers } => {
                Self::send_webhook(alert, url, headers).await
            }
            NotificationTarget::Email { address } => Self::send_email(alert, address).await,
            NotificationTarget::Slack {
                webhook_url,
                channel,
            } => Self::send_slack(alert, webhook_url, channel).await,
            NotificationTarget::Discord { webhook_url } => {
                Self::send_discord(alert, webhook_url).await
            }
            NotificationTarget::SMS { phone_number } => Self::send_sms(alert, phone_number).await,
            NotificationTarget::PagerDuty { integration_key } => {
                Self::send_pagerduty(alert, integration_key).await
            }
        }
    }

    /// Send webhook notification
    async fn send_webhook(
        alert: &Alert,
        url: &str,
        headers: &HashMap<String, String>,
    ) -> Result<()> {
        let client = reqwest::Client::new();
        let payload = serde_json::to_string(alert)?;

        let mut request = client.post(url).body(payload);

        for (key, value) in headers {
            request = request.header(key, value);
        }

        request = request.header("Content-Type", "application/json");

        let response = request.send().await?;

        if response.status().is_success() {
            info!("Webhook notification sent successfully to {}", url);
        } else {
            error!("Webhook notification failed: {}", response.status());
        }

        Ok(())
    }

    /// Send email notification
    async fn send_email(alert: &Alert, address: &str) -> Result<()> {
        // Email sending would require SMTP configuration
        // For now, just log
        info!("Would send email to {}: {}", address, alert.message);
        Ok(())
    }

    /// Send Slack notification
    async fn send_slack(alert: &Alert, webhook_url: &str, channel: &str) -> Result<()> {
        let client = reqwest::Client::new();
        let color = match alert.severity {
            AlertSeverity::Info => "good",
            AlertSeverity::Warning => "warning",
            AlertSeverity::Critical => "danger",
            AlertSeverity::Emergency => "danger",
        };

        let payload = serde_json::json!({
            "channel": channel,
            "attachments": [{
                "color": color,
                "title": format!("{:?} Alert", alert.severity),
                "text": alert.message,
                "fields": [
                    {
                        "title": "Alert ID",
                        "value": alert.id,
                        "short": true
                    },
                    {
                        "title": "Type",
                        "value": format!("{:?}", alert.alert_type),
                        "short": true
                    }
                ],
                "ts": alert.triggered_at.duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or(Duration::from_secs(0)).as_secs()
            }]
        });

        let response = client.post(webhook_url).json(&payload).send().await?;

        if response.status().is_success() {
            info!("Slack notification sent successfully");
        } else {
            error!("Slack notification failed: {}", response.status());
        }

        Ok(())
    }

    /// Send Discord notification
    async fn send_discord(alert: &Alert, webhook_url: &str) -> Result<()> {
        let client = reqwest::Client::new();
        let color = match alert.severity {
            AlertSeverity::Info => 3447003,       // Blue
            AlertSeverity::Warning => 16776960,   // Yellow
            AlertSeverity::Critical => 15158332,  // Red
            AlertSeverity::Emergency => 10038562, // Dark red
        };

        let payload = serde_json::json!({
            "embeds": [{
                "title": format!("{:?} Alert", alert.severity),
                "description": alert.message,
                "color": color,
                "fields": [
                    {
                        "name": "Alert ID",
                        "value": alert.id,
                        "inline": true
                    },
                    {
                        "name": "Type",
                        "value": format!("{:?}", alert.alert_type),
                        "inline": true
                    }
                ],
                "timestamp": chrono::Utc::now().to_rfc3339()
            }]
        });

        let response = client.post(webhook_url).json(&payload).send().await?;

        if response.status().is_success() {
            info!("Discord notification sent successfully");
        } else {
            error!("Discord notification failed: {}", response.status());
        }

        Ok(())
    }

    /// Send SMS notification
    async fn send_sms(alert: &Alert, phone_number: &str) -> Result<()> {
        // SMS sending would require integration with SMS service (Twilio, etc.)
        // For now, just log
        info!("Would send SMS to {}: {}", phone_number, alert.message);
        Ok(())
    }

    /// Send PagerDuty notification
    async fn send_pagerduty(alert: &Alert, integration_key: &str) -> Result<()> {
        let client = reqwest::Client::new();
        let event_action = match alert.status {
            AlertStatus::Active => "trigger",
            AlertStatus::Resolved => "resolve",
            _ => "trigger",
        };

        let payload = serde_json::json!({
            "routing_key": integration_key,
            "event_action": event_action,
            "dedup_key": alert.rule_id,
            "payload": {
                "summary": alert.message,
                "severity": format!("{:?}", alert.severity).to_lowercase(),
                "source": "arthachain",
                "component": format!("{:?}", alert.alert_type),
                "custom_details": alert.context
            }
        });

        let response = client
            .post("https://events.pagerduty.com/v2/enqueue")
            .json(&payload)
            .send()
            .await?;

        if response.status().is_success() {
            info!("PagerDuty notification sent successfully");
        } else {
            error!("PagerDuty notification failed: {}", response.status());
        }

        Ok(())
    }

    /// Clean up resolved alerts
    async fn cleanup_resolved_alerts(active_alerts: &Arc<RwLock<HashMap<String, Alert>>>) {
        // This would check if alert conditions are no longer true
        // and mark them as resolved
        // For now, we don't implement automatic resolution
    }

    /// Acknowledge alert
    pub async fn acknowledge_alert(&self, alert_id: &str, user: &str) -> Result<()> {
        let mut alerts = self.active_alerts.write().await;
        if let Some(alert) = alerts.get_mut(alert_id) {
            alert.status = AlertStatus::Acknowledged;
            alert.acknowledged_by = Some(user.to_string());
            alert.acknowledged_at = Some(SystemTime::now());
            info!("Alert {} acknowledged by {}", alert_id, user);
            Ok(())
        } else {
            Err(anyhow!("Alert not found: {}", alert_id))
        }
    }

    /// Resolve alert
    pub async fn resolve_alert(&self, alert_id: &str) -> Result<()> {
        let mut alerts = self.active_alerts.write().await;
        if let Some(alert) = alerts.get_mut(alert_id) {
            alert.status = AlertStatus::Resolved;
            info!("Alert {} resolved", alert_id);
            Ok(())
        } else {
            Err(anyhow!("Alert not found: {}", alert_id))
        }
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.read().await.values().cloned().collect()
    }

    /// Get alert history
    pub async fn get_alert_history(&self, limit: Option<usize>) -> Vec<Alert> {
        let history = self.alert_history.read().await;
        let limit = limit.unwrap_or(100);
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Subscribe to alerts
    pub fn subscribe_alerts(&self) -> broadcast::Receiver<Alert> {
        self.event_sender.subscribe()
    }

    /// Stop alerting system
    pub async fn stop(&self) {
        if let Some(handle) = self.monitor_handle.lock().await.take() {
            handle.abort();
        }
        info!("Advanced alerting system stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_operators() {
        assert!(AdvancedAlertingSystem::compare_value(
            10.0,
            &ComparisonOperator::GreaterThan,
            5.0
        ));
        assert!(!AdvancedAlertingSystem::compare_value(
            5.0,
            &ComparisonOperator::GreaterThan,
            10.0
        ));
        assert!(AdvancedAlertingSystem::compare_value(
            5.0,
            &ComparisonOperator::LessThan,
            10.0
        ));
        assert!(AdvancedAlertingSystem::compare_value(
            5.0,
            &ComparisonOperator::Equals,
            5.0
        ));
    }

    #[tokio::test]
    async fn test_alert_creation() {
        let rule = AlertRule {
            id: "test_rule".to_string(),
            name: "Test Alert".to_string(),
            alert_type: AlertType::SystemHealth,
            severity: AlertSeverity::Warning,
            condition: AlertCondition::MetricThreshold {
                metric: "cpu".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 80.0,
                duration_secs: 300,
            },
            cooldown_secs: 300,
            enabled: true,
            escalation_policy: None,
        };

        let alert = AdvancedAlertingSystem::create_alert(&rule).await;
        assert_eq!(alert.rule_id, "test_rule");
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.status, AlertStatus::Active);
    }
}
