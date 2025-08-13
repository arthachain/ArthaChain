use super::{Alert, AlertSeverity, AlertType, MonitoringConfig};
use anyhow::Result;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tokio::time::interval;

/// Alert rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Severity level
    pub severity: AlertSeverity,
    /// Metric to monitor
    pub metric: String,
    /// Condition
    pub condition: AlertCondition,
    /// Threshold value
    pub threshold: f64,
    /// Duration before triggering (seconds)
    pub duration_secs: u64,
    /// Cooldown period (seconds)
    pub cooldown_secs: u64,
    /// Alert message template
    pub message_template: String,
    /// Is rule enabled
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Alert state tracking
#[derive(Debug, Clone)]
struct AlertState {
    /// Rule ID
    rule_id: String,
    /// First trigger time
    first_trigger: SystemTime,
    /// Last trigger time
    last_trigger: SystemTime,
    /// Trigger count
    trigger_count: u32,
    /// Is currently alerting
    is_alerting: bool,
    /// Last alert sent
    last_alert_sent: Option<SystemTime>,
}

/// Alert manager
pub struct AlertManager {
    /// Configuration
    config: MonitoringConfig,
    /// Alert rules
    rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    /// Alert states
    states: Arc<RwLock<HashMap<String, AlertState>>>,
    /// Active alerts
    active_alerts: Arc<RwLock<VecDeque<Alert>>>,
    /// Alert history
    alert_history: Arc<RwLock<VecDeque<Alert>>>,
    /// Alert checker handle
    checker_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// HTTP client for webhooks
    http_client: reqwest::Client,
}

impl AlertManager {
    /// Create new alert manager
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            rules: Arc::new(RwLock::new(HashMap::new())),
            states: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            alert_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            checker_handle: Arc::new(Mutex::new(None)),
            http_client: reqwest::Client::new(),
        }
    }

    /// Start alert checking
    pub async fn start(&self, interval_secs: u64) -> Result<()> {
        // Add default alert rules
        self.add_default_rules().await?;

        let rules = self.rules.clone();
        let states = self.states.clone();
        let active_alerts = self.active_alerts.clone();
        let alert_history = self.alert_history.clone();
        let config = self.config.clone();
        let http_client = self.http_client.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(interval_secs));

            loop {
                interval.tick().await;

                // Check all rules
                let rule_map = rules.read().await;
                for (rule_id, rule) in rule_map.iter() {
                    if !rule.enabled {
                        continue;
                    }

                    if let Err(e) = Self::check_rule(
                        rule,
                        &states,
                        &active_alerts,
                        &alert_history,
                        &config,
                        &http_client,
                    )
                    .await
                    {
                        error!("Failed to check alert rule {}: {}", rule_id, e);
                    }
                }

                // Clean up old alerts
                Self::cleanup_old_alerts(&active_alerts, &alert_history).await;
            }
        });

        *self.checker_handle.lock().await = Some(handle);
        info!(
            "Alert manager started with interval {} seconds",
            interval_secs
        );
        Ok(())
    }

    /// Add default alert rules
    async fn add_default_rules(&self) -> Result<()> {
        let default_rules = vec![
            AlertRule {
                id: "high_cpu".to_string(),
                name: "High CPU Usage".to_string(),
                alert_type: AlertType::SystemResource,
                severity: AlertSeverity::Warning,
                metric: "cpu_usage".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 80.0,
                duration_secs: 300, // 5 minutes
                cooldown_secs: 600, // 10 minutes
                message_template: "CPU usage is {value}%, exceeding threshold of {threshold}%"
                    .to_string(),
                enabled: true,
            },
            AlertRule {
                id: "high_memory".to_string(),
                name: "High Memory Usage".to_string(),
                alert_type: AlertType::SystemResource,
                severity: AlertSeverity::Warning,
                metric: "memory_usage".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 85.0,
                duration_secs: 300,
                cooldown_secs: 600,
                message_template: "Memory usage is {value}%, exceeding threshold of {threshold}%"
                    .to_string(),
                enabled: true,
            },
            AlertRule {
                id: "low_peer_count".to_string(),
                name: "Low Peer Count".to_string(),
                alert_type: AlertType::NetworkIssue,
                severity: AlertSeverity::Error,
                metric: "peer_count".to_string(),
                condition: AlertCondition::LessThan,
                threshold: 3.0,
                duration_secs: 120, // 2 minutes
                cooldown_secs: 300, // 5 minutes
                message_template: "Peer count is {value}, below threshold of {threshold}"
                    .to_string(),
                enabled: true,
            },
            AlertRule {
                id: "consensus_failure".to_string(),
                name: "Consensus Failures".to_string(),
                alert_type: AlertType::ConsensusFailure,
                severity: AlertSeverity::Critical,
                metric: "consensus_failures".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 5.0,
                duration_secs: 60,  // 1 minute
                cooldown_secs: 300, // 5 minutes
                message_template: "Consensus failures: {value}, exceeding threshold of {threshold}"
                    .to_string(),
                enabled: true,
            },
            AlertRule {
                id: "storage_latency".to_string(),
                name: "High Storage Latency".to_string(),
                alert_type: AlertType::StorageIssue,
                severity: AlertSeverity::Warning,
                metric: "storage_latency".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 1000.0,  // 1 second
                duration_secs: 180, // 3 minutes
                cooldown_secs: 600, // 10 minutes
                message_template:
                    "Storage latency is {value}ms, exceeding threshold of {threshold}ms".to_string(),
                enabled: true,
            },
        ];

        for rule in default_rules {
            self.add_rule(rule).await?;
        }

        Ok(())
    }

    /// Check a single alert rule
    async fn check_rule(
        rule: &AlertRule,
        states: &Arc<RwLock<HashMap<String, AlertState>>>,
        active_alerts: &Arc<RwLock<VecDeque<Alert>>>,
        alert_history: &Arc<RwLock<VecDeque<Alert>>>,
        config: &MonitoringConfig,
        http_client: &reqwest::Client,
    ) -> Result<()> {
        // Get current metric value
        let current_value = Self::get_metric_value(&rule.metric).await?;

        // Check condition
        let condition_met =
            Self::evaluate_condition(&rule.condition, current_value, rule.threshold);

        let now = SystemTime::now();
        let mut states_map = states.write().await;

        let state = states_map.entry(rule.id.clone()).or_insert(AlertState {
            rule_id: rule.id.clone(),
            first_trigger: now,
            last_trigger: now,
            trigger_count: 0,
            is_alerting: false,
            last_alert_sent: None,
        });

        if condition_met {
            if !state.is_alerting {
                // First time condition is met
                state.first_trigger = now;
                state.trigger_count = 1;
            } else {
                state.trigger_count += 1;
            }
            state.last_trigger = now;

            // Check if duration threshold is met
            let duration_met = now
                .duration_since(state.first_trigger)
                .unwrap_or(Duration::from_secs(0))
                .as_secs()
                >= rule.duration_secs;

            // Check cooldown period
            let cooldown_passed = if let Some(last_sent) = state.last_alert_sent {
                now.duration_since(last_sent)
                    .unwrap_or(Duration::from_secs(0))
                    .as_secs()
                    >= rule.cooldown_secs
            } else {
                true
            };

            if duration_met && cooldown_passed {
                // Trigger alert
                let alert = Self::create_alert(rule, current_value, now);

                // Add to active alerts
                let mut active = active_alerts.write().await;
                active.push_back(alert.clone());
                if active.len() > 1000 {
                    active.pop_front();
                }

                // Add to history
                let mut history = alert_history.write().await;
                history.push_back(alert.clone());
                if history.len() > 10000 {
                    history.pop_front();
                }

                // Send webhook notification
                if let Some(webhook_url) = &config.alert_webhook_url {
                    if let Err(e) =
                        Self::send_webhook_notification(http_client, webhook_url, &alert).await
                    {
                        error!("Failed to send webhook notification: {}", e);
                    }
                }

                state.is_alerting = true;
                state.last_alert_sent = Some(now);

                info!("Alert triggered: {} - {}", rule.name, alert.message);
            }
        } else {
            // Condition not met, reset state
            if state.is_alerting {
                info!("Alert resolved: {}", rule.name);
            }
            state.is_alerting = false;
            state.trigger_count = 0;
        }

        Ok(())
    }

    /// Get current metric value
    async fn get_metric_value(metric: &str) -> Result<f64> {
        // In production, this would get values from the metrics registry
        // For now, return simulated values
        match metric {
            "cpu_usage" => Ok(super::CPU_USAGE.get()),
            "memory_usage" => Ok(super::MEMORY_USAGE.get() / (1024.0 * 1024.0)), // Convert to MB
            "peer_count" => Ok(super::PEER_COUNT.get()),
            "consensus_failures" => Ok(super::CONSENSUS_FAILURES.get()),
            "storage_latency" => Ok(100.0), // Simulated value
            _ => Ok(0.0),
        }
    }

    /// Evaluate alert condition
    fn evaluate_condition(condition: &AlertCondition, value: f64, threshold: f64) -> bool {
        match condition {
            AlertCondition::GreaterThan => value > threshold,
            AlertCondition::LessThan => value < threshold,
            AlertCondition::Equals => (value - threshold).abs() < f64::EPSILON,
            AlertCondition::NotEquals => (value - threshold).abs() >= f64::EPSILON,
            AlertCondition::GreaterThanOrEqual => value >= threshold,
            AlertCondition::LessThanOrEqual => value <= threshold,
        }
    }

    /// Create alert from rule and current value
    fn create_alert(rule: &AlertRule, value: f64, timestamp: SystemTime) -> Alert {
        let message = rule
            .message_template
            .replace("{value}", &format!("{:.2}", value))
            .replace("{threshold}", &format!("{:.2}", rule.threshold));

        let mut context = HashMap::new();
        context.insert("rule_id".to_string(), rule.id.clone());
        context.insert("metric".to_string(), rule.metric.clone());
        context.insert("value".to_string(), value.to_string());
        context.insert("threshold".to_string(), rule.threshold.to_string());

        Alert {
            id: format!(
                "{}_{}",
                rule.id,
                timestamp
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            severity: rule.severity.clone(),
            alert_type: rule.alert_type.clone(),
            message,
            component: rule.metric.clone(),
            timestamp,
            context,
        }
    }

    /// Send webhook notification
    async fn send_webhook_notification(
        client: &reqwest::Client,
        webhook_url: &str,
        alert: &Alert,
    ) -> Result<()> {
        let payload = serde_json::json!({
            "alert": alert,
            "timestamp": alert.timestamp,
            "severity": alert.severity,
            "type": alert.alert_type,
            "message": alert.message,
            "component": alert.component
        });

        let response = client
            .post(webhook_url)
            .json(&payload)
            .timeout(Duration::from_secs(10))
            .send()
            .await?;

        if response.status().is_success() {
            info!("Webhook notification sent successfully");
        } else {
            warn!(
                "Webhook notification failed with status: {}",
                response.status()
            );
        }

        Ok(())
    }

    /// Clean up old alerts
    async fn cleanup_old_alerts(
        active_alerts: &Arc<RwLock<VecDeque<Alert>>>,
        alert_history: &Arc<RwLock<VecDeque<Alert>>>,
    ) {
        let cutoff = SystemTime::now() - Duration::from_secs(3600); // Keep alerts for 1 hour

        // Clean active alerts
        let mut active = active_alerts.write().await;
        while let Some(alert) = active.front() {
            if alert.timestamp < cutoff {
                active.pop_front();
            } else {
                break;
            }
        }

        // Clean history (keep more for longer)
        let history_cutoff = SystemTime::now() - Duration::from_secs(24 * 3600 * 7); // Keep for 1 week
        let mut history = alert_history.write().await;
        while let Some(alert) = history.front() {
            if alert.timestamp < history_cutoff {
                history.pop_front();
            } else {
                break;
            }
        }
    }

    /// Add alert rule
    pub async fn add_rule(&self, rule: AlertRule) -> Result<()> {
        self.rules.write().await.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Remove alert rule
    pub async fn remove_rule(&self, rule_id: &str) -> Result<()> {
        self.rules.write().await.remove(rule_id);
        self.states.write().await.remove(rule_id);
        Ok(())
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.read().await.iter().cloned().collect()
    }

    /// Get alert history
    pub async fn get_alert_history(&self, limit: Option<usize>) -> Vec<Alert> {
        let history = self.alert_history.read().await;
        let limit = limit.unwrap_or(100);
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Stop alert manager
    pub async fn stop(&self) {
        if let Some(handle) = self.checker_handle.lock().await.take() {
            handle.abort();
        }
        info!("Alert manager stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_alert_manager() {
        let config = MonitoringConfig::default();
        let alert_manager = AlertManager::new(config);

        // Add a test rule
        let rule = AlertRule {
            id: "test_rule".to_string(),
            name: "Test Rule".to_string(),
            alert_type: AlertType::SystemResource,
            severity: AlertSeverity::Warning,
            metric: "cpu_usage".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 50.0,
            duration_secs: 1,
            cooldown_secs: 1,
            message_template: "Test alert: {value}".to_string(),
            enabled: true,
        };

        alert_manager.add_rule(rule).await.unwrap();

        // Start alert manager
        alert_manager.start(1).await.unwrap();

        // Wait a bit
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop alert manager
        alert_manager.stop().await;
    }
}
