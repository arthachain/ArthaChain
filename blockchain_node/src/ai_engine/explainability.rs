use crate::ai_engine::security::NodeScore;
use crate::utils::security_logger::{SecurityCategory, SecurityLevel, SecurityLogger};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Mutex;

/// Score change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreChangeEvent {
    /// Node ID
    pub node_id: String,
    /// Timestamp of change
    pub timestamp: SystemTime,
    /// Previous score
    pub previous_score: f32,
    /// New score
    pub new_score: f32,
    /// Score difference
    pub score_delta: f32,
    /// Metric that changed
    pub metric_type: String,
    /// Reason for the change
    pub reason: String,
    /// Evidence for the change
    pub evidence: Option<String>,
    /// Contributing factors (with weights)
    pub factors: HashMap<String, f32>,
}

/// Parameters for recording score changes
#[derive(Debug, Clone)]
pub struct ScoreChangeParams {
    /// Node ID
    pub node_id: String,
    /// Previous score
    pub previous_score: f32,
    /// New score
    pub new_score: f32,
    /// Metric that changed
    pub metric_type: String,
    /// Reason for the change
    pub reason: String,
    /// Evidence for the change
    pub evidence: Option<String>,
    /// Contributing factors (with weights)
    pub factors: HashMap<String, f32>,
}

/// Score decision explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreExplanation {
    /// Node ID
    pub node_id: String,
    /// Overall score
    pub overall_score: f32,
    /// Device health score
    pub device_health_score: f32,
    /// Network score
    pub network_score: f32,
    /// Storage score
    pub storage_score: f32,
    /// Engagement score
    pub engagement_score: f32,
    /// AI behavior score
    pub ai_behavior_score: f32,
    /// Timestamp of explanation
    pub timestamp: SystemTime,
    /// Score history (last 10 changes)
    pub recent_changes: Vec<ScoreChangeEvent>,
    /// Component explanations (detailed)
    pub component_explanations: HashMap<String, String>,
    /// Factors with highest positive impact
    pub positive_factors: Vec<(String, f32)>,
    /// Factors with highest negative impact
    pub negative_factors: Vec<(String, f32)>,
    /// Overall explanation
    pub summary: String,
}

/// Confidence level for AI decisions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Very high confidence (>90%)
    VeryHigh,
    /// High confidence (70-90%)
    High,
    /// Medium confidence (50-70%)
    Medium,
    /// Low confidence (30-50%)
    Low,
    /// Very low confidence (<30%)
    VeryLow,
}

/// Feature importance record for explainability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature name
    pub feature: String,
    /// Importance score (0-1)
    pub importance: f32,
    /// Description of the feature
    pub description: String,
}

/// AI Security decision explainer
pub struct AIExplainer {
    /// Security logger
    security_logger: Arc<SecurityLogger>,
    /// Recent score changes
    recent_changes: Arc<Mutex<HashMap<String, Vec<ScoreChangeEvent>>>>,
    /// Feature importance maps
    feature_importance: HashMap<String, Vec<FeatureImportance>>,
    /// Component explanations
    component_explanations: HashMap<String, HashMap<String, String>>,
}

impl AIExplainer {
    /// Create a new AI explainer
    pub fn new(security_logger: Arc<SecurityLogger>) -> Self {
        let mut feature_importance = HashMap::new();

        // Initialize feature importance for device health metrics
        feature_importance.insert(
            "device_health".to_string(),
            vec![
                FeatureImportance {
                    feature: "cpu_usage".to_string(),
                    importance: 0.2,
                    description: "CPU utilization affects node reliability".to_string(),
                },
                FeatureImportance {
                    feature: "memory_usage".to_string(),
                    importance: 0.2,
                    description: "Memory utilization affects node reliability".to_string(),
                },
                FeatureImportance {
                    feature: "disk_available".to_string(),
                    importance: 0.15,
                    description: "Available disk space affects storage capability".to_string(),
                },
                FeatureImportance {
                    feature: "uptime".to_string(),
                    importance: 0.25,
                    description: "Node uptime indicates stability".to_string(),
                },
                FeatureImportance {
                    feature: "avg_response_time".to_string(),
                    importance: 0.2,
                    description: "Response time affects transaction processing".to_string(),
                },
            ],
        );

        // Network metrics importance
        feature_importance.insert(
            "network".to_string(),
            vec![
                FeatureImportance {
                    feature: "latency".to_string(),
                    importance: 0.25,
                    description: "Network latency affects consensus participation".to_string(),
                },
                FeatureImportance {
                    feature: "connection_stability".to_string(),
                    importance: 0.3,
                    description: "Connection stability affects reliability".to_string(),
                },
                FeatureImportance {
                    feature: "peer_count".to_string(),
                    importance: 0.15,
                    description: "Number of peers affects network integration".to_string(),
                },
                FeatureImportance {
                    feature: "packet_loss".to_string(),
                    importance: 0.2,
                    description: "Packet loss rate affects data transfer reliability".to_string(),
                },
                FeatureImportance {
                    feature: "sync_status".to_string(),
                    importance: 0.1,
                    description: "Sync status affects block propagation".to_string(),
                },
            ],
        );

        // AI behavior metrics importance
        feature_importance.insert(
            "ai_behavior".to_string(),
            vec![
                FeatureImportance {
                    feature: "anomaly_score".to_string(),
                    importance: 0.25,
                    description: "Anomaly detection identifies unusual behaviors".to_string(),
                },
                FeatureImportance {
                    feature: "fraud_probability".to_string(),
                    importance: 0.2,
                    description: "Fraud probability measures likelihood of malicious behavior"
                        .to_string(),
                },
                FeatureImportance {
                    feature: "threat_level".to_string(),
                    importance: 0.2,
                    description: "Threat level assesses security risk".to_string(),
                },
                FeatureImportance {
                    feature: "pattern_consistency".to_string(),
                    importance: 0.15,
                    description: "Pattern consistency tracks behavioral stability".to_string(),
                },
                FeatureImportance {
                    feature: "sybil_probability".to_string(),
                    importance: 0.2,
                    description: "Sybil probability measures identity manipulation risk"
                        .to_string(),
                },
            ],
        );

        // Create component explanation templates
        let mut component_explanations = HashMap::new();

        // Device health explanations
        let mut device_explanations = HashMap::new();
        device_explanations.insert(
            "high".to_string(),
            "The node has excellent hardware performance with good resource utilization. CPU and memory usage are within optimal ranges, and the node has demonstrated stable uptime with minimal dropped connections.".to_string()
        );
        device_explanations.insert(
            "medium".to_string(),
            "The node has adequate hardware performance but shows some resource constraints. CPU or memory usage occasionally spikes, and there have been some brief periods of unavailability.".to_string()
        );
        device_explanations.insert(
            "low".to_string(),
            "The node has poor hardware performance with concerning resource utilization. CPU or memory usage frequently exceeds optimal ranges, and the node has experienced significant downtime or dropped connections.".to_string()
        );
        component_explanations.insert("device_health".to_string(), device_explanations);

        // Network explanations
        let mut network_explanations = HashMap::new();
        network_explanations.insert(
            "high".to_string(),
            "The node has excellent network connectivity with low latency and packet loss. It maintains a healthy number of peer connections and demonstrates stable network participation.".to_string()
        );
        network_explanations.insert(
            "medium".to_string(),
            "The node has adequate network connectivity but shows occasional latency spikes or packet loss. Peer connections fluctuate, and there have been brief periods of network instability.".to_string()
        );
        network_explanations.insert(
            "low".to_string(),
            "The node has poor network connectivity with high latency and packet loss. It struggles to maintain peer connections and shows significant network instability.".to_string()
        );
        component_explanations.insert("network".to_string(), network_explanations);

        // AI behavior explanations
        let mut ai_explanations = HashMap::new();
        ai_explanations.insert(
            "high".to_string(),
            "The node demonstrates consistent and trustworthy behavior patterns. Anomaly detection and fraud probability scores are very low, and the node's actions align with expected legitimate behavior.".to_string()
        );
        ai_explanations.insert(
            "medium".to_string(),
            "The node shows mostly consistent behavior with some occasional anomalies. Certain actions have triggered moderate anomaly scores, but there's insufficient evidence to confirm malicious intent.".to_string()
        );
        ai_explanations.insert(
            "low".to_string(),
            "The node exhibits concerning behavior patterns consistent with potential malicious activity. High anomaly scores have been detected, along with suspicious transaction patterns or network behavior.".to_string()
        );
        component_explanations.insert("ai_behavior".to_string(), ai_explanations);

        Self {
            security_logger,
            recent_changes: Arc::new(Mutex::new(HashMap::new())),
            feature_importance,
            component_explanations,
        }
    }

    /// Explain a node score
    pub async fn explain_score(
        &self,
        node_id: &str,
        score: &NodeScore,
    ) -> Result<ScoreExplanation> {
        // Get recent changes for this node
        let recent_changes = {
            let changes_map = self.recent_changes.lock().await;
            changes_map.get(node_id).cloned().unwrap_or_default()
        };

        // Determine score level for each component
        let device_level = Self::get_score_level(score.device_health_score);
        let network_level = Self::get_score_level(score.network_score);
        let storage_level = Self::get_score_level(score.storage_score);
        let engagement_level = Self::get_score_level(score.engagement_score);
        let ai_behavior_level = Self::get_score_level(score.ai_behavior_score);

        // Get explanations for each component
        let mut component_explanations = HashMap::new();

        if let Some(explanations) = self.component_explanations.get("device_health") {
            if let Some(explanation) = explanations.get(&device_level) {
                component_explanations.insert("device_health".to_string(), explanation.clone());
            }
        }

        if let Some(explanations) = self.component_explanations.get("network") {
            if let Some(explanation) = explanations.get(&network_level) {
                component_explanations.insert("network".to_string(), explanation.clone());
            }
        }

        if let Some(explanations) = self.component_explanations.get("ai_behavior") {
            if let Some(explanation) = explanations.get(&ai_behavior_level) {
                component_explanations.insert("ai_behavior".to_string(), explanation.clone());
            }
        }

        // Calculate factors with highest impact
        let mut all_factors = Vec::new();

        // Add device health factors
        if let Some(importance_list) = self.feature_importance.get("device_health") {
            for factor in importance_list {
                all_factors.push((
                    format!("device_health.{}", factor.feature),
                    factor.importance * score.device_health_score * 0.2, // 20% weight in overall score
                ));
            }
        }

        // Add network factors
        if let Some(importance_list) = self.feature_importance.get("network") {
            for factor in importance_list {
                all_factors.push((
                    format!("network.{}", factor.feature),
                    factor.importance * score.network_score * 0.3, // 30% weight in overall score
                ));
            }
        }

        // Add AI behavior factors
        if let Some(importance_list) = self.feature_importance.get("ai_behavior") {
            for factor in importance_list {
                all_factors.push((
                    format!("ai_behavior.{}", factor.feature),
                    factor.importance * score.ai_behavior_score * 0.2, // 20% weight in overall score
                ));
            }
        }

        // Sort by impact (absolute value)
        all_factors.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        // Get top positive and negative factors
        let mut positive_factors = Vec::new();
        let mut negative_factors = Vec::new();

        for (factor, impact) in &all_factors {
            if *impact > 0.0 {
                positive_factors.push((factor.clone(), *impact));
            } else {
                negative_factors.push((factor.clone(), *impact));
            }

            if positive_factors.len() >= 5 && negative_factors.len() >= 5 {
                break;
            }
        }

        // Sort by magnitude
        positive_factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        negative_factors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Limit to top 5
        let positive_factors = positive_factors.into_iter().take(5).collect();
        let negative_factors = negative_factors.into_iter().take(5).collect();

        // Generate summary
        let summary = Self::generate_summary(
            score,
            &device_level,
            &network_level,
            &storage_level,
            &engagement_level,
            &ai_behavior_level,
        );

        Ok(ScoreExplanation {
            node_id: node_id.to_string(),
            overall_score: score.overall_score,
            device_health_score: score.device_health_score,
            network_score: score.network_score,
            storage_score: score.storage_score,
            engagement_score: score.engagement_score,
            ai_behavior_score: score.ai_behavior_score,
            timestamp: SystemTime::now(),
            recent_changes: recent_changes.into_iter().take(10).collect(),
            component_explanations,
            positive_factors,
            negative_factors,
            summary,
        })
    }

    /// Record a score change event
    pub async fn record_score_change(&self, params: ScoreChangeParams) -> Result<()> {
        let event = ScoreChangeEvent {
            node_id: params.node_id.clone(),
            timestamp: SystemTime::now(),
            previous_score: params.previous_score,
            new_score: params.new_score,
            score_delta: params.new_score - params.previous_score,
            metric_type: params.metric_type.clone(),
            reason: params.reason.clone(),
            evidence: params.evidence.clone(),
            factors: params.factors.clone(),
        };

        // Add to recent changes
        {
            let mut changes_map = self.recent_changes.lock().await;
            let node_changes = changes_map.entry(params.node_id.clone()).or_default();

            node_changes.push(event.clone());

            // Keep only the last 100 changes
            if node_changes.len() > 100 {
                let excess = node_changes.len() - 100;
                node_changes.drain(0..excess);
            }
        }

        // Log significant changes
        let security_level = if (params.new_score - params.previous_score).abs() > 0.1 {
            if params.new_score < params.previous_score {
                SecurityLevel::Medium
            } else {
                SecurityLevel::Low
            }
        } else {
            SecurityLevel::Info
        };

        let category = match params.metric_type.as_str() {
            "device_health" => SecurityCategory::NodeBehavior,
            "network" => SecurityCategory::Network,
            "ai_behavior" => SecurityCategory::NodeBehavior,
            _ => SecurityCategory::NodeBehavior,
        };

        // Log to security logger
        self.security_logger
            .log_event(
                security_level,
                category,
                Some(&params.node_id),
                &format!(
                    "{} score changed by {:.4}: {}",
                    params.metric_type,
                    params.new_score - params.previous_score,
                    params.reason
                ),
                serde_json::to_value(&event)?,
            )
            .await?;

        Ok(())
    }

    /// Get confidence level for a score
    pub fn get_confidence_level(score: f32, evidence_count: usize) -> ConfidenceLevel {
        // Higher confidence with more evidence
        let base_confidence = if evidence_count > 20 {
            0.9
        } else if evidence_count > 10 {
            0.8
        } else if evidence_count > 5 {
            0.7
        } else {
            0.6
        };

        // Adjust based on score extremes
        let score_confidence = if !(0.2..=0.8).contains(&score) {
            // More confident about very high or very low scores
            0.9
        } else {
            // Less confident about middle scores
            0.7
        };

        let combined_confidence = (base_confidence + score_confidence) / 2.0;

        match combined_confidence {
            c if c >= 0.9 => ConfidenceLevel::VeryHigh,
            c if c >= 0.7 => ConfidenceLevel::High,
            c if c >= 0.5 => ConfidenceLevel::Medium,
            c if c >= 0.3 => ConfidenceLevel::Low,
            _ => ConfidenceLevel::VeryLow,
        }
    }

    /// Get factors most affecting a metric score
    pub fn explain_metric_score(
        &self,
        metric_type: &str,
        _metric_score: f32,
        metrics: &serde_json::Value,
    ) -> Result<HashMap<String, f32>> {
        let mut factors = HashMap::new();

        // Get feature importance for this metric
        if let Some(importance_list) = self.feature_importance.get(metric_type) {
            for feature in importance_list {
                // Try to get the feature value from metrics
                if let Ok(feature_value) = Self::extract_feature_value(metrics, &feature.feature) {
                    // Normalize feature value to 0-1 scale
                    let normalized_value =
                        Self::normalize_feature_value(&feature.feature, feature_value)?;

                    // Calculate contribution
                    let contribution = normalized_value * feature.importance;
                    factors.insert(feature.feature.clone(), contribution);
                }
            }
        }

        Ok(factors)
    }

    /// Helper to extract feature value from JSON metrics
    fn extract_feature_value(metrics: &serde_json::Value, feature: &str) -> Result<f64> {
        if let Some(value) = metrics.get(feature) {
            if let Some(num) = value.as_f64() {
                return Ok(num);
            } else if let Some(integer) = value.as_i64() {
                return Ok(integer as f64);
            } else if let Some(boolean) = value.as_bool() {
                return Ok(if boolean { 1.0 } else { 0.0 });
            }
        }

        Err(anyhow::anyhow!(
            "Feature not found or not numeric: {}",
            feature
        ))
    }

    /// Normalize a feature value to 0-1 scale
    fn normalize_feature_value(
        feature_name: &str,
        feature_value: f64,
    ) -> Result<f32, anyhow::Error> {
        // Handle specific feature normalizations based on name
        let normalized = match feature_name {
            "cpu_usage" => {
                // Lower is better, optimal is 20-60%
                if feature_value <= 20.0 {
                    1.0
                } else if feature_value <= 60.0 {
                    1.0 - (feature_value - 20.0) / 40.0 * 0.3
                } else {
                    0.7 - (feature_value - 60.0) / 40.0 * 0.7
                }
            }
            "memory_usage" => {
                // Lower is better, optimal is 30-70%
                if feature_value <= 30.0 {
                    1.0
                } else if feature_value <= 70.0 {
                    1.0 - (feature_value - 30.0) / 40.0 * 0.3
                } else {
                    0.7 - (feature_value - 70.0) / 30.0 * 0.7
                }
            }
            "disk_available" => {
                // Higher is better, GB scale
                let gb_value = feature_value / 1_000_000_000.0;
                if gb_value >= 100.0 {
                    1.0
                } else if gb_value >= 10.0 {
                    0.7 + (gb_value - 10.0) / 90.0 * 0.3
                } else if gb_value >= 1.0 {
                    0.3 + (gb_value - 1.0) / 9.0 * 0.4
                } else {
                    gb_value * 0.3
                }
            }
            "gb_model_size" => {
                // Normalize model size: models over 3GB get a 1.0 score
                if feature_value >= 3.0 {
                    1.0
                } else {
                    // Scale proportionally for smaller models
                    feature_value / 3.0
                }
            }
            "inference_time_ms" => {
                // Normalize inference time: lower is better
                // Values under 100ms get high scores, over 1000ms get low scores
                if feature_value <= 100.0 {
                    1.0
                } else if feature_value >= 1000.0 {
                    0.1
                } else {
                    // Linear scaling between 100ms and 1000ms
                    1.0 - (feature_value - 100.0) / 900.0
                }
            }
            "latency" => {
                // Lower is better (ms)
                if feature_value <= 50.0 {
                    1.0
                } else if feature_value <= 200.0 {
                    1.0 - (feature_value - 50.0) / 150.0 * 0.5
                } else if feature_value <= 1000.0 {
                    0.5 - (feature_value - 200.0) / 800.0 * 0.5
                } else {
                    0.0
                }
            }
            "packet_loss" => {
                // Lower is better (percentage)
                1.0 - feature_value
            }
            "connection_stability" => {
                // Higher is better (already 0-1)
                feature_value
            }
            "accuracy_score" | "f1_score" | "precision" | "recall" => {
                // These metrics are already in 0-1 range
                feature_value
            }
            "anomaly_score" | "fraud_probability" | "threat_level" | "sybil_probability" => {
                // Lower is better (already 0-1)
                1.0 - feature_value
            }
            "pattern_consistency" => {
                // Higher is better (already 0-1)
                feature_value
            }
            // Default normalization for unknown features
            _ => {
                if (0.0..=1.0).contains(&feature_value) {
                    // Already normalized
                    feature_value
                } else if feature_value >= 0.0 {
                    // Assume higher is better with diminishing returns
                    1.0 - (1.0 / (1.0 + feature_value / 100.0))
                } else {
                    // Negative values normalized to 0
                    0.0
                }
            }
        };

        Ok(normalized as f32)
    }

    /// Get score level (low/medium/high) as a string
    fn get_score_level(score: f32) -> String {
        if score >= 0.8 {
            "high".to_string()
        } else if score >= 0.5 {
            "medium".to_string()
        } else {
            "low".to_string()
        }
    }

    /// Generate a summary based on component scores
    fn generate_summary(
        score: &NodeScore,
        device_level: &str,
        network_level: &str,
        _storage_level: &str,
        _engagement_level: &str,
        ai_behavior_level: &str,
    ) -> String {
        let trust_tier = if score.overall_score >= 0.9 {
            "Diamond"
        } else if score.overall_score >= 0.7 {
            "Standard"
        } else if score.overall_score >= 0.5 {
            "Limited"
        } else {
            "Restricted"
        };

        let mut summary = format!(
            "Node has an overall trust score of {:.2} ({trust_tier}). ",
            score.overall_score
        );

        // Check for critical issues
        let mut critical_issues = Vec::new();

        if score.ai_behavior_score < 0.5 {
            critical_issues.push("suspicious behavior patterns");
        }

        if score.device_health_score < 0.5 {
            critical_issues.push("unreliable device health");
        }

        if score.network_score < 0.5 {
            critical_issues.push("poor network connectivity");
        }

        if !critical_issues.is_empty() {
            summary.push_str(&format!(
                "Critical issues detected: {}. ",
                critical_issues.join(", ")
            ));
        }

        // Add component details
        summary.push_str(&format!("Device health is {} ({:.2}), network performance is {} ({:.2}), and AI behavior trustworthiness is {} ({:.2}). ",
            device_level, score.device_health_score,
            network_level, score.network_score,
            ai_behavior_level, score.ai_behavior_score));

        // Add qualification for consensus
        if score.overall_score >= 0.6 {
            summary.push_str("Node qualifies for consensus participation.");
        } else {
            summary.push_str(&format!("Node does not qualify for consensus participation. Minimum required score: 0.60, current: {:.2}",
                score.overall_score));
        }

        summary
    }

    /// Get recent score changes for a node
    pub async fn get_recent_changes(&self, node_id: &str) -> Vec<ScoreChangeEvent> {
        let changes_map = self.recent_changes.lock().await;
        changes_map.get(node_id).cloned().unwrap_or_default()
    }

    /// Export all explanations to JSON
    pub async fn export_explanations(
        &self,
        node_scores: &HashMap<String, NodeScore>,
    ) -> Result<String> {
        let mut explanations = Vec::new();

        for (node_id, score) in node_scores {
            let explanation = self.explain_score(node_id, score).await?;
            explanations.push(explanation);
        }

        let json = serde_json::to_string_pretty(&explanations)?;
        Ok(json)
    }
}

/// CLI report generator for the explainability system
pub async fn generate_explainability_report(
    explainer: &AIExplainer,
    node_id: &str,
    score: &NodeScore,
) -> Result<String> {
    let explanation = explainer.explain_score(node_id, score).await?;

    let mut report = String::new();
    report.push_str(&format!("# AI Score Explanation for Node: {node_id}\n\n"));
    let timestamp =
        chrono::DateTime::<chrono::Local>::from(explanation.timestamp).format("%Y-%m-%d %H:%M:%S");
    report.push_str(&format!("**Generated:** {timestamp}\n\n"));

    report.push_str("## Summary\n\n");
    report.push_str(&explanation.summary);
    report.push_str("\n\n");

    report.push_str("## Score Components\n\n");
    report.push_str("| Component | Score | Level |\n");
    report.push_str("|-----------|-------|-------|\n");
    report.push_str(&format!(
        "| Overall | {:.2} | {} |\n",
        explanation.overall_score,
        AIExplainer::get_score_level(explanation.overall_score)
    ));
    report.push_str(&format!(
        "| Device Health | {:.2} | {} |\n",
        explanation.device_health_score,
        AIExplainer::get_score_level(explanation.device_health_score)
    ));
    report.push_str(&format!(
        "| Network | {:.2} | {} |\n",
        explanation.network_score,
        AIExplainer::get_score_level(explanation.network_score)
    ));
    report.push_str(&format!(
        "| Storage | {:.2} | {} |\n",
        explanation.storage_score,
        AIExplainer::get_score_level(explanation.storage_score)
    ));
    report.push_str(&format!(
        "| Engagement | {:.2} | {} |\n",
        explanation.engagement_score,
        AIExplainer::get_score_level(explanation.engagement_score)
    ));
    report.push_str(&format!(
        "| AI Behavior | {:.2} | {} |\n",
        explanation.ai_behavior_score,
        AIExplainer::get_score_level(explanation.ai_behavior_score)
    ));
    report.push('\n');

    report.push_str("## Component Explanations\n\n");
    for (component, explanation_text) in &explanation.component_explanations {
        report.push_str(&format!("### {component}\n\n"));
        report.push_str(&format!("{explanation_text}\n\n"));
    }

    report.push_str("## Top Positive Factors\n\n");
    if explanation.positive_factors.is_empty() {
        report.push_str("No significant positive factors.\n\n");
    } else {
        report.push_str("| Factor | Impact |\n");
        report.push_str("|--------|--------|\n");
        for (factor, impact) in &explanation.positive_factors {
            report.push_str(&format!("| {factor} | +{impact:.4} |\n"));
        }
        report.push('\n');
    }

    report.push_str("## Top Negative Factors\n\n");
    if explanation.negative_factors.is_empty() {
        report.push_str("No significant negative factors.\n\n");
    } else {
        report.push_str("| Factor | Impact |\n");
        report.push_str("|--------|--------|\n");
        for (factor, impact) in &explanation.negative_factors {
            report.push_str(&format!("| {factor} | {impact:.4} |\n"));
        }
        report.push('\n');
    }

    report.push_str("## Recent Score Changes\n\n");
    if explanation.recent_changes.is_empty() {
        report.push_str("No recent score changes recorded.\n\n");
    } else {
        report.push_str("| Timestamp | Metric | Previous | New | Change | Reason |\n");
        report.push_str("|-----------|--------|----------|-----|--------|--------|\n");

        for change in &explanation.recent_changes {
            let timestamp =
                chrono::DateTime::<chrono::Local>::from(change.timestamp).format("%Y-%m-%d %H:%M");

            report.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} | {:.2} | {} |\n",
                timestamp,
                change.metric_type,
                change.previous_score,
                change.new_score,
                change.score_delta,
                change.reason
            ));
        }
        report.push('\n');
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ai_explainer() {
        // Setup a temporary path for logs
        let temp_dir = tempfile::tempdir().unwrap();
        let log_path = temp_dir.path().join("security.log");

        // Create security logger
        let security_logger =
            Arc::new(SecurityLogger::new(log_path.to_str().unwrap(), 100).unwrap());

        // Create explainer
        let explainer = AIExplainer::new(security_logger);

        // Create test score
        let score = NodeScore {
            overall_score: 0.75,
            device_health_score: 0.8,
            network_score: 0.7,
            storage_score: 0.65,
            engagement_score: 0.78,
            ai_behavior_score: 0.82,
            last_updated: SystemTime::now(),
            history: vec![(SystemTime::now(), 0.75)],
        };

        // Record a score change
        let mut factors = HashMap::new();
        factors.insert("cpu_usage".to_string(), 0.2);
        factors.insert("memory_usage".to_string(), -0.1);

        let params = ScoreChangeParams {
            node_id: "test-node".to_string(),
            previous_score: 0.7,
            new_score: 0.75,
            metric_type: "device_health".to_string(),
            reason: "Improved CPU performance".to_string(),
            evidence: Some("CPU usage decreased from 85% to 45%".to_string()),
            factors,
        };

        explainer.record_score_change(params).await.unwrap();

        // Get explanation
        let explanation = explainer.explain_score("test-node", &score).await.unwrap();

        // Verify explanation
        assert_eq!(explanation.node_id, "test-node");
        assert_eq!(explanation.overall_score, 0.75);
        assert_eq!(explanation.device_health_score, 0.8);
        assert!(!explanation.recent_changes.is_empty());
        assert!(!explanation.summary.is_empty());

        // Verify explanations for components
        assert!(explanation
            .component_explanations
            .contains_key("device_health"));
        assert!(explanation.component_explanations.contains_key("network"));
        assert!(explanation
            .component_explanations
            .contains_key("ai_behavior"));

        // Generate report
        let report = generate_explainability_report(&explainer, "test-node", &score)
            .await
            .unwrap();
        assert!(report.contains("AI Score Explanation for Node: test-node"));
    }
}
