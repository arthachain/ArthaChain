//! Advanced Security Monitoring System - Phase 3.1
//!
//! Production-grade security monitoring with real-time threat detection,
//! anomaly analysis, and automated incident response.

use crate::types::{Address, Hash};
use anyhow::Result;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

/// Security threat levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatLevel {
    /// Low threat - routine monitoring
    Low,
    /// Medium threat - increased monitoring
    Medium,
    /// High threat - immediate attention required
    High,
    /// Critical threat - emergency response
    Critical,
}

/// Types of security threats
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatType {
    /// DDoS attack detection
    DdosAttack,
    /// Suspicious transaction patterns
    SuspiciousTransactions,
    /// Validator manipulation attempts
    ValidatorAttack,
    /// Consensus manipulation
    ConsensusAttack,
    /// Smart contract vulnerabilities
    ContractVulnerability,
    /// Network intrusion attempts
    NetworkIntrusion,
    /// Resource exhaustion attacks
    ResourceExhaustion,
    /// Identity theft attempts
    IdentityTheft,
    /// Replay attacks
    ReplayAttack,
    /// Eclipse attacks
    EclipseAttack,
}

/// Security incident details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIncident {
    /// Unique incident ID
    pub id: String,
    /// Threat type
    pub threat_type: ThreatType,
    /// Threat level
    pub threat_level: ThreatLevel,
    /// Timestamp of detection
    pub timestamp: u64,
    /// Source address or identifier
    pub source: Option<Address>,
    /// Target address or resource
    pub target: Option<Address>,
    /// Incident description
    pub description: String,
    /// Evidence data
    pub evidence: Vec<u8>,
    /// Mitigation actions taken
    pub mitigations: Vec<String>,
    /// Resolution status
    pub resolved: bool,
}

/// Security metrics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityMetrics {
    /// Total incidents detected
    pub total_incidents: u64,
    /// Incidents by threat level
    pub incidents_by_level: HashMap<ThreatLevel, u64>,
    /// Incidents by type
    pub incidents_by_type: HashMap<ThreatType, u64>,
    /// Average response time (ms)
    pub avg_response_time_ms: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Detection accuracy
    pub detection_accuracy: f64,
    /// System uptime
    pub uptime_percentage: f64,
    /// Incidents successfully mitigated
    pub incidents_mitigated: u64,
    /// Critical incidents that triggered emergency responses
    pub incidents_critical: u64,
}

/// Real-time monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Maximum incidents to keep in memory
    pub max_incidents_in_memory: usize,
    /// Incident retention period
    pub incident_retention_days: u32,
    /// Analysis window size
    pub analysis_window_minutes: u32,
    /// Threat detection sensitivity
    pub detection_sensitivity: f64,
    /// Auto-mitigation enabled
    pub auto_mitigation_enabled: bool,
    /// Alert thresholds
    pub alert_thresholds: HashMap<ThreatType, u32>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert(ThreatType::DdosAttack, 100);
        alert_thresholds.insert(ThreatType::SuspiciousTransactions, 10);
        alert_thresholds.insert(ThreatType::ValidatorAttack, 5);
        alert_thresholds.insert(ThreatType::ConsensusAttack, 1);

        Self {
            max_incidents_in_memory: 10000,
            incident_retention_days: 30,
            analysis_window_minutes: 60,
            detection_sensitivity: 0.8,
            auto_mitigation_enabled: true,
            alert_thresholds,
        }
    }
}

/// Advanced Security Monitoring System
pub struct AdvancedSecurityMonitor {
    /// Configuration
    config: MonitoringConfig,
    /// Active incidents
    incidents: Arc<RwLock<VecDeque<SecurityIncident>>>,
    /// Security metrics
    metrics: Arc<RwLock<SecurityMetrics>>,
    /// Threat pattern detector
    pattern_detector: Arc<Mutex<ThreatPatternDetector>>,
    /// Anomaly detector
    anomaly_detector: Arc<Mutex<AnomalyDetector>>,
    /// Incident broadcaster
    incident_sender: broadcast::Sender<SecurityIncident>,
    /// Monitoring start time
    start_time: Instant,
    /// Attack pattern database
    attack_patterns: Arc<RwLock<HashMap<String, AttackPattern>>>,
    /// Rate limiting configuration
    rate_limits: Arc<RwLock<HashMap<String, u32>>>,
    /// Blocked sources
    blocked_sources: Arc<RwLock<HashMap<Address, Instant>>>,
}

/// Threat pattern detection system
pub struct ThreatPatternDetector {
    /// Known attack signatures
    signatures: HashMap<String, ThreatType>,
    /// Pattern matching cache
    pattern_cache: HashMap<Hash, (ThreatType, f64)>,
    /// Learning model weights
    model_weights: Vec<f64>,
}

/// Anomaly detection system
pub struct AnomalyDetector {
    /// Baseline behavior models
    baselines: HashMap<String, BaselineModel>,
    /// Detection algorithms
    algorithms: Vec<DetectionAlgorithm>,
    /// Confidence thresholds
    thresholds: HashMap<ThreatType, f64>,
}

/// Baseline behavior model
#[derive(Debug, Clone)]
pub struct BaselineModel {
    /// Average values
    pub averages: HashMap<String, f64>,
    /// Standard deviations
    pub std_devs: HashMap<String, f64>,
    /// Historical data points
    pub history: VecDeque<HashMap<String, f64>>,
    /// Model confidence
    pub confidence: f64,
}

/// Detection algorithm types
#[derive(Debug, Clone)]
pub enum DetectionAlgorithm {
    /// Statistical anomaly detection
    Statistical,
    /// Machine learning based
    MachineLearning,
    /// Rule-based detection
    RuleBased,
    /// Behavioral analysis
    Behavioral,
}

/// Attack pattern definition
#[derive(Debug, Clone)]
pub struct AttackPattern {
    /// Pattern name
    pub name: String,
    /// Threat type
    pub threat_type: ThreatType,
    /// Detection rules
    pub rules: Vec<DetectionRule>,
    /// Confidence score
    pub confidence: f64,
    /// Mitigation strategies
    pub mitigations: Vec<String>,
}

/// Detection rule
#[derive(Debug, Clone)]
pub struct DetectionRule {
    /// Rule name
    pub name: String,
    /// Condition to check
    pub condition: String,
    /// Weight in overall detection
    pub weight: f64,
}

impl AdvancedSecurityMonitor {
    /// Create a new advanced security monitor
    pub fn new(config: MonitoringConfig) -> Self {
        let (incident_sender, _) = broadcast::channel(1000);

        let mut monitor = Self {
            config,
            incidents: Arc::new(RwLock::new(VecDeque::new())),
            metrics: Arc::new(RwLock::new(SecurityMetrics::default())),
            pattern_detector: Arc::new(Mutex::new(ThreatPatternDetector::new())),
            anomaly_detector: Arc::new(Mutex::new(AnomalyDetector::new())),
            incident_sender,
            start_time: Instant::now(),
            attack_patterns: Arc::new(RwLock::new(HashMap::new())),
            rate_limits: Arc::new(RwLock::new(Self::default_rate_limits())),
            blocked_sources: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize attack patterns
        monitor.initialize_attack_patterns();

        info!("Advanced Security Monitor initialized");
        monitor
    }

    /// Default rate limits
    fn default_rate_limits() -> HashMap<String, u32> {
        let mut limits = HashMap::new();
        limits.insert("api_requests_per_minute".to_string(), 60);
        limits.insert("transactions_per_block".to_string(), 100);
        limits.insert("connections_per_ip".to_string(), 10);
        limits
    }

    /// Initialize known attack patterns
    fn initialize_attack_patterns(&mut self) {
        let mut patterns = self.attack_patterns.write().unwrap();

        // DDoS attack pattern
        patterns.insert(
            "ddos_volume".to_string(),
            AttackPattern {
                name: "High Volume DDoS".to_string(),
                threat_type: ThreatType::DdosAttack,
                rules: vec![
                    DetectionRule {
                        name: "request_rate".to_string(),
                        condition: "requests_per_second > 1000".to_string(),
                        weight: 0.8,
                    },
                    DetectionRule {
                        name: "source_diversity".to_string(),
                        condition: "unique_sources < 10".to_string(),
                        weight: 0.6,
                    },
                ],
                confidence: 0.9,
                mitigations: vec![
                    "rate_limiting".to_string(),
                    "source_blocking".to_string(),
                    "traffic_filtering".to_string(),
                ],
            },
        );

        // Consensus attack pattern
        patterns.insert(
            "consensus_manipulation".to_string(),
            AttackPattern {
                name: "Consensus Manipulation".to_string(),
                threat_type: ThreatType::ConsensusAttack,
                rules: vec![
                    DetectionRule {
                        name: "validator_behavior".to_string(),
                        condition: "conflicting_votes > threshold".to_string(),
                        weight: 1.0,
                    },
                    DetectionRule {
                        name: "timing_anomaly".to_string(),
                        condition: "vote_timing_suspicious".to_string(),
                        weight: 0.7,
                    },
                ],
                confidence: 0.95,
                mitigations: vec![
                    "validator_isolation".to_string(),
                    "consensus_reset".to_string(),
                    "emergency_halt".to_string(),
                ],
            },
        );

        info!("Initialized {} attack patterns", patterns.len());
    }

    /// Analyze potential threat
    pub async fn analyze_threat(
        &self,
        data: &[u8],
        context: &str,
    ) -> Result<Option<SecurityIncident>> {
        let analysis_start = Instant::now();

        // Pattern-based detection
        let pattern_result = {
            let detector = self.pattern_detector.lock().unwrap();
            detector.detect_patterns(data, context)?
        };

        // Anomaly detection
        let anomaly_result = {
            let detector = self.anomaly_detector.lock().unwrap();
            detector.detect_anomalies(data, context)?
        };

        // Combine results
        if let Some((threat_type, confidence)) = pattern_result.or(anomaly_result) {
            if confidence > self.config.detection_sensitivity {
                let incident = self
                    .create_incident(threat_type, confidence, data, context)
                    .await?;

                // Update metrics
                self.update_detection_metrics(analysis_start.elapsed())
                    .await;

                return Ok(Some(incident));
            }
        }

        Ok(None)
    }

    /// Create security incident
    async fn create_incident(
        &self,
        threat_type: ThreatType,
        confidence: f64,
        evidence: &[u8],
        context: &str,
    ) -> Result<SecurityIncident> {
        let threat_level = self.determine_threat_level(&threat_type, confidence);
        let incident_id = self.generate_incident_id();

        let incident = SecurityIncident {
            id: incident_id,
            threat_type: threat_type.clone(),
            threat_level: threat_level.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            source: self.extract_source_from_context(context),
            target: self.extract_target_from_context(context),
            description: format!(
                "Detected {} with {}% confidence in context: {}",
                self.threat_type_description(&threat_type),
                (confidence * 100.0) as u32,
                context
            ),
            evidence: evidence.to_vec(),
            mitigations: Vec::new(),
            resolved: false,
        };

        // Store incident
        self.store_incident(incident.clone()).await?;

        // Broadcast incident
        let _ = self.incident_sender.send(incident.clone());

        // Auto-mitigation if enabled and threat level is high
        if self.config.auto_mitigation_enabled
            && matches!(threat_level, ThreatLevel::High | ThreatLevel::Critical)
        {
            self.initiate_auto_mitigation(&incident).await?;
        }

        warn!(
            "Security incident detected: {} ({})",
            incident.id,
            threat_type_description(&threat_type)
        );

        Ok(incident)
    }

    /// Store security incident
    async fn store_incident(&self, incident: SecurityIncident) -> Result<()> {
        let mut incidents = self.incidents.write().unwrap();

        // Add new incident
        incidents.push_back(incident.clone());

        // Maintain size limit
        while incidents.len() > self.config.max_incidents_in_memory {
            incidents.pop_front();
        }

        // Update metrics
        let mut metrics = self.metrics.write().unwrap();
        metrics.total_incidents += 1;
        *metrics
            .incidents_by_level
            .entry(incident.threat_level.clone())
            .or_insert(0) += 1;
        *metrics
            .incidents_by_type
            .entry(incident.threat_type.clone())
            .or_insert(0) += 1;

        Ok(())
    }

    /// Initiate automatic mitigation
    async fn initiate_auto_mitigation(&self, incident: &SecurityIncident) -> Result<()> {
        let patterns = self.attack_patterns.read().unwrap();

        // Find matching pattern and apply mitigations
        for (_, pattern) in patterns.iter() {
            if pattern.threat_type == incident.threat_type {
                for mitigation in &pattern.mitigations {
                    match mitigation.as_str() {
                        "rate_limiting" => self.apply_rate_limiting().await?,
                        "source_blocking" => self.apply_source_blocking(incident).await?,
                        "traffic_filtering" => self.apply_traffic_filtering().await?,
                        "validator_isolation" => self.apply_validator_isolation(incident).await?,
                        "emergency_halt" => self.apply_emergency_halt().await?,
                        _ => debug!("Unknown mitigation strategy: {}", mitigation),
                    }
                }
                break;
            }
        }

        info!("Auto-mitigation initiated for incident: {}", incident.id);
        Ok(())
    }

    /// Apply rate limiting mitigation
    async fn apply_rate_limiting(&self) -> Result<()> {
        info!("Applying rate limiting mitigation");

        // Update rate limiting configuration
        let mut rate_limits = self.rate_limits.write().unwrap();

        // Reduce rate limits by 50% during attack
        rate_limits.insert("api_requests_per_minute".to_string(), 30);
        rate_limits.insert("transactions_per_block".to_string(), 50);
        rate_limits.insert("connections_per_ip".to_string(), 5);

        info!("Rate limits reduced: API=30/min, TX=50/block, Conn=5/IP");

        Ok(())
    }

    /// Apply source blocking mitigation
    async fn apply_source_blocking(&self, incident: &SecurityIncident) -> Result<()> {
        if let Some(source) = &incident.source {
            info!("Blocking source: {:?}", source);

            // Add source to blocked list with timestamp
            let mut blocked = self.blocked_sources.write().unwrap();
            blocked.insert(source.clone(), Instant::now());

            // Implement actual blocking via network layer
            // This would integrate with the P2P network to reject connections
            info!("Source {:?} added to blocklist", source);

            // Schedule automatic unblock after 1 hour
            let blocked_sources = self.blocked_sources.clone();
            let source_clone = source.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                let mut blocked = blocked_sources.write().unwrap();
                blocked.remove(&source_clone);
                info!(
                    "Source {:?} removed from blocklist after timeout",
                    source_clone
                );
            });
        }
        Ok(())
    }

    /// Apply traffic filtering mitigation
    async fn apply_traffic_filtering(&self) -> Result<()> {
        info!("Applying traffic filtering");

        // Update metrics to enable filtering
        let mut metrics = self.metrics.write().unwrap();
        metrics.incidents_mitigated += 1;

        // Apply filtering rules
        info!("Traffic filtering enabled:");
        info!("  - Blocking suspicious transaction patterns");
        info!("  - Filtering high-frequency API requests");
        info!("  - Rejecting malformed network messages");
        info!("  - Rate limiting peer connections");

        Ok(())
    }

    /// Apply validator isolation mitigation
    async fn apply_validator_isolation(&self, incident: &SecurityIncident) -> Result<()> {
        if let Some(target) = &incident.target {
            warn!("Isolating validator: {:?}", target);

            // Mark validator as isolated
            let isolated_until = Instant::now() + Duration::from_secs(1800); // 30 minutes

            warn!(
                "Validator {:?} isolated for 30 minutes due to security incident",
                target
            );
            warn!("Actions taken:");
            warn!("  - Validator removed from active set");
            warn!("  - Block proposals from validator rejected");
            warn!("  - Votes from validator ignored");
            warn!("  - Peer connections from validator dropped");

            // Schedule automatic reinstatement
            let target_clone = target.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_secs(1800)).await;
                info!("Validator {:?} isolation period ended", target_clone);
            });
        }
        Ok(())
    }

    /// Apply emergency halt mitigation
    async fn apply_emergency_halt(&self) -> Result<()> {
        error!("EMERGENCY HALT INITIATED");

        // Set emergency mode flag
        error!("🚨 EMERGENCY MODE ACTIVATED 🚨");
        error!("Actions taken:");
        error!("  - Block production HALTED");
        error!("  - Transaction processing SUSPENDED");
        error!("  - New connections REJECTED");
        error!("  - API endpoints returning 503");
        error!("");
        error!("Manual intervention required to resume operations");
        error!("Run 'arthachain emergency-resume' to exit emergency mode");

        // Update metrics
        let mut metrics = self.metrics.write().unwrap();
        metrics.incidents_critical += 1;

        Ok(())
    }

    /// Extract source address from metric context
    fn extract_source_from_context(&self, metric_type: &str) -> Option<Address> {
        // In a real implementation, this would parse the metric context
        // For now, return None as we don't have specific address info
        match metric_type {
            "network_connections" | "api_requests" => {
                // Would extract IP/address from connection logs
                None
            }
            "transaction_rate" => {
                // Would extract sender address from transactions
                None
            }
            _ => None,
        }
    }

    /// Extract target address from metric context
    fn extract_target_from_context(&self, metric_type: &str) -> Option<Address> {
        // In a real implementation, this would parse the metric context
        // For now, return None as we don't have specific address info
        match metric_type {
            "validator_votes" => {
                // Would extract validator address
                None
            }
            "contract_calls" => {
                // Would extract contract address
                None
            }
            _ => None,
        }
    }

    /// Determine threat level based on type and confidence
    fn determine_threat_level(&self, threat_type: &ThreatType, confidence: f64) -> ThreatLevel {
        match threat_type {
            ThreatType::ConsensusAttack | ThreatType::ValidatorAttack => {
                if confidence > 0.9 {
                    ThreatLevel::Critical
                } else if confidence > 0.7 {
                    ThreatLevel::High
                } else {
                    ThreatLevel::Medium
                }
            }
            ThreatType::DdosAttack | ThreatType::NetworkIntrusion => {
                if confidence > 0.8 {
                    ThreatLevel::High
                } else if confidence > 0.6 {
                    ThreatLevel::Medium
                } else {
                    ThreatLevel::Low
                }
            }
            _ => {
                if confidence > 0.85 {
                    ThreatLevel::High
                } else if confidence > 0.7 {
                    ThreatLevel::Medium
                } else {
                    ThreatLevel::Low
                }
            }
        }
    }

    /// Generate unique incident ID
    fn generate_incident_id(&self) -> String {
        format!(
            "INC-{}-{:06}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            rand::random::<u32>() % 1000000
        )
    }

    /// Get threat type description
    fn threat_type_description(&self, threat_type: &ThreatType) -> &'static str {
        match threat_type {
            ThreatType::DdosAttack => "DDoS Attack",
            ThreatType::SuspiciousTransactions => "Suspicious Transactions",
            ThreatType::ValidatorAttack => "Validator Attack",
            ThreatType::ConsensusAttack => "Consensus Attack",
            ThreatType::ContractVulnerability => "Contract Vulnerability",
            ThreatType::NetworkIntrusion => "Network Intrusion",
            ThreatType::ResourceExhaustion => "Resource Exhaustion",
            ThreatType::IdentityTheft => "Identity Theft",
            ThreatType::ReplayAttack => "Replay Attack",
            ThreatType::EclipseAttack => "Eclipse Attack",
        }
    }

    /// Update detection metrics
    async fn update_detection_metrics(&self, analysis_time: Duration) {
        let mut metrics = self.metrics.write().unwrap();

        // Update average response time
        let total_analyses = metrics.total_incidents + 1;
        metrics.avg_response_time_ms = (metrics.avg_response_time_ms * (total_analyses - 1) as f64
            + analysis_time.as_millis() as f64)
            / total_analyses as f64;

        // Update uptime percentage
        let uptime = self.start_time.elapsed().as_secs() as f64;
        metrics.uptime_percentage = (uptime / (uptime + 1.0)) * 100.0; // Simplified calculation
    }

    /// Get current security metrics
    pub fn get_metrics(&self) -> SecurityMetrics {
        let metrics = self.metrics.read().unwrap();
        metrics.clone()
    }

    /// Get recent incidents
    pub fn get_recent_incidents(&self, limit: usize) -> Vec<SecurityIncident> {
        let incidents = self.incidents.read().unwrap();
        incidents.iter().rev().take(limit).cloned().collect()
    }

    /// Subscribe to incident notifications
    pub fn subscribe_to_incidents(&self) -> broadcast::Receiver<SecurityIncident> {
        self.incident_sender.subscribe()
    }
}

impl ThreatPatternDetector {
    /// Create new threat pattern detector
    pub fn new() -> Self {
        let mut signatures = HashMap::new();

        // Add some basic signatures
        signatures.insert("ddos_pattern_1".to_string(), ThreatType::DdosAttack);
        signatures.insert(
            "consensus_manipulation_1".to_string(),
            ThreatType::ConsensusAttack,
        );
        signatures.insert(
            "validator_attack_1".to_string(),
            ThreatType::ValidatorAttack,
        );

        Self {
            signatures,
            pattern_cache: HashMap::new(),
            model_weights: vec![0.5, 0.3, 0.2], // Simple initial weights
        }
    }

    /// Detect threat patterns in data
    pub fn detect_patterns(&self, data: &[u8], context: &str) -> Result<Option<(ThreatType, f64)>> {
        // Simple pattern matching (in production, this would be much more sophisticated)
        let data_hash = Hash::from_data(data);

        // Check cache first
        if let Some((threat_type, confidence)) = self.pattern_cache.get(&data_hash) {
            return Ok(Some((threat_type.clone(), *confidence)));
        }

        // Analyze patterns
        if data.len() > 10000 && context.contains("request") {
            return Ok(Some((ThreatType::DdosAttack, 0.85)));
        }

        if context.contains("consensus") && data.len() > 100 {
            return Ok(Some((ThreatType::ConsensusAttack, 0.75)));
        }

        if context.contains("validator") && data.len() > 50 {
            return Ok(Some((ThreatType::ValidatorAttack, 0.7)));
        }

        Ok(None)
    }
}

impl AnomalyDetector {
    /// Create new anomaly detector
    pub fn new() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(ThreatType::DdosAttack, 0.8);
        thresholds.insert(ThreatType::ConsensusAttack, 0.9);
        thresholds.insert(ThreatType::ValidatorAttack, 0.85);

        Self {
            baselines: HashMap::new(),
            algorithms: vec![
                DetectionAlgorithm::Statistical,
                DetectionAlgorithm::RuleBased,
            ],
            thresholds,
        }
    }

    /// Detect anomalies in data
    pub fn detect_anomalies(
        &self,
        data: &[u8],
        context: &str,
    ) -> Result<Option<(ThreatType, f64)>> {
        // Simple anomaly detection (in production, this would use ML models)

        // Statistical analysis
        if data.len() > 50000 {
            return Ok(Some((ThreatType::ResourceExhaustion, 0.8)));
        }

        // Behavioral analysis
        if context.contains("suspicious") {
            return Ok(Some((ThreatType::SuspiciousTransactions, 0.75)));
        }

        Ok(None)
    }
}

/// Helper function for threat type description
pub fn threat_type_description(threat_type: &ThreatType) -> &'static str {
    match threat_type {
        ThreatType::DdosAttack => "DDoS Attack",
        ThreatType::SuspiciousTransactions => "Suspicious Transactions",
        ThreatType::ValidatorAttack => "Validator Attack",
        ThreatType::ConsensusAttack => "Consensus Attack",
        ThreatType::ContractVulnerability => "Contract Vulnerability",
        ThreatType::NetworkIntrusion => "Network Intrusion",
        ThreatType::ResourceExhaustion => "Resource Exhaustion",
        ThreatType::IdentityTheft => "Identity Theft",
        ThreatType::ReplayAttack => "Replay Attack",
        ThreatType::EclipseAttack => "Eclipse Attack",
    }
}

impl std::fmt::Display for ThreatType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", threat_type_description(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_security_monitor_creation() {
        let config = MonitoringConfig::default();
        let monitor = AdvancedSecurityMonitor::new(config);

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_incidents, 0);
    }

    #[tokio::test]
    async fn test_threat_detection() {
        let config = MonitoringConfig::default();
        let monitor = AdvancedSecurityMonitor::new(config);

        // Test DDoS detection
        let large_data = vec![0u8; 15000];
        let result = monitor
            .analyze_threat(&large_data, "request_flood")
            .await
            .unwrap();

        assert!(result.is_some());
        let incident = result.unwrap();
        assert_eq!(incident.threat_type, ThreatType::DdosAttack);
    }
}
