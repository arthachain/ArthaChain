use crate::consensus::byzantine::ByzantineFaultType;
use crate::ledger::block::Block;
use crate::ledger::transaction::Transaction;
use crate::network::types::NodeId;
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Security level for the consensus system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Low security (fast but less secure)
    Low,
    /// Standard security (balanced)
    Standard,
    /// High security (slower but more secure)
    High,
    /// Paranoid (maximum security)
    Paranoid,
}

/// Configuration for the security manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Current security level
    pub security_level: SecurityLevel,
    /// Maximum number of faults to track
    pub max_tracked_faults: usize,
    /// Time window for fault tracking in seconds
    pub fault_tracking_window_secs: u64,
    /// Enable automatic banning of malicious validators
    pub enable_auto_ban: bool,
    /// Threshold for automatic banning (number of faults)
    pub auto_ban_threshold: usize,
    /// Duration of bans in seconds
    pub ban_duration_secs: u64,
    /// Threshold percentage for byzantine fault detection
    pub byzantine_detection_threshold: f64,
    /// Enable neural network detection
    pub enable_neural_detection: bool,
    /// Enable signature verification
    pub verify_signatures: bool,
    /// Verify all transactions in blocks
    pub verify_all_transactions: bool,
    /// Enable equivocation protection
    pub enable_equivocation_protection: bool,
    /// Time synchronization threshold in milliseconds
    pub time_sync_threshold_ms: u64,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::Standard,
            max_tracked_faults: 1000,
            fault_tracking_window_secs: 3600,
            enable_auto_ban: true,
            auto_ban_threshold: 5,
            ban_duration_secs: 86400,
            byzantine_detection_threshold: 0.67,
            enable_neural_detection: false,
            verify_signatures: true,
            verify_all_transactions: true,
            enable_equivocation_protection: true,
            time_sync_threshold_ms: 500,
        }
    }
}

/// A security incident in the consensus system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIncident {
    /// Unique ID for the incident
    pub id: String,
    /// Type of fault
    pub fault_type: ByzantineFaultType,
    /// Node responsible for the incident
    pub node_id: NodeId,
    /// Timestamp of the incident
    pub timestamp: u64,
    /// Block height when incident occurred
    pub block_height: Option<u64>,
    /// Evidence supporting the incident
    pub evidence: Vec<u8>,
    /// Severity level
    pub severity: IncidentSeverity,
    /// Additional details
    pub details: HashMap<String, String>,
    /// Witnesses who observed the incident
    pub witnesses: Vec<NodeId>,
}

/// Severity of a security incident
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IncidentSeverity {
    /// Low severity (warning)
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Status of a banned validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanStatus {
    /// Node ID of the banned validator
    pub node_id: NodeId,
    /// Timestamp when ban started
    pub ban_start: u64,
    /// Duration of ban in seconds
    pub ban_duration: u64,
    /// Reason for ban
    pub reason: String,
    /// Related incidents
    pub incidents: Vec<String>,
}

impl BanStatus {
    /// Check if the ban is still active
    pub fn is_active(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now < self.ban_start + self.ban_duration
    }
}

/// Signature verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureVerificationResult {
    /// Whether the signature is valid
    pub is_valid: bool,
    /// Time taken for verification in milliseconds
    pub verification_time_ms: u64,
    /// Error message if verification failed
    pub error: Option<String>,
}

/// The security manager for consensus
pub struct SecurityManager {
    /// Configuration
    config: RwLock<SecurityConfig>,
    /// Incidents
    incidents: RwLock<HashMap<String, SecurityIncident>>,
    /// Banned validators
    banned_validators: RwLock<HashMap<NodeId, BanStatus>>,
    /// Fault counters by validator
    validator_faults: RwLock<HashMap<NodeId, Vec<(u64, ByzantineFaultType)>>>,
    /// Running flag
    running: RwLock<bool>,
    /// Last cleanup time
    last_cleanup: RwLock<Instant>,
    /// Validators
    validators: Arc<RwLock<HashSet<NodeId>>>,
}

impl SecurityManager {
    /// Create a new security manager
    pub fn new(config: SecurityConfig, validators: Arc<RwLock<HashSet<NodeId>>>) -> Self {
        Self {
            config: RwLock::new(config),
            incidents: RwLock::new(HashMap::new()),
            banned_validators: RwLock::new(HashMap::new()),
            validator_faults: RwLock::new(HashMap::new()),
            running: RwLock::new(false),
            last_cleanup: RwLock::new(Instant::now()),
            validators,
        }
    }

    /// Start the security manager
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(anyhow!("Security manager already running"));
        }

        *running = true;

        // Start the cleanup task
        self.start_cleanup_task();

        info!(
            "Security manager started with level: {:?}",
            self.config.read().await.security_level
        );
        Ok(())
    }

    /// Stop the security manager
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(anyhow!("Security manager not running"));
        }

        *running = false;
        info!("Security manager stopped");
        Ok(())
    }

    /// Start the cleanup task
    fn start_cleanup_task(&self) {
        let self_clone = Arc::new(self.clone());

        tokio::spawn(async move {
            let interval = Duration::from_secs(300); // 5 minutes
            let mut timer = tokio::time::interval(interval);

            loop {
                timer.tick().await;

                let is_running = *self_clone.running.read().await;
                if !is_running {
                    break;
                }

                // Cleanup old incidents and faults
                if let Err(e) = self_clone.cleanup().await {
                    warn!("Error during security cleanup: {}", e);
                }
            }
        });
    }

    /// Clean up old incidents and bans
    async fn cleanup(&self) -> Result<()> {
        // Update last cleanup time
        let mut last_cleanup = self.last_cleanup.write().await;
        *last_cleanup = Instant::now();

        // Clean up expired bans
        let mut expired_bans = Vec::new();
        {
            let banned = self.banned_validators.read().await;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            for (node_id, status) in banned.iter() {
                if now >= status.ban_start + status.ban_duration {
                    expired_bans.push(node_id.clone());
                }
            }
        }

        if !expired_bans.is_empty() {
            let mut banned = self.banned_validators.write().await;
            for node_id in expired_bans.iter() {
                banned.remove(node_id);
                info!("Ban expired for validator: {}", node_id);
            }
        }

        // Clean up old fault entries
        let config = self.config.read().await;
        let window = config.fault_tracking_window_secs;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let cutoff = now.saturating_sub(window);

        let mut validator_faults = self.validator_faults.write().await;
        for faults in validator_faults.values_mut() {
            faults.retain(|(timestamp, _)| *timestamp >= cutoff);
        }

        // Remove entries with no faults
        validator_faults.retain(|_, faults| !faults.is_empty());

        debug!(
            "Security cleanup complete: removed {} expired bans",
            expired_bans.len()
        );
        Ok(())
    }

    /// Report a security incident
    pub async fn report_incident(&self, incident: SecurityIncident) -> Result<()> {
        // Check if we're running
        let is_running = *self.running.read().await;
        if !is_running {
            return Err(anyhow!("Security manager is not running"));
        }

        // Validate the incident
        if incident.node_id.is_empty() {
            return Err(anyhow!("Invalid incident: missing node ID"));
        }

        // Store the incident
        {
            let mut incidents = self.incidents.write().await;
            incidents.insert(incident.id.clone(), incident.clone());
        }

        // Update fault counter for the validator
        {
            let mut validator_faults = self.validator_faults.write().await;
            let faults = validator_faults
                .entry(incident.node_id.clone())
                .or_insert_with(Vec::new);

            faults.push((incident.timestamp, incident.fault_type));

            // Check if we should ban this validator
            let config = self.config.read().await;
            if config.enable_auto_ban && faults.len() >= config.auto_ban_threshold {
                // Ban the validator
                self.ban_validator(
                    incident.node_id.clone(),
                    format!("Exceeded fault threshold with {} faults", faults.len()),
                    config.ban_duration_secs,
                    vec![incident.id.clone()],
                )
                .await?;
            }
        }

        info!(
            "Security incident reported: {} - {:?} by {}",
            incident.id, incident.fault_type, incident.node_id
        );
        Ok(())
    }

    /// Ban a validator
    pub async fn ban_validator(
        &self,
        node_id: NodeId,
        reason: String,
        duration: u64,
        related_incidents: Vec<String>,
    ) -> Result<()> {
        // Check if the node is already banned
        let is_banned = {
            let banned = self.banned_validators.read().await;
            banned.contains_key(&node_id)
        };

        if is_banned {
            return Ok(()); // Already banned
        }

        // Create ban status
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let status = BanStatus {
            node_id: node_id.clone(),
            ban_start: now,
            ban_duration: duration,
            reason: reason.clone(),
            incidents: related_incidents.clone(),
        };

        // Add to banned validators
        {
            let mut banned = self.banned_validators.write().await;
            banned.insert(node_id.clone(), status);
        }

        info!(
            "Validator {} banned for {} seconds: {}",
            node_id, duration, reason
        );
        Ok(())
    }

    /// Check if a validator is banned
    pub async fn is_validator_banned(&self, node_id: &str) -> bool {
        let banned = self.banned_validators.read().await;
        banned
            .get(node_id)
            .map(|status| status.is_active())
            .unwrap_or(false)
    }

    /// Verify a block's security properties
    pub async fn verify_block(&self, block: &Block) -> Result<bool> {
        // Check if we're running
        let is_running = *self.running.read().await;
        if !is_running {
            return Err(anyhow!("Security manager is not running"));
        }

        let config = self.config.read().await;

        // Check if the block proposer is banned
        if let Some(proposer) = &block.proposer {
            if self.is_validator_banned(proposer).await {
                return Err(anyhow!("Block proposed by banned validator: {}", proposer));
            }
        }

        // Verify block timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if let Some(timestamp) = block.timestamp {
            // Block timestamp should not be too far in the future
            if timestamp > now + 5 {
                return Err(anyhow!("Block timestamp is in the future"));
            }

            // Block timestamp should not be too old
            if now > timestamp + 3600 {
                return Err(anyhow!("Block timestamp is too old"));
            }
        }

        // Verify block signature if present
        if config.verify_signatures && !block.signature.is_empty() {
            if !self
                .verify_signature(
                    &block.hash,
                    &block.signature,
                    block.proposer.as_deref().unwrap_or("unknown"),
                )
                .await
                .is_valid
            {
                return Err(anyhow!("Invalid block signature"));
            }
        }

        // Verify transactions if configured
        if config.verify_all_transactions {
            for tx in &block.txs {
                if !self.verify_transaction(tx).await? {
                    return Err(anyhow!("Invalid transaction in block"));
                }
            }
        }

        // Block is valid
        Ok(true)
    }

    /// Verify a transaction's security properties
    pub async fn verify_transaction(&self, tx: &Transaction) -> Result<bool> {
        let config = self.config.read().await;

        // Verify transaction signature if present
        if config.verify_signatures && !tx.signature.is_empty() {
            if !self
                .verify_signature(&tx.hash, &tx.signature, &tx.sender)
                .await
                .is_valid
            {
                return Err(anyhow!("Invalid transaction signature"));
            }
        }

        // Transaction is valid
        Ok(true)
    }

    /// Verify a signature
    pub async fn verify_signature(
        &self,
        data: &[u8],
        signature: &[u8],
        signer: &str,
    ) -> SignatureVerificationResult {
        // In a real implementation, this would verify the signature
        // against the signer's public key
        let start_time = Instant::now();

        // Simple check for example
        let is_valid = !signature.is_empty();

        SignatureVerificationResult {
            is_valid,
            verification_time_ms: start_time.elapsed().as_millis() as u64,
            error: if is_valid {
                None
            } else {
                Some("Invalid signature".to_string())
            },
        }
    }

    /// Detect byzantine validators
    pub async fn detect_byzantine_validators(&self) -> Result<Vec<(NodeId, ByzantineFaultType)>> {
        let config = self.config.read().await;
        let validator_faults = self.validator_faults.read().await;
        let mut result = Vec::new();

        for (node_id, faults) in validator_faults.iter() {
            // Count recent faults
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let window = config.fault_tracking_window_secs;
            let cutoff = now.saturating_sub(window);

            let recent_faults = faults
                .iter()
                .filter(|(timestamp, _)| *timestamp >= cutoff)
                .count();

            // Group by fault type to find most common
            let mut fault_counts = HashMap::new();
            for (_, fault_type) in faults {
                *fault_counts.entry(fault_type).or_insert(0) += 1;
            }

            // If we have enough faults, report it
            if recent_faults > 0 {
                let most_common_fault = fault_counts
                    .iter()
                    .max_by_key(|(_, count)| **count)
                    .map(|(fault, _)| **fault)
                    .unwrap_or(ByzantineFaultType::Unknown);

                result.push((node_id.clone(), most_common_fault));
            }
        }

        Ok(result)
    }

    /// Get statistics about security incidents
    pub async fn get_statistics(&self) -> SecurityStatistics {
        let incidents = self.incidents.read().await;
        let banned = self.banned_validators.read().await;
        let validator_faults = self.validator_faults.read().await;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Count incidents by type
        let mut incidents_by_type = HashMap::new();
        for incident in incidents.values() {
            *incidents_by_type.entry(incident.fault_type).or_insert(0) += 1;
        }

        // Count incidents by severity
        let mut incidents_by_severity = HashMap::new();
        for incident in incidents.values() {
            *incidents_by_severity.entry(incident.severity).or_insert(0) += 1;
        }

        // Calculate time windows
        let hour_ago = now - 3600;
        let day_ago = now - 86400;
        let week_ago = now - 604800;

        let incidents_last_hour = incidents
            .values()
            .filter(|i| i.timestamp >= hour_ago)
            .count();

        let incidents_last_day = incidents
            .values()
            .filter(|i| i.timestamp >= day_ago)
            .count();

        let incidents_last_week = incidents
            .values()
            .filter(|i| i.timestamp >= week_ago)
            .count();

        SecurityStatistics {
            total_incidents: incidents.len(),
            total_banned_validators: banned.len(),
            active_banned_validators: banned.values().filter(|b| b.is_active()).count(),
            incidents_by_type,
            incidents_by_severity,
            incidents_last_hour,
            incidents_last_day,
            incidents_last_week,
            validator_with_most_faults: validator_faults
                .iter()
                .max_by_key(|(_, faults)| faults.len())
                .map(|(node_id, faults)| (node_id.clone(), faults.len())),
        }
    }

    /// Get all security incidents
    pub async fn get_all_incidents(&self) -> Vec<SecurityIncident> {
        let incidents = self.incidents.read().await;
        incidents.values().cloned().collect()
    }

    /// Get incidents for a specific validator
    pub async fn get_validator_incidents(&self, node_id: &str) -> Vec<SecurityIncident> {
        let incidents = self.incidents.read().await;
        incidents
            .values()
            .filter(|i| i.node_id == node_id)
            .cloned()
            .collect()
    }

    /// Get all banned validators
    pub async fn get_banned_validators(&self) -> Vec<BanStatus> {
        let banned = self.banned_validators.read().await;
        banned.values().cloned().collect()
    }

    /// Unban a validator
    pub async fn unban_validator(&self, node_id: &str) -> Result<()> {
        let mut banned = self.banned_validators.write().await;
        if banned.remove(node_id).is_some() {
            info!("Validator {} manually unbanned", node_id);
            Ok(())
        } else {
            Err(anyhow!("Validator {} is not banned", node_id))
        }
    }

    /// Clear all incidents
    pub async fn clear_incidents(&self) -> Result<usize> {
        let mut incidents = self.incidents.write().await;
        let count = incidents.len();
        incidents.clear();
        Ok(count)
    }

    /// Update security configuration
    pub async fn update_config(&self, config: SecurityConfig) -> Result<()> {
        let mut cfg = self.config.write().await;

        // Log if security level changed
        if cfg.security_level != config.security_level {
            info!(
                "Changing security level from {:?} to {:?}",
                cfg.security_level, config.security_level
            );
        }

        *cfg = config;
        Ok(())
    }
}

impl Clone for SecurityManager {
    fn clone(&self) -> Self {
        // This is a partial clone for internal use
        Self {
            config: RwLock::new(self.config.try_read().unwrap_or_default().clone()),
            incidents: RwLock::new(HashMap::new()),
            banned_validators: RwLock::new(HashMap::new()),
            validator_faults: RwLock::new(HashMap::new()),
            running: RwLock::new(false),
            last_cleanup: RwLock::new(Instant::now()),
            validators: self.validators.clone(),
        }
    }
}

/// Statistics about security incidents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStatistics {
    /// Total number of incidents
    pub total_incidents: usize,
    /// Total number of banned validators
    pub total_banned_validators: usize,
    /// Number of currently active banned validators
    pub active_banned_validators: usize,
    /// Incidents by fault type
    pub incidents_by_type: HashMap<ByzantineFaultType, usize>,
    /// Incidents by severity
    pub incidents_by_severity: HashMap<IncidentSeverity, usize>,
    /// Incidents in the last hour
    pub incidents_last_hour: usize,
    /// Incidents in the last day
    pub incidents_last_day: usize,
    /// Incidents in the last week
    pub incidents_last_week: usize,
    /// Validator with most faults
    pub validator_with_most_faults: Option<(NodeId, usize)>,
}
