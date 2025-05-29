use crate::config::Config;
use crate::ledger::state::State;
use crate::ledger::transaction::Transaction;
#[cfg(test)] // Only needed for tests
use crate::ledger::transaction::TransactionType;
use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use ort::environment::{get_environment, Environment};
use ort::session::Session;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
#[cfg(test)] // Only needed for tests
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio::time;

/// Represents a node's health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceHealthMetrics {
    /// CPU usage (0-100%)
    pub cpu_usage: f32,
    /// Memory usage (0-100%)
    pub memory_usage: f32,
    /// Disk space available (bytes)
    pub disk_available: u64,
    /// Number of cores
    pub num_cores: u32,
    /// Uptime in seconds
    pub uptime: u64,
    /// Operating system info
    pub os_info: String,
    /// Average response time (ms)
    pub avg_response_time: f32,
    /// Dropped connections count
    pub dropped_connections: u32,
    /// Hardware temperature (C)
    pub temperature: Option<f32>,
}

/// Represents a node's network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Bandwidth usage (bytes/s)
    pub bandwidth_usage: u64,
    /// Latency (ms)
    pub latency: f32,
    /// Packet loss rate (0-1)
    pub packet_loss: f32,
    /// Connection stability (0-1)
    pub connection_stability: f32,
    /// Peer count
    pub peer_count: u32,
    /// Geographical location consistency (0-1)
    pub geo_consistency: f32,
    /// P2P network score (0-1)
    pub p2p_score: f32,
    /// Sync status (0-1)
    pub sync_status: f32,
}

/// Represents a node's storage contribution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    /// Total storage provided (bytes)
    pub storage_provided: u64,
    /// Storage utilization (0-1)
    pub storage_utilization: f32,
    /// Data retrieval success rate (0-1)
    pub retrieval_success_rate: f32,
    /// Average retrieval time (ms)
    pub avg_retrieval_time: f32,
    /// Data redundancy level
    pub redundancy_level: f32,
    /// Data integrity violations count
    pub integrity_violations: u32,
    /// Storage uptime (0-1)
    pub storage_uptime: f32,
    /// Storage growth rate (bytes/day)
    pub storage_growth_rate: u64,
}

/// Represents a node's engagement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementMetrics {
    /// Participation in validation (0-1)
    pub validation_participation: f32,
    /// Transaction submission frequency
    pub transaction_frequency: f32,
    /// Network participation time (seconds)
    pub participation_time: u64,
    /// Community contribution score (0-1)
    pub community_contribution: f32,
    /// Governance participation (0-1)
    pub governance_participation: f32,
    /// Staking percentage (of total stake)
    pub staking_percentage: f32,
    /// Referral count
    pub referrals: u32,
    /// Social verification strength (0-1)
    pub social_verification: f32,
}

/// Represents a node's AI behavior metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIBehaviorMetrics {
    /// Anomaly detection score (0-1)
    pub anomaly_score: f32,
    /// Risk assessment (0-1)
    pub risk_assessment: f32,
    /// Fraud probability (0-1)
    pub fraud_probability: f32,
    /// Security threat level (0-1)
    pub threat_level: f32,
    /// Behavioral pattern consistency (0-1)
    pub pattern_consistency: f32,
    /// Sybil attack probability (0-1)
    pub sybil_probability: f32,
    /// Historical reliability (0-1)
    pub historical_reliability: f32,
    /// Identity verification strength (0-1)
    pub identity_verification: f32,
}

/// Comprehensive node scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeScore {
    /// Overall score (0-1)
    pub overall_score: f32,
    /// Device health score (0-1)
    pub device_health_score: f32,
    /// Network score (0-1)
    pub network_score: f32,
    /// Storage contribution score (0-1)
    pub storage_score: f32,
    /// Engagement score (0-1)
    pub engagement_score: f32,
    /// AI behavior score (0-1)
    pub ai_behavior_score: f32,
    /// Last update time
    pub last_updated: std::time::SystemTime,
    /// Score history (newest first)
    pub history: Vec<(std::time::SystemTime, f32)>,
}

/// Trust tier levels for reward multipliers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustTier {
    /// Diamond tier (90-100 score): extra rewards
    Diamond,
    /// Standard tier (70-89 score): normal rewards
    Standard,
    /// Limited tier (50-69 score): reduced rewards
    Limited,
    /// Restricted tier (<50 score): minimal rewards
    Restricted,
}

/// Mode for AI execution based on device capabilities
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AIExecutionMode {
    /// Use local ONNX models (full capability)
    Local,
    /// Use TFLite/quantized models (medium capability)
    Distilled,
    /// Use API-based inference (minimal capability)
    Remote,
}

impl NodeScore {
    /// Apply time decay to scores to gradually forgive old behavior
    pub fn apply_decay(&mut self) {
        let now = SystemTime::now();

        // Calculate decay based on days since last update
        if let Ok(duration) = now.duration_since(self.last_updated) {
            let days = duration.as_secs() as f32 / 86400.0;
            let decay_factor = 0.98_f32.powf(days);

            // Apply decay to historical negative impacts only
            if self.overall_score < 0.7 {
                // Calculate recovery amount (more recovery for older scores)
                let recovery = (0.7 - self.overall_score) * (1.0 - decay_factor);

                // Apply recovery with a cap to prevent sudden jumps
                let max_recovery = 0.05; // Max 5% recovery per application
                let applied_recovery = recovery.min(max_recovery);

                self.overall_score += applied_recovery;

                // Adjust individual subscores proportionally
                if self.device_health_score < 0.7 {
                    self.device_health_score += applied_recovery;
                }
                if self.network_score < 0.7 {
                    self.network_score += applied_recovery;
                }
                if self.storage_score < 0.7 {
                    self.storage_score += applied_recovery;
                }
                if self.engagement_score < 0.7 {
                    self.engagement_score += applied_recovery;
                }
                if self.ai_behavior_score < 0.7 {
                    self.ai_behavior_score += applied_recovery;
                }

                debug!(
                    "Applied score decay: +{applied_recovery:.4} recovery after {days} days for node"
                );
            }

            // Cap all scores at 1.0
            self.overall_score = self.overall_score.min(1.0);
            self.device_health_score = self.device_health_score.min(1.0);
            self.network_score = self.network_score.min(1.0);
            self.storage_score = self.storage_score.min(1.0);
            self.engagement_score = self.engagement_score.min(1.0);
            self.ai_behavior_score = self.ai_behavior_score.min(1.0);

            // Update timestamp
            self.last_updated = now;

            // Add new score to history
            self.history.push((now, self.overall_score));

            // Limit history to last 90 days
            self.history.retain(|(timestamp, _)| {
                if let Ok(age) = now.duration_since(*timestamp) {
                    age.as_secs() < 90 * 86400 // 90 days in seconds
                } else {
                    false
                }
            });
        }
    }

    /// Get trust tier based on overall score
    pub fn get_trust_tier(&self) -> TrustTier {
        match self.overall_score {
            s if s >= 0.9 => TrustTier::Diamond,  // 90-100
            s if s >= 0.7 => TrustTier::Standard, // 70-89
            s if s >= 0.5 => TrustTier::Limited,  // 50-69
            _ => TrustTier::Restricted,           // <50
        }
    }

    /// Get reward multiplier based on trust tier
    pub fn get_reward_multiplier(&self) -> f32 {
        match self.get_trust_tier() {
            TrustTier::Diamond => 1.5,    // 50% bonus
            TrustTier::Standard => 1.0,   // standard rewards
            TrustTier::Limited => 0.7,    // 30% penalty
            TrustTier::Restricted => 0.5, // 50% penalty
        }
    }
}

/// History entry for node scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeScoreHistory {
    /// Timestamp of the score
    pub timestamp: std::time::SystemTime,
    /// Score value
    pub score: f32,
}

/// Combined metrics for a node
#[derive(Debug, Clone)]
pub struct NodeMetrics {
    /// Device health metrics
    pub device_health: DeviceHealthMetrics,
    /// Network performance metrics
    pub network: NetworkMetrics,
    /// Storage metrics
    pub storage: StorageMetrics,
    /// Engagement metrics
    pub engagement: EngagementMetrics,
    /// AI behavior metrics
    pub ai_behavior: AIBehaviorMetrics,
}

/// Weights for combining different score components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Weight for device health (0-1)
    pub device_health_weight: f32,
    /// Weight for network performance (0-1)
    pub network_weight: f32,
    /// Weight for storage contribution (0-1)
    pub storage_weight: f32,
    /// Weight for engagement (0-1)
    pub engagement_weight: f32,
    /// Weight for AI behavior (0-1)
    pub ai_behavior_weight: f32,
}

impl ScoringWeights {
    /// Create default scoring weights
    pub fn create_default() -> Self {
        ScoringWeights {
            device_health_weight: 0.2,
            network_weight: 0.2,
            storage_weight: 0.2,
            engagement_weight: 0.2,
            ai_behavior_weight: 0.2,
        }
    }

    /// Create custom scoring weights
    pub fn new(
        device_health_weight: f32,
        network_weight: f32,
        storage_weight: f32,
        engagement_weight: f32,
        ai_behavior_weight: f32,
    ) -> Result<Self> {
        // Validate weights are between 0 and 1
        for weight in [
            device_health_weight,
            network_weight,
            storage_weight,
            engagement_weight,
            ai_behavior_weight,
        ]
        .iter()
        {
            if *weight < 0.0 || *weight > 1.0 {
                return Err(anyhow!("Weights must be between 0 and 1"));
            }
        }

        // Validate weights sum to 1.0
        let sum = device_health_weight
            + network_weight
            + storage_weight
            + engagement_weight
            + ai_behavior_weight;
        if (sum - 1.0).abs() > 0.001 {
            return Err(anyhow!("Weights must sum to 1.0, got {sum}"));
        }

        Ok(Self {
            device_health_weight,
            network_weight,
            storage_weight,
            engagement_weight,
            ai_behavior_weight,
        })
    }
}

/// Represents a node's security AI system
pub struct SecurityAI {
    /// Configuration
    config: Config,
    /// Blockchain state
    state: Arc<RwLock<State>>,
    /// Node trust scores
    node_scores: Arc<tokio::sync::Mutex<HashMap<String, NodeScore>>>,
    /// Transaction risk scores
    transaction_scores: Arc<tokio::sync::Mutex<HashMap<String, f64>>>,
    /// Last model reload time
    last_model_reload: Arc<tokio::sync::Mutex<Instant>>,
    /// ONNX Runtime environment
    ort_environment: Arc<Environment>,
    /// Device health model
    #[allow(dead_code)]
    device_health_model: Option<Arc<Session>>,
    /// Network model
    #[allow(dead_code)]
    network_model: Option<Arc<Session>>,
    /// Storage model
    #[allow(dead_code)]
    storage_model: Option<Arc<Session>>,
    /// Engagement model
    #[allow(dead_code)]
    engagement_model: Option<Arc<Session>>,
    /// AI behavior model
    #[allow(dead_code)]
    ai_behavior_model: Option<Arc<Session>>,
    /// Scoring weights
    scoring_weights: Arc<tokio::sync::RwLock<ScoringWeights>>,
    /// Running flag
    #[allow(dead_code)]
    running: AtomicBool,
    /// AI execution mode (based on device capability)
    execution_mode: AIExecutionMode,
    /// Remote API endpoint for inference fallback
    remote_api_endpoint: Option<String>,
}

impl SecurityAI {
    /// Create a new SecurityAI instance
    pub fn new(config: Config, state: Arc<RwLock<State>>) -> Result<Self> {
        // Initialize ONNX Runtime
        let ort_environment = get_environment()?;

        // Get remote API endpoint from config or environment
        let remote_api_endpoint = std::env::var("SECURITY_AI_REMOTE_ENDPOINT").ok();

        // Default to local execution
        let execution_mode = AIExecutionMode::Local;

        Ok(SecurityAI {
            config,
            state,
            node_scores: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            transaction_scores: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            last_model_reload: Arc::new(tokio::sync::Mutex::new(Instant::now())),
            ort_environment,
            device_health_model: None,
            network_model: None,
            storage_model: None,
            engagement_model: None,
            ai_behavior_model: None,
            scoring_weights: Arc::new(tokio::sync::RwLock::new(ScoringWeights::create_default())),
            running: AtomicBool::new(false),
            execution_mode,
            remote_api_endpoint,
        })
    }

    /// Start the SecurityAI service
    pub async fn start(&mut self) -> Result<JoinHandle<()>> {
        // Load models
        self.reload_models().await?;

        // Clone Arc references for the background task
        let _node_scores = self.node_scores.clone();
        let transaction_scores = self.transaction_scores.clone();
        let last_model_reload = self.last_model_reload.clone();
        let _ort_environment = self.ort_environment.clone();
        let _config = self.config.clone();
        let _scoring_weights = self.scoring_weights.clone();
        let state = self.state.clone();

        // Start background task
        let handle = tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                // Reload models periodically (every 24 hours)
                let last_reload = {
                    let guard = last_model_reload.lock().await;
                    *guard
                };

                if last_reload.elapsed() > Duration::from_secs(24 * 60 * 60) {
                    info!("Reloading AI security models");
                    // In a real implementation, this would reload the models
                    // For now, just update the timestamp
                    let mut guard = last_model_reload.lock().await;
                    *guard = Instant::now();
                }

                // Periodically clean up old scores
                let _now = std::time::SystemTime::now();

                // Clean up transaction scores older than 1 hour
                {
                    let mut scores = transaction_scores.lock().await;
                    scores.retain(|_, _| {
                        // In a real implementation, we would check the timestamp
                        // For the stub, just keep all scores
                        true
                    });
                }

                // Update node scores
                {
                    let _state_guard = state.read().await;
                    // In a real implementation, iterate through known nodes and update scores
                }
            }
        });

        Ok(handle)
    }

    /// Reload AI models from disk
    pub async fn reload_models(&mut self) -> Result<()> {
        // Get model directory from config
        let model_dir = self.config.ai_model_dir.clone();

        // Check execution mode
        match self.execution_mode {
            AIExecutionMode::Remote => {
                // No models to load for remote mode
                info!("Remote inference mode active - no local models to load");
                return Ok(());
            }
            AIExecutionMode::Distilled => {
                info!("Loading distilled lightweight models");
                // For distilled models, we use a specific subfolder
                let distilled_dir = model_dir.join("distilled");
                if !distilled_dir.exists() {
                    warn!("Distilled models directory not found: {distilled_dir:?}");
                    std::fs::create_dir_all(&distilled_dir)?;
                }

                return self.load_models_from_dir(&distilled_dir).await;
            }
            AIExecutionMode::Local => {
                // Continue with normal model loading
            }
        }

        // Check for versioned models
        let requested_version = std::env::var("AI_MODEL_VERSION").unwrap_or_else(|_| {
            // Default to latest by default
            "latest".to_string()
        });

        let model_base_dir = if requested_version == "latest" {
            // Find the latest version directory
            let mut versions = Vec::new();
            if model_dir.exists() {
                for entry in std::fs::read_dir(&model_dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.is_dir() {
                        if let Some(dir_name) = path.file_name() {
                            if let Some(dir_name_str) = dir_name.to_str() {
                                if dir_name_str.starts_with("v") {
                                    versions.push(dir_name_str.to_string());
                                }
                            }
                        }
                    }
                }
            }

            // Sort versions and get the latest
            if !versions.is_empty() {
                versions.sort();
                let latest_version = versions.last().unwrap().clone();
                info!("Using latest model version: {latest_version}");
                model_dir.join(latest_version)
            } else {
                // No versioned directories found, use base directory
                info!("No versioned models found, using base directory");
                model_dir.clone()
            }
        } else {
            // Use the specific requested version
            let version_dir = model_dir.join(&requested_version);
            if !version_dir.exists() {
                warn!(
                    "Requested model version {requested_version} not found, falling back to base directory"
                );
                model_dir.clone()
            } else {
                info!("Using requested model version: {requested_version}");
                version_dir
            }
        };

        // Make sure the model directory exists
        if !model_base_dir.exists() {
            std::fs::create_dir_all(&model_base_dir)?;

            // Log a warning that we're creating the model directory
            warn!("Model directory does not exist, creating: {model_base_dir:?}");

            // No models to load yet, so return early
            return Ok(());
        }

        info!("Loading AI models from {model_base_dir:?}");

        // Call helper to load models from the selected directory
        self.load_models_from_dir(&model_base_dir).await
    }

    /// Helper to load models from a specific directory
    async fn load_models_from_dir(&mut self, dir_path: &std::path::Path) -> Result<()> {
        // Log that we're loading models
        info!("Loading AI models from {dir_path:?}");

        // For now, we'll set the models to None and use fallback calculations
        // In a real implementation, we would load ONNX models here
        self.device_health_model = None;
        self.network_model = None;
        self.storage_model = None;
        self.engagement_model = None;
        self.ai_behavior_model = None;

        // Update last reload time
        let mut last_reload = self.last_model_reload.lock().await;
        *last_reload = Instant::now();

        Ok(())
    }

    /// Evaluate a transaction for security risks
    pub async fn evaluate_transaction(&self, transaction: &Transaction) -> Result<f64> {
        // Check if we already have a score for this transaction
        let tx_hash = transaction.hash().to_string();

        let mut scores = self.transaction_scores.lock().await;

        if let Some(score) = scores.get(&tx_hash) {
            return Ok(*score);
        }

        // Otherwise, calculate a new score
        // In a real implementation, this would use the AI model
        // For now, we'll use a simple heuristic

        let mut rng = thread_rng();
        let base_score = 0.8 + (rng.gen::<f64>() * 0.2); // Score between 0.8 and 1.0

        // Add the score to our cache
        scores.insert(tx_hash, base_score);

        Ok(base_score)
    }

    /// Evaluate a node's security using available metrics
    pub async fn evaluate_node(&self, node_id: &str, metrics: &NodeMetrics) -> Result<NodeScore> {
        let device_health_score = self.calculate_device_health_score(&metrics.device_health);
        let network_score = self.calculate_network_score(&metrics.network);
        let storage_score = self.calculate_storage_score(&metrics.storage);
        let engagement_score = self.calculate_engagement_score(&metrics.engagement);
        let ai_behavior_score = self.calculate_ai_behavior_score(&metrics.ai_behavior);

        // Apply weights to calculate overall score
        let weights = self.scoring_weights.read().await;
        let overall_score = device_health_score * weights.device_health_weight
            + network_score * weights.network_weight
            + storage_score * weights.storage_weight
            + engagement_score * weights.engagement_weight
            + ai_behavior_score * weights.ai_behavior_weight;

        let now = SystemTime::now();

        // Get any existing score data to preserve history
        let mut history = Vec::new();
        {
            let scores = self.node_scores.lock().await;
            if let Some(existing_score) = scores.get(node_id) {
                history = existing_score.history.clone();
            }
        }

        // Add current score to history
        history.push((now, overall_score));

        // Limit history size to last 30 days
        if history.len() > 30 {
            history.sort_by(|a, b| a.0.cmp(&b.0));
            let len = history.len();
            history = history.into_iter().skip(len - 30).collect();
        }

        let node_score = NodeScore {
            overall_score,
            device_health_score,
            network_score,
            storage_score,
            engagement_score,
            ai_behavior_score,
            last_updated: now,
            history,
        };

        // Store the score
        {
            let mut scores = self.node_scores.lock().await;
            scores.insert(node_id.to_string(), node_score.clone());
        }

        Ok(node_score)
    }

    /// Get the score for a particular node
    pub async fn get_node_score(&self, node_id: &str) -> Option<NodeScore> {
        let scores = self.node_scores.lock().await;
        scores.get(node_id).cloned()
    }

    /// Get all node scores
    pub async fn get_all_node_scores(&self) -> HashMap<String, NodeScore> {
        let scores = self.node_scores.lock().await;
        scores.clone()
    }

    /// Calculate device health score from metrics
    fn calculate_device_health_score(&self, metrics: &DeviceHealthMetrics) -> f32 {
        // Skip ONNX model-based scoring for now and use the fallback calculation
        self.calculate_device_health_score_fallback(metrics)
    }

    /// Fallback calculation for device health score
    fn calculate_device_health_score_fallback(&self, metrics: &DeviceHealthMetrics) -> f32 {
        // Normalize CPU usage (0-100% -> 0-1 score, inverted)
        let cpu_score = 1.0 - (metrics.cpu_usage / 100.0);

        // Normalize memory usage (0-100% -> 0-1 score, inverted)
        let memory_score = 1.0 - (metrics.memory_usage / 100.0);

        // Normalize disk space (simple heuristic, > 10GB = 1.0, < 100MB = 0.0)
        let disk_score = if metrics.disk_available > 10_000_000_000 {
            1.0
        } else if metrics.disk_available < 100_000_000 {
            0.0
        } else {
            // Linear interpolation between 100MB and 10GB
            (metrics.disk_available as f32 - 100_000_000.0) / (10_000_000_000.0 - 100_000_000.0)
        };

        // Normalize response time (< 50ms = 1.0, > 500ms = 0.0)
        let response_score = if metrics.avg_response_time < 50.0 {
            1.0
        } else if metrics.avg_response_time > 500.0 {
            0.0
        } else {
            // Linear interpolation between 50ms and 500ms
            1.0 - ((metrics.avg_response_time - 50.0) / 450.0)
        };

        // Normalize dropped connections (0 = 1.0, >10 = 0.0)
        let connection_score = if metrics.dropped_connections == 0 {
            1.0
        } else if metrics.dropped_connections > 10 {
            0.0
        } else {
            // Linear interpolation
            1.0 - (metrics.dropped_connections as f32 / 10.0)
        };

        // Calculate weighted average
        // This is a simplistic example - in a real system these weights would be tuned
        let weighted_sum = 0.2 * cpu_score
            + 0.2 * memory_score
            + 0.2 * disk_score
            + 0.3 * response_score
            + 0.1 * connection_score;

        // Ensure score is between 0 and 1
        weighted_sum.clamp(0.0, 1.0)
    }

    /// Calculate score based on network metrics
    fn calculate_network_score(&self, metrics: &NetworkMetrics) -> f32 {
        let mut score = 1.0f32;

        // Latency (lower is better)
        if metrics.latency > 500.0 {
            score -= 0.5;
        } else if metrics.latency > 200.0 {
            score -= 0.3;
        } else if metrics.latency > 100.0 {
            score -= 0.1;
        }

        // Bandwidth (higher is better)
        if metrics.bandwidth_usage < 1024 * 1024 {
            // Less than 1MB/s
            score -= 0.3;
        } else if metrics.bandwidth_usage < 5 * 1024 * 1024 {
            // Less than 5MB/s
            score -= 0.1;
        }

        // Packet loss (lower is better)
        if metrics.packet_loss > 0.05 {
            score -= 0.4;
        } else if metrics.packet_loss > 0.01 {
            score -= 0.2;
        }

        // Connection count (higher is better, up to a point)
        if metrics.peer_count < 5 {
            score -= 0.2;
        } else if metrics.peer_count > 50 {
            // Too many connections might indicate a DoS attempt
            score -= 0.1;
        }

        // Ensure score is between 0 and 1
        score.clamp(0.0f32, 1.0f32)
    }

    /// Calculate score based on storage metrics
    fn calculate_storage_score(&self, metrics: &StorageMetrics) -> f32 {
        let mut score = 0.5f32; // Base score

        // Storage provided (more is better)
        if metrics.storage_provided > 100 * 1024 * 1024 * 1024 {
            // More than 100GB
            score += 0.3;
        } else if metrics.storage_provided > 10 * 1024 * 1024 * 1024 {
            // More than 10GB
            score += 0.2;
        } else if metrics.storage_provided > 1024 * 1024 * 1024 {
            // More than 1GB
            score += 0.1;
        }

        // Availability (higher is better)
        score += metrics.storage_utilization * 0.4;

        // Read/write speed (higher is better)
        if metrics.retrieval_success_rate > 0.9 {
            score += 0.2;
        } else if metrics.retrieval_success_rate > 0.5 {
            score += 0.1;
        }

        // Ensure score is between 0 and 1
        score.clamp(0.0f32, 1.0f32)
    }

    /// Calculate score based on engagement metrics
    fn calculate_engagement_score(&self, metrics: &EngagementMetrics) -> f32 {
        let mut score = 0.6f32; // Base score

        // Transactions relayed (more is better, up to a point)
        if metrics.transaction_frequency > 1000.0 {
            score += 0.1;
        }

        // Blocks proposed (more is better)
        if metrics.participation_time > 86400 {
            score += 0.1;
        }

        // Validation participation (higher is better)
        score += metrics.validation_participation * 0.2;

        // Online percentage (higher is better)
        score += 0.2;

        // Ensure score is between 0 and 1
        score.clamp(0.0f32, 1.0f32)
    }

    /// Calculate score based on AI behavior metrics
    fn calculate_ai_behavior_score(&self, metrics: &AIBehaviorMetrics) -> f32 {
        let mut score = 1.0f32;

        // Anomaly detection (lower is better)
        score -= metrics.anomaly_score * 0.4;

        // Policy compliance (higher is better)
        score += metrics.risk_assessment * 0.3;

        // Suspicious pattern detection (lower is better)
        score -= metrics.pattern_consistency * 0.3;

        // Ensure score is between 0 and 1
        score.clamp(0.0f32, 1.0f32)
    }

    /// Remove a node's scoring data
    pub async fn remove_node_score(&self, node_id: &str) -> bool {
        let mut scores = self.node_scores.lock().await;
        scores.remove(node_id).is_some()
    }

    /// Clear all node scores
    pub async fn clear_all_node_scores(&self) {
        let mut scores = self.node_scores.lock().await;
        scores.clear();
    }

    /// Get the scoring weights
    pub async fn get_scoring_weights(&self) -> ScoringWeights {
        self.scoring_weights.read().await.clone()
    }

    /// Update the scoring weights
    pub async fn update_scoring_weights(&self, new_weights: ScoringWeights) -> Result<()> {
        let mut weights = self.scoring_weights.write().await;
        *weights = new_weights;
        Ok(())
    }

    /// Detect device capabilities and set appropriate execution mode
    pub fn detect_capabilities(&mut self) -> Result<()> {
        // Get system information
        let available_memory = self.get_available_memory_mb()?;
        let cpu_cores = self.get_cpu_cores()?;

        // Choose execution mode based on device capabilities
        self.execution_mode = if available_memory < 512 || cpu_cores < 2 {
            info!(
                "Low-resource device detected: {available_memory}MB RAM, {cpu_cores} cores. Using remote inference mode."
            );

            // Check if remote endpoint is configured
            if self.remote_api_endpoint.is_none() {
                warn!("Remote inference mode selected but no endpoint configured. Falling back to distilled mode.");
                AIExecutionMode::Distilled
            } else {
                AIExecutionMode::Remote
            }
        } else if available_memory < 2048 || cpu_cores < 4 {
            info!(
                "Medium-resource device detected: {available_memory}MB RAM, {cpu_cores} cores. Using distilled model mode."
            );
            AIExecutionMode::Distilled
        } else {
            info!(
                "High-resource device detected: {available_memory}MB RAM, {cpu_cores} cores. Using full local inference."
            );
            AIExecutionMode::Local
        };

        Ok(())
    }

    /// Get available system memory in MB
    fn get_available_memory_mb(&self) -> Result<u64> {
        // For simplicity, return a reasonable default for all platforms
        Ok(4096)
    }

    /// Get number of CPU cores
    fn get_cpu_cores(&self) -> Result<u32> {
        // Use num_cpus crate to detect CPU cores
        Ok(num_cpus::get() as u32)
    }

    /// Perform remote inference when local execution isn't possible
    pub async fn remote_inference(
        &self,
        model_type: &str,
        _input_data: Vec<u8>,
    ) -> Result<Vec<f32>> {
        // Ensure we have a remote endpoint
        let endpoint = self
            .remote_api_endpoint
            .as_ref()
            .ok_or_else(|| anyhow!("Remote inference requested but no endpoint configured"))?;

        info!("Performing remote inference for {model_type} model using endpoint {endpoint}");

        // In a real implementation, this would call an API
        // For now, just return some reasonable defaults
        match model_type {
            "device_health" => Ok(vec![0.8, 0.7, 0.9]),
            "network" => Ok(vec![0.75, 0.8, 0.7]),
            "storage" => Ok(vec![0.85, 0.9, 0.8]),
            "engagement" => Ok(vec![0.7, 0.65, 0.75]),
            "ai_behavior" => Ok(vec![0.9, 0.85, 0.8]),
            _ => Err(anyhow!(
                "Unknown model type for remote inference: {model_type}"
            )),
        }
    }

    /// Start monitoring with a specified interval
    pub async fn start_monitoring(&mut self, interval: Duration) -> Result<()> {
        let state = self.state.clone();
        let node_scores = self.node_scores.clone();
        let transaction_scores = self.transaction_scores.clone();

        tokio::spawn(async move {
            let mut interval = time::interval(interval);
            loop {
                interval.tick().await;

                if let Err(e) =
                    update_security_scores(&state, &node_scores, &transaction_scores).await
                {
                    warn!("Failed to update security scores: {e}");
                }
            }
        });

        Ok(())
    }
}

/// Update security scores for nodes and transactions
#[allow(dead_code)]
async fn update_security_scores(
    _state: &Arc<RwLock<State>>,
    node_scores: &Arc<tokio::sync::Mutex<HashMap<String, NodeScore>>>,
    transaction_scores: &Arc<tokio::sync::Mutex<HashMap<String, f64>>>,
) -> Result<()> {
    // In a real implementation, this would:
    // 1. Use the AI model to evaluate patterns in recent blockchain activity
    // 2. Update scores for nodes and transactions based on the evaluation
    // 3. Possibly take proactive measures for high-risk entities

    // For this example, we'll just use random fluctuations

    // Update node scores
    {
        let mut scores = node_scores.lock().await;

        for (_node_id, score) in scores.iter_mut() {
            // Add a small random adjustment
            let mut rng = thread_rng();
            let adjustment = (rng.gen::<f32>() - 0.5) * 0.05;
            score.overall_score = (score.overall_score + adjustment).clamp(0.0, 1.0);
        }
    }

    // Update transaction scores
    {
        let mut scores = transaction_scores.lock().await;

        // Remove old transactions to keep the cache manageable
        if scores.len() > 10000 {
            let keys: Vec<String> = scores.keys().take(5000).cloned().collect();
            for key in keys {
                scores.remove(&key);
            }
        }
    }

    debug!("Updated security scores");
    Ok(())
}

/// Reload AI models
#[allow(dead_code)]
async fn reload_ai_models(
    _model_dir: &Path,
    last_reload: &Arc<tokio::sync::Mutex<Instant>>,
) -> Result<()> {
    // In a real implementation, this would:
    // 1. Check if new models are available
    // 2. Load the new models into memory
    // 3. Replace the old models with the new ones

    // For this example, we'll just update the last reload time
    let mut last_reload = last_reload.lock().await;
    *last_reload = Instant::now();

    info!("Reloaded AI models");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    // Mock implementation of Config and State for tests
    fn create_test_config() -> Config {
        // Let's create a minimal config to avoid type errors
        let mut config = Config::new();
        config.data_dir = PathBuf::from("/tmp/models");
        config.ai_model_dir = PathBuf::from("/tmp/models");
        config
    }

    fn create_test_state() -> Arc<RwLock<State>> {
        // This function would create a test state
        let config = create_test_config();
        Arc::new(RwLock::new(State::new(&config).unwrap()))
    }

    #[tokio::test]
    async fn test_security_ai_creation() {
        let config = create_test_config();
        let state = create_test_state();
        let security = SecurityAI::new(config, state).unwrap();
        assert!(!security.running.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_scoring_weights() {
        let config = create_test_config();
        let state = create_test_state();
        let security = SecurityAI::new(config, state).unwrap();

        // Test default weights
        let default_weights = security.get_scoring_weights().await;
        assert_eq!(default_weights.device_health_weight, 0.2);
        assert_eq!(default_weights.network_weight, 0.2);
        assert_eq!(default_weights.storage_weight, 0.2);
        assert_eq!(default_weights.engagement_weight, 0.2);
        assert_eq!(default_weights.ai_behavior_weight, 0.2);

        // Test updating weights
        let new_weights = ScoringWeights {
            device_health_weight: 0.3,
            network_weight: 0.3,
            storage_weight: 0.2,
            engagement_weight: 0.1,
            ai_behavior_weight: 0.1,
        };

        security
            .update_scoring_weights(new_weights.clone())
            .await
            .unwrap();
        let updated_weights = security.get_scoring_weights().await;
        assert_eq!(updated_weights.device_health_weight, 0.3);
        assert_eq!(updated_weights.network_weight, 0.3);
        assert_eq!(updated_weights.storage_weight, 0.2);
        assert_eq!(updated_weights.engagement_weight, 0.1);
        assert_eq!(updated_weights.ai_behavior_weight, 0.1);
    }

    #[tokio::test]
    async fn test_transaction_evaluation() {
        let config = create_test_config();
        let state = create_test_state();
        let security = SecurityAI::new(config, state).unwrap();

        // Create a test transaction
        let transaction = Transaction::new(
            TransactionType::Transfer,
            "sender_addr".to_string(),
            "recipient_addr".to_string(),
            1000,   // amount
            1,      // nonce
            10,     // gas_price
            100,    // gas_limit
            vec![], // data
        );

        let result = security.evaluate_transaction(&transaction).await.unwrap();
        assert!((0.0..=1.0).contains(&result));
    }

    #[tokio::test]
    async fn test_node_evaluation() {
        let config = create_test_config();
        let state = create_test_state();
        let security = SecurityAI::new(config, state).unwrap();

        // Create test metrics
        let node_metrics = NodeMetrics {
            device_health: DeviceHealthMetrics {
                cpu_usage: 65.0,
                memory_usage: 70.0,
                disk_available: 1024 * 1024 * 1024 * 100, // 100 GB
                num_cores: 8,
                uptime: 86400,
                os_info: "Linux".to_string(),
                avg_response_time: 50.0,
                dropped_connections: 0,
                temperature: Some(45.0),
            },
            network: NetworkMetrics {
                bandwidth_usage: 10 * 1024 * 1024, // 10 MB/s
                latency: 120.0,
                packet_loss: 0.005,
                connection_stability: 0.98,
                peer_count: 20,
                geo_consistency: 0.9,
                p2p_score: 0.95,
                sync_status: 1.0,
            },
            storage: StorageMetrics {
                storage_provided: 50 * 1024 * 1024 * 1024, // 50 GB
                storage_utilization: 0.99,
                retrieval_success_rate: 0.98,
                avg_retrieval_time: 100.0,
                redundancy_level: 3.0,
                integrity_violations: 0,
                storage_uptime: 0.99,
                storage_growth_rate: 1024 * 1024, // 1 MB/day
            },
            engagement: EngagementMetrics {
                validation_participation: 0.98,
                transaction_frequency: 5000.0,
                participation_time: 86400 * 30, // 30 days
                community_contribution: 0.85,
                governance_participation: 0.75,
                staking_percentage: 0.05,
                referrals: 10,
                social_verification: 0.9,
            },
            ai_behavior: AIBehaviorMetrics {
                anomaly_score: 0.05,
                risk_assessment: 0.95,
                fraud_probability: 0.01,
                threat_level: 0.02,
                pattern_consistency: 0.01,
                sybil_probability: 0.01,
                historical_reliability: 0.97,
                identity_verification: 0.95,
            },
        };

        let node_id = "test_node_1";
        let result = security
            .evaluate_node(node_id, &node_metrics)
            .await
            .unwrap();

        // Check that scores are in valid range
        assert!((0.0..=1.0).contains(&result.overall_score));
        assert!((0.0..=1.0).contains(&result.device_health_score));
        assert!((0.0..=1.0).contains(&result.network_score));
        assert!((0.0..=1.0).contains(&result.storage_score));
        assert!((0.0..=1.0).contains(&result.engagement_score));
        assert!((0.0..=1.0).contains(&result.ai_behavior_score));

        // Test score retrieval
        let retrieved_score = security.get_node_score(node_id).await;
        assert!(retrieved_score.is_some());
        assert_eq!(retrieved_score.unwrap().overall_score, result.overall_score);

        // Test all scores retrieval
        let all_scores = security.get_all_node_scores().await;
        assert_eq!(all_scores.len(), 1);
        assert!(all_scores.contains_key(node_id));

        // Test score removal
        let removed = security.remove_node_score(node_id).await;
        assert!(removed);

        // Check it's gone
        let retrieved_after_remove = security.get_node_score(node_id).await;
        assert!(retrieved_after_remove.is_none());
    }
}
