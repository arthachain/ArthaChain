use crate::ledger::block::Block;
use crate::ledger::transaction::Transaction;
use crate::network::types::NodeId;
use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Enable anomaly detection
    pub enabled: bool,
    /// Window size for time series analysis
    pub window_size: usize,
    /// Alerting threshold (standard deviations)
    pub alert_threshold: f64,
    /// Analysis interval in milliseconds
    pub analysis_interval_ms: u64,
    /// Minimum data points required for analysis
    pub min_data_points: usize,
    /// Enable machine learning detection
    pub enable_ml_detection: bool,
    /// Confidence threshold for ML-based detection
    pub ml_confidence_threshold: f64,
    /// Maximum anomalies to track
    pub max_anomalies: usize,
    /// Auto-recover from anomalies
    pub auto_recover: bool,
    /// Recovery timeout in milliseconds
    pub recovery_timeout_ms: u64,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_size: 100,
            alert_threshold: 3.0,
            analysis_interval_ms: 10000, // 10 seconds
            min_data_points: 20,
            enable_ml_detection: true,
            ml_confidence_threshold: 0.75,
            max_anomalies: 1000,
            auto_recover: true,
            recovery_timeout_ms: 60000, // 1 minute
        }
    }
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Unusual transaction volume
    TransactionVolume,
    /// Unusual gas/fee price
    FeeAnomaly,
    /// Block time deviation
    BlockTimeDeviation,
    /// Unusual block size
    BlockSizeAnomaly,
    /// Network latency spike
    NetworkLatencySpike,
    /// Validator behavior anomaly
    ValidatorBehavior,
    /// Mempool congestion
    MempoolCongestion,
    /// Chain reorganization
    ChainReorg,
    /// State inconsistency
    StateInconsistency,
    /// Smart contract behavior
    SmartContractBehavior,
    /// Consensus participation drop
    ConsensusDrop,
    /// Unusual transaction pattern
    TransactionPattern,
}

/// Detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// Time when the anomaly was detected
    pub detection_time: u64,
    /// Anomaly score (higher means more severe)
    pub score: f64,
    /// Current value that triggered the anomaly
    pub current_value: f64,
    /// Expected value range
    pub expected_range: (f64, f64),
    /// Related entities (blocks, transactions, nodes)
    pub related_entities: AnomalyEntities,
    /// Status of the anomaly
    pub status: AnomalyStatus,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Recovery time if resolved
    pub recovery_time: Option<u64>,
}

/// Entities related to an anomaly
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnomalyEntities {
    /// Related block hashes
    pub block_hashes: Vec<Vec<u8>>,
    /// Related transaction hashes
    pub tx_hashes: Vec<Vec<u8>>,
    /// Related node IDs
    pub node_ids: Vec<NodeId>,
}

/// Status of an anomaly
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyStatus {
    /// Newly detected
    New,
    /// Acknowledged by the system
    Acknowledged,
    /// Being investigated
    Investigating,
    /// Resolved
    Resolved,
    /// False positive
    FalsePositive,
    /// Ignored
    Ignored,
}

/// The anomaly detector
pub struct AnomalyDetector {
    /// Configuration
    config: RwLock<AnomalyDetectionConfig>,
    /// Time series data for different metrics
    time_series: RwLock<HashMap<String, VecDeque<f64>>>,
    /// Detected anomalies
    anomalies: RwLock<VecDeque<Anomaly>>,
    /// Running status
    running: RwLock<bool>,
    /// ML model for anomaly detection
    #[cfg(feature = "ml_detection")]
    ml_model: Option<Arc<dyn AnomalyModel>>,
    /// Last analysis time
    last_analysis: RwLock<Instant>,
    /// Known baseline metrics
    baseline_metrics: RwLock<HashMap<String, (f64, f64)>>,
}

/// Trait for ML-based anomaly detection models
#[cfg(feature = "ml_detection")]
pub trait AnomalyModel: Send + Sync {
    /// Detect anomalies in the provided features
    fn detect(&self, features: Vec<f64>) -> Result<(bool, f64, Option<AnomalyType>)>;
    /// Update the model with new data
    fn update(&mut self, features: Vec<f64>, is_anomaly: bool) -> Result<()>;
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(config: AnomalyDetectionConfig) -> Self {
        Self {
            config: RwLock::new(config),
            time_series: RwLock::new(HashMap::new()),
            anomalies: RwLock::new(VecDeque::new()),
            running: RwLock::new(false),
            #[cfg(feature = "ml_detection")]
            ml_model: None,
            last_analysis: RwLock::new(Instant::now()),
            baseline_metrics: RwLock::new(HashMap::new()),
        }
    }

    /// Start the anomaly detector
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(anyhow!("Anomaly detector already running"));
        }

        // Initialize ML model if enabled
        #[cfg(feature = "ml_detection")]
        if self.config.read().await.enable_ml_detection {
            // In a real implementation, this would load or create a ML model
            // self.ml_model = Some(Arc::new(MyAnomalyModel::new()?));
        }

        *running = true;

        // Start periodic analysis task
        self.start_analysis_task();

        info!("Anomaly detector started");
        Ok(())
    }

    /// Stop the anomaly detector
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(anyhow!("Anomaly detector not running"));
        }

        *running = false;
        info!("Anomaly detector stopped");
        Ok(())
    }

    /// Start the periodic analysis task
    fn start_analysis_task(&self) {
        let self_clone = Arc::new(self.clone());

        tokio::spawn(async move {
            let interval_ms = {
                let config = self_clone.config.read().await;
                config.analysis_interval_ms
            };

            let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));

            loop {
                interval.tick().await;

                let is_running = {
                    let running = self_clone.running.read().await;
                    *running
                };

                if !is_running {
                    break;
                }

                if let Err(e) = self_clone.run_periodic_analysis().await {
                    warn!("Error in anomaly analysis: {}", e);
                }

                // Check for auto-recovery
                if let Err(e) = self_clone.check_for_recovery().await {
                    warn!("Error checking for anomaly recovery: {}", e);
                }
            }
        });
    }

    /// Run periodic analysis of metrics
    async fn run_periodic_analysis(&self) -> Result<()> {
        let config = self.config.read().await;
        if !config.enabled {
            return Ok(());
        }

        *self.last_analysis.write().await = Instant::now();

        // Analyze all time series data
        let time_series = self.time_series.read().await;
        let baseline_metrics = self.baseline_metrics.read().await;

        let mut new_anomalies = Vec::new();

        for (metric_name, values) in time_series.iter() {
            if values.len() < config.min_data_points {
                continue; // Not enough data
            }

            // Calculate statistics
            let mean = calculate_mean(values);
            let std_dev = calculate_std_dev(values, mean);

            // Get the latest value
            if let Some(current_value) = values.back() {
                let z_score = (current_value - mean) / std_dev;

                // Check if value exceeds threshold
                if z_score.abs() > config.alert_threshold {
                    // This is an anomaly
                    let expected_range = (
                        mean - config.alert_threshold * std_dev,
                        mean + config.alert_threshold * std_dev,
                    );

                    // Determine anomaly type
                    let anomaly_type = match metric_name.as_str() {
                        "tx_volume" => AnomalyType::TransactionVolume,
                        "block_time" => AnomalyType::BlockTimeDeviation,
                        "block_size" => AnomalyType::BlockSizeAnomaly,
                        "fee_rate" => AnomalyType::FeeAnomaly,
                        "network_latency" => AnomalyType::NetworkLatencySpike,
                        "mempool_size" => AnomalyType::MempoolCongestion,
                        "consensus_participation" => AnomalyType::ConsensusDrop,
                        _ => AnomalyType::TransactionPattern,
                    };

                    // Create anomaly record
                    let anomaly = Anomaly {
                        anomaly_type,
                        detection_time: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        score: z_score.abs(),
                        current_value: *current_value,
                        expected_range,
                        related_entities: AnomalyEntities::default(),
                        status: AnomalyStatus::New,
                        metadata: HashMap::new(),
                        recovery_time: None,
                    };

                    new_anomalies.push(anomaly);
                }
            }

            // Update baseline if needed
            if !baseline_metrics.contains_key(metric_name) {
                let mut baseline = self.baseline_metrics.write().await;
                baseline.insert(metric_name.clone(), (mean, std_dev));
            }
        }

        // Record new anomalies
        if !new_anomalies.is_empty() {
            let mut anomalies = self.anomalies.write().await;

            for anomaly in new_anomalies {
                info!(
                    "Detected {} anomaly: value={}, score={:.2}, expected=({:.2}, {:.2})",
                    anomaly.anomaly_type,
                    anomaly.current_value,
                    anomaly.score,
                    anomaly.expected_range.0,
                    anomaly.expected_range.1
                );

                anomalies.push_back(anomaly);

                // Limit the number of stored anomalies
                while anomalies.len() > config.max_anomalies {
                    anomalies.pop_front();
                }
            }
        }

        Ok(())
    }

    /// Check for automatic recovery of anomalies
    async fn check_for_recovery(&self) -> Result<()> {
        let config = self.config.read().await;
        if !config.enabled || !config.auto_recover {
            return Ok(());
        }

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut anomalies = self.anomalies.write().await;
        let recovery_timeout = config.recovery_timeout_ms / 1000; // Convert to seconds

        for anomaly in anomalies.iter_mut() {
            if anomaly.status == AnomalyStatus::New || anomaly.status == AnomalyStatus::Acknowledged
            {
                // Check if the anomaly has persisted for too long
                if current_time - anomaly.detection_time > recovery_timeout {
                    // Auto-recover
                    anomaly.status = AnomalyStatus::Resolved;
                    anomaly.recovery_time = Some(current_time);

                    info!(
                        "Auto-recovered from {} anomaly after {} seconds",
                        anomaly.anomaly_type,
                        current_time - anomaly.detection_time
                    );
                }
            }
        }

        Ok(())
    }

    /// Record a new metric value
    pub async fn record_metric(&self, name: &str, value: f64) -> Result<Option<Anomaly>> {
        let config = self.config.read().await;
        if !config.enabled {
            return Ok(None);
        }

        let mut time_series = self.time_series.write().await;
        let series = time_series
            .entry(name.to_string())
            .or_insert_with(VecDeque::new);

        // Add the new value
        series.push_back(value);

        // Limit series length
        while series.len() > config.window_size {
            series.pop_front();
        }

        // Check for anomalies if we have enough data
        if series.len() >= config.min_data_points {
            let mean = calculate_mean(series);
            let std_dev = calculate_std_dev(series, mean);

            let z_score = (value - mean) / std_dev;

            if z_score.abs() > config.alert_threshold {
                // This is an anomaly
                let expected_range = (
                    mean - config.alert_threshold * std_dev,
                    mean + config.alert_threshold * std_dev,
                );

                // Determine anomaly type
                let anomaly_type = match name {
                    "tx_volume" => AnomalyType::TransactionVolume,
                    "block_time" => AnomalyType::BlockTimeDeviation,
                    "block_size" => AnomalyType::BlockSizeAnomaly,
                    "fee_rate" => AnomalyType::FeeAnomaly,
                    "network_latency" => AnomalyType::NetworkLatencySpike,
                    "mempool_size" => AnomalyType::MempoolCongestion,
                    "consensus_participation" => AnomalyType::ConsensusDrop,
                    _ => AnomalyType::TransactionPattern,
                };

                // Create anomaly record
                let anomaly = Anomaly {
                    anomaly_type,
                    detection_time: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    score: z_score.abs(),
                    current_value: value,
                    expected_range,
                    related_entities: AnomalyEntities::default(),
                    status: AnomalyStatus::New,
                    metadata: HashMap::new(),
                    recovery_time: None,
                };

                // Record the anomaly
                let mut anomalies = self.anomalies.write().await;
                anomalies.push_back(anomaly.clone());

                // Limit the number of stored anomalies
                while anomalies.len() > config.max_anomalies {
                    anomalies.pop_front();
                }

                return Ok(Some(anomaly));
            }
        }

        Ok(None)
    }

    /// Process a block for anomaly detection
    pub async fn process_block(&self, block: &Block) -> Result<Vec<Anomaly>> {
        let config = self.config.read().await;
        if !config.enabled {
            return Ok(Vec::new());
        }

        let mut detected_anomalies = Vec::new();

        // Record block metrics
        let tx_count = block.txs.len() as f64;
        if let Some(anomaly) = self.record_metric("tx_volume", tx_count).await? {
            // Add block information to the anomaly
            let mut updated_anomaly = anomaly.clone();
            updated_anomaly
                .related_entities
                .block_hashes
                .push(block.hash.clone());

            // Update the stored anomaly
            let mut anomalies = self.anomalies.write().await;
            if let Some(last) = anomalies.back_mut() {
                if last.detection_time == updated_anomaly.detection_time
                    && last.anomaly_type == updated_anomaly.anomaly_type
                {
                    *last = updated_anomaly.clone();
                }
            }

            detected_anomalies.push(updated_anomaly);
        }

        // Calculate block size (approximate)
        let block_size = block.txs.iter().map(|tx| tx.size.unwrap_or(0)).sum::<u64>() as f64;

        if let Some(anomaly) = self.record_metric("block_size", block_size).await? {
            // Add block information to the anomaly
            let mut updated_anomaly = anomaly.clone();
            updated_anomaly
                .related_entities
                .block_hashes
                .push(block.hash.clone());

            // Update the stored anomaly
            let mut anomalies = self.anomalies.write().await;
            if let Some(last) = anomalies.back_mut() {
                if last.detection_time == updated_anomaly.detection_time
                    && last.anomaly_type == updated_anomaly.anomaly_type
                {
                    *last = updated_anomaly.clone();
                }
            }

            detected_anomalies.push(updated_anomaly);
        }

        // Record block time if we have timestamps
        if let Some(timestamp) = block.timestamp {
            if let Some(proposal_time) = block.proposal_timestamp {
                let block_time = (timestamp - proposal_time) as f64;

                if let Some(anomaly) = self.record_metric("block_time", block_time).await? {
                    // Add block information to the anomaly
                    let mut updated_anomaly = anomaly.clone();
                    updated_anomaly
                        .related_entities
                        .block_hashes
                        .push(block.hash.clone());

                    // Update the stored anomaly
                    let mut anomalies = self.anomalies.write().await;
                    if let Some(last) = anomalies.back_mut() {
                        if last.detection_time == updated_anomaly.detection_time
                            && last.anomaly_type == updated_anomaly.anomaly_type
                        {
                            *last = updated_anomaly.clone();
                        }
                    }

                    detected_anomalies.push(updated_anomaly);
                }
            }
        }

        // Process transactions for fee anomalies
        for tx in &block.txs {
            if let Some(fee) = tx.fee {
                let fee_rate = fee as f64 / tx.size.unwrap_or(1) as f64;

                if let Some(anomaly) = self.record_metric("fee_rate", fee_rate).await? {
                    // Add transaction information to the anomaly
                    let mut updated_anomaly = anomaly.clone();
                    updated_anomaly
                        .related_entities
                        .block_hashes
                        .push(block.hash.clone());
                    updated_anomaly
                        .related_entities
                        .tx_hashes
                        .push(tx.hash.clone());

                    // Update the stored anomaly
                    let mut anomalies = self.anomalies.write().await;
                    if let Some(last) = anomalies.back_mut() {
                        if last.detection_time == updated_anomaly.detection_time
                            && last.anomaly_type == updated_anomaly.anomaly_type
                        {
                            *last = updated_anomaly.clone();
                        }
                    }

                    detected_anomalies.push(updated_anomaly);
                }
            }
        }

        Ok(detected_anomalies)
    }

    /// Update anomaly status
    pub async fn update_anomaly_status(
        &self,
        detection_time: u64,
        anomaly_type: AnomalyType,
        new_status: AnomalyStatus,
    ) -> Result<bool> {
        let mut anomalies = self.anomalies.write().await;

        for anomaly in anomalies.iter_mut() {
            if anomaly.detection_time == detection_time && anomaly.anomaly_type == anomaly_type {
                anomaly.status = new_status;

                // Add recovery time if being resolved
                if new_status == AnomalyStatus::Resolved {
                    anomaly.recovery_time = Some(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    );
                }

                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get all anomalies
    pub async fn get_all_anomalies(&self) -> Vec<Anomaly> {
        self.anomalies.read().await.iter().cloned().collect()
    }

    /// Get anomalies by type
    pub async fn get_anomalies_by_type(&self, anomaly_type: AnomalyType) -> Vec<Anomaly> {
        self.anomalies
            .read()
            .await
            .iter()
            .filter(|a| a.anomaly_type == anomaly_type)
            .cloned()
            .collect()
    }

    /// Get anomalies by status
    pub async fn get_anomalies_by_status(&self, status: AnomalyStatus) -> Vec<Anomaly> {
        self.anomalies
            .read()
            .await
            .iter()
            .filter(|a| a.status == status)
            .cloned()
            .collect()
    }

    /// Get anomalies for a block
    pub async fn get_anomalies_for_block(&self, block_hash: &[u8]) -> Vec<Anomaly> {
        self.anomalies
            .read()
            .await
            .iter()
            .filter(|a| {
                a.related_entities
                    .block_hashes
                    .iter()
                    .any(|h| h == block_hash)
            })
            .cloned()
            .collect()
    }

    /// Get anomalies for a transaction
    pub async fn get_anomalies_for_transaction(&self, tx_hash: &[u8]) -> Vec<Anomaly> {
        self.anomalies
            .read()
            .await
            .iter()
            .filter(|a| a.related_entities.tx_hashes.iter().any(|h| h == tx_hash))
            .cloned()
            .collect()
    }

    /// Update configuration
    pub async fn update_config(&self, config: AnomalyDetectionConfig) -> Result<()> {
        let mut cfg = self.config.write().await;
        *cfg = config;
        Ok(())
    }

    /// Clear all anomalies
    pub async fn clear_anomalies(&self) -> Result<usize> {
        let mut anomalies = self.anomalies.write().await;
        let count = anomalies.len();
        anomalies.clear();
        Ok(count)
    }

    /// Get current baselines for metrics
    pub async fn get_baselines(&self) -> HashMap<String, (f64, f64)> {
        self.baseline_metrics.read().await.clone()
    }
}

impl Clone for AnomalyDetector {
    fn clone(&self) -> Self {
        // This is a partial clone for use in async tasks
        Self {
            config: RwLock::new(self.config.try_read().unwrap_or_default().clone()),
            time_series: RwLock::new(HashMap::new()),
            anomalies: RwLock::new(VecDeque::new()),
            running: RwLock::new(false),
            #[cfg(feature = "ml_detection")]
            ml_model: None,
            last_analysis: RwLock::new(Instant::now()),
            baseline_metrics: RwLock::new(HashMap::new()),
        }
    }
}

/// Calculate the mean of a sequence of values
fn calculate_mean(values: &VecDeque<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let sum: f64 = values.iter().sum();
    sum / values.len() as f64
}

/// Calculate the standard deviation
fn calculate_std_dev(values: &VecDeque<f64>, mean: f64) -> f64 {
    if values.len() <= 1 {
        return 1.0; // Default to avoid division by zero
    }

    let variance: f64 = values
        .iter()
        .map(|v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() as f64 - 1.0);

    variance.sqrt().max(0.00001) // Avoid division by zero later
}
