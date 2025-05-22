use crate::consensus::byzantine::ByzantineEvidence;
use crate::consensus::byzantine::ByzantineFaultType;
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

/// Types of fraud that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FraudType {
    /// Double spending attack
    DoubleSpend,
    /// Block withholding
    BlockWithholding,
    /// Equivocation (double signing)
    Equivocation,
    /// Balance manipulation
    BalanceManipulation,
    /// Invalid state transition
    InvalidStateTransition,
    /// Unauthorized transaction
    UnauthorizedTransaction,
    /// Replay attack
    ReplayAttack,
    /// Forged signature
    ForgedSignature,
    /// Front-running
    FrontRunning,
    /// Time manipulation
    TimeManipulation,
    /// Fee manipulation
    FeeManipulation,
    /// Invalid proof of work
    InvalidProofOfWork,
    /// Sybil attack
    SybilAttack,
    /// Transaction censoring
    TransactionCensoring,
}

/// Configuration for fraud detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudDetectionConfig {
    /// Enable fraud detection
    pub enabled: bool,
    /// History size to keep for double-spend detection
    pub tx_history_size: usize,
    /// Enable machine learning detection
    pub enable_ml_detection: bool,
    /// Confidence threshold for ML detection
    pub ml_confidence_threshold: f64,
    /// Reporting threshold (number of observations)
    pub reporting_threshold: usize,
    /// Auto-report detected fraud
    pub auto_report: bool,
    /// Analysis interval in milliseconds
    pub analysis_interval_ms: u64,
    /// Specificity preference (0.0-1.0) - higher means fewer false positives
    pub specificity_preference: f64,
    /// Enable real-time alerting
    pub enable_alerting: bool,
    /// Enable on-chain reporting
    pub enable_on_chain_reporting: bool,
}

impl Default for FraudDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tx_history_size: 10000,
            enable_ml_detection: true,
            ml_confidence_threshold: 0.85,
            reporting_threshold: 3,
            auto_report: true,
            analysis_interval_ms: 30000, // 30 seconds
            specificity_preference: 0.7,
            enable_alerting: true,
            enable_on_chain_reporting: true,
        }
    }
}

/// Evidence of fraud
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudEvidence {
    /// Type of fraud
    pub fraud_type: FraudType,
    /// Related transaction hashes
    pub tx_hashes: Vec<Vec<u8>>,
    /// Related block hashes
    pub block_hashes: Vec<Vec<u8>>,
    /// Suspected malicious node
    pub suspect_node: Option<NodeId>,
    /// Evidence data
    pub evidence_data: Vec<u8>,
    /// Detection timestamp
    pub timestamp: u64,
    /// Detection confidence (0.0-1.0)
    pub confidence: f64,
    /// Detection method
    pub detection_method: DetectionMethod,
    /// Description of the fraud
    pub description: String,
}

/// Method used for fraud detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Rule-based detection
    RuleBased,
    /// Statistical detection
    Statistical,
    /// Machine learning detection
    MachineLearning,
    /// Reported by peer
    PeerReport,
    /// Hybrid detection
    Hybrid,
}

/// The fraud detection engine
pub struct FraudDetectionEngine {
    /// Configuration
    config: RwLock<FraudDetectionConfig>,
    /// Transaction history for double-spend detection
    tx_history: RwLock<VecDeque<Transaction>>,
    /// Transaction outputs already spent
    spent_outputs: RwLock<HashSet<String>>,
    /// Detected fraud
    detected_fraud: RwLock<Vec<FraudEvidence>>,
    /// Detection metrics
    metrics: RwLock<FraudDetectionMetrics>,
    /// Running flag
    running: RwLock<bool>,
    /// Block header history
    block_headers: RwLock<HashMap<Vec<u8>, BlockHeaderInfo>>,
    /// Suspected malicious nodes
    suspicious_nodes: RwLock<HashMap<NodeId, Vec<FraudEvidence>>>,
    /// AI model for fraud detection
    #[cfg(feature = "ml_detection")]
    ml_model: Option<Arc<crate::ai_engine::AnomalyDetector>>,
}

/// Block header information for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlockHeaderInfo {
    /// Block hash
    pub hash: Vec<u8>,
    /// Block height
    pub height: u64,
    /// Previous block hash
    pub prev_hash: Vec<u8>,
    /// Block timestamp
    pub timestamp: u64,
    /// Miner/validator ID
    pub miner: NodeId,
}

/// Metrics for fraud detection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FraudDetectionMetrics {
    /// Total analyzed transactions
    pub total_transactions_analyzed: u64,
    /// Total analyzed blocks
    pub total_blocks_analyzed: u64,
    /// Total detected frauds
    pub total_frauds_detected: u64,
    /// Total false positives (if known)
    pub total_false_positives: u64,
    /// Detected frauds by type
    pub frauds_by_type: HashMap<FraudType, u64>,
    /// Total auto-reported frauds
    pub total_auto_reported: u64,
    /// Average detection time in milliseconds
    pub avg_detection_time_ms: f64,
    /// Detection rate (fraud per 1000 transactions)
    pub detection_rate: f64,
}

impl FraudDetectionEngine {
    /// Create a new fraud detection engine
    pub fn new(config: FraudDetectionConfig) -> Self {
        Self {
            config: RwLock::new(config),
            tx_history: RwLock::new(VecDeque::new()),
            spent_outputs: RwLock::new(HashSet::new()),
            detected_fraud: RwLock::new(Vec::new()),
            metrics: RwLock::new(FraudDetectionMetrics::default()),
            running: RwLock::new(false),
            block_headers: RwLock::new(HashMap::new()),
            suspicious_nodes: RwLock::new(HashMap::new()),
            #[cfg(feature = "ml_detection")]
            ml_model: None,
        }
    }

    /// Start the fraud detection engine
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(anyhow!("Fraud detection engine already running"));
        }

        // Initialize the ML model if enabled
        #[cfg(feature = "ml_detection")]
        if self.config.read().await.enable_ml_detection {
            self.ml_model = Some(Arc::new(crate::ai_engine::AnomalyDetector::new().await?));
        }

        *running = true;

        // Start periodic analysis task
        self.start_analysis_task();

        info!("Fraud detection engine started");
        Ok(())
    }

    /// Stop the fraud detection engine
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(anyhow!("Fraud detection engine not running"));
        }

        *running = false;
        info!("Fraud detection engine stopped");
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
                    warn!("Error in fraud analysis: {}", e);
                }
            }
        });
    }

    /// Run periodic analysis of transaction and block data
    async fn run_periodic_analysis(&self) -> Result<()> {
        let config = self.config.read().await;
        if !config.enabled {
            return Ok(());
        }

        // Analyze transaction patterns
        self.analyze_transaction_patterns().await?;

        // Analyze block patterns
        self.analyze_block_patterns().await?;

        Ok(())
    }

    /// Analyze transaction patterns for fraud
    async fn analyze_transaction_patterns(&self) -> Result<()> {
        let tx_history = self.tx_history.read().await;
        if tx_history.is_empty() {
            return Ok(());
        }

        let start_time = Instant::now();
        let mut frauds_detected = Vec::new();

        // Check for double-spend attempts
        let mut seen_inputs = HashSet::new();
        for tx in tx_history.iter() {
            for input in &tx.inputs {
                let input_key = format!("{}:{}", hex::encode(&input.prev_tx), input.prev_index);
                if seen_inputs.contains(&input_key) {
                    // Potential double-spend
                    frauds_detected.push(FraudEvidence {
                        fraud_type: FraudType::DoubleSpend,
                        tx_hashes: vec![tx.hash.clone()],
                        block_hashes: Vec::new(),
                        suspect_node: None,
                        evidence_data: input_key.as_bytes().to_vec(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        confidence: 0.95,
                        detection_method: DetectionMethod::RuleBased,
                        description: format!(
                            "Double-spend attempt detected for input {}",
                            input_key
                        ),
                    });
                }

                seen_inputs.insert(input_key);
            }
        }

        // Record detected frauds
        if !frauds_detected.is_empty() {
            let mut detected = self.detected_fraud.write().await;
            let mut metrics = self.metrics.write().await;

            for fraud in &frauds_detected {
                detected.push(fraud.clone());

                // Update metrics
                metrics.total_frauds_detected += 1;
                *metrics
                    .frauds_by_type
                    .entry(fraud.fraud_type.clone())
                    .or_insert(0) += 1;

                info!(
                    "Detected fraud: {} (confidence: {:.2})",
                    fraud.description, fraud.confidence
                );

                // Auto-report if enabled
                if self.config.read().await.auto_report {
                    if let Err(e) = self.report_fraud(fraud).await {
                        warn!("Failed to auto-report fraud: {}", e);
                    } else {
                        metrics.total_auto_reported += 1;
                    }
                }
            }

            // Update detection time
            let elapsed_ms = start_time.elapsed().as_millis() as f64;
            let alpha = 0.2; // Smoothing factor for exponential moving average
            metrics.avg_detection_time_ms =
                alpha * elapsed_ms + (1.0 - alpha) * metrics.avg_detection_time_ms;
        }

        Ok(())
    }

    /// Analyze block patterns for fraud
    async fn analyze_block_patterns(&self) -> Result<()> {
        let block_headers = self.block_headers.read().await;
        if block_headers.is_empty() {
            return Ok(());
        }

        let start_time = Instant::now();
        let mut frauds_detected = Vec::new();

        // Check for time manipulation
        for (_, header) in block_headers.iter() {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            if header.timestamp > current_time + 300 {
                // 5 minutes in the future
                frauds_detected.push(FraudEvidence {
                    fraud_type: FraudType::TimeManipulation,
                    tx_hashes: Vec::new(),
                    block_hashes: vec![header.hash.clone()],
                    suspect_node: Some(header.miner.clone()),
                    evidence_data: header.timestamp.to_be_bytes().to_vec(),
                    timestamp: current_time,
                    confidence: 0.9,
                    detection_method: DetectionMethod::RuleBased,
                    description: format!(
                        "Block timestamp too far in the future: {} (current: {})",
                        header.timestamp, current_time
                    ),
                });

                // Add to suspicious nodes
                let mut suspicious = self.suspicious_nodes.write().await;
                suspicious
                    .entry(header.miner.clone())
                    .or_insert_with(Vec::new)
                    .push(frauds_detected.last().unwrap().clone());
            }
        }

        // Record detected frauds
        if !frauds_detected.is_empty() {
            let mut detected = self.detected_fraud.write().await;
            let mut metrics = self.metrics.write().await;

            for fraud in &frauds_detected {
                detected.push(fraud.clone());

                // Update metrics
                metrics.total_frauds_detected += 1;
                *metrics
                    .frauds_by_type
                    .entry(fraud.fraud_type.clone())
                    .or_insert(0) += 1;

                info!(
                    "Detected fraud: {} (confidence: {:.2})",
                    fraud.description, fraud.confidence
                );

                // Auto-report if enabled
                if self.config.read().await.auto_report {
                    if let Err(e) = self.report_fraud(fraud).await {
                        warn!("Failed to auto-report fraud: {}", e);
                    } else {
                        metrics.total_auto_reported += 1;
                    }
                }
            }

            // Update detection time
            let elapsed_ms = start_time.elapsed().as_millis() as f64;
            let alpha = 0.2; // Smoothing factor for exponential moving average
            metrics.avg_detection_time_ms =
                alpha * elapsed_ms + (1.0 - alpha) * metrics.avg_detection_time_ms;
        }

        Ok(())
    }

    /// Process a new transaction for fraud detection
    pub async fn process_transaction(&self, tx: &Transaction) -> Result<Option<FraudEvidence>> {
        let config = self.config.read().await;
        if !config.enabled {
            return Ok(None);
        }

        let start_time = Instant::now();
        let mut fraud_evidence = None;

        // Check for double-spend
        let is_double_spend = {
            let mut spent = self.spent_outputs.write().await;
            let mut double_spend = false;

            for input in &tx.inputs {
                let input_key = format!("{}:{}", hex::encode(&input.prev_tx), input.prev_index);
                if spent.contains(&input_key) {
                    double_spend = true;
                    break;
                }

                spent.insert(input_key);
            }

            double_spend
        };

        if is_double_spend {
            let evidence = FraudEvidence {
                fraud_type: FraudType::DoubleSpend,
                tx_hashes: vec![tx.hash.clone()],
                block_hashes: Vec::new(),
                suspect_node: None,
                evidence_data: tx.hash.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                confidence: 0.95,
                detection_method: DetectionMethod::RuleBased,
                description: "Double-spend detected".to_string(),
            };

            fraud_evidence = Some(evidence.clone());

            // Record the fraud
            let mut detected = self.detected_fraud.write().await;
            detected.push(evidence);
        }

        // Add to transaction history
        {
            let mut history = self.tx_history.write().await;
            history.push_back(tx.clone());

            // Trim history if needed
            while history.len() > config.tx_history_size {
                history.pop_front();
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_transactions_analyzed += 1;

            if let Some(ref fraud) = fraud_evidence {
                metrics.total_frauds_detected += 1;
                *metrics
                    .frauds_by_type
                    .entry(fraud.fraud_type.clone())
                    .or_insert(0) += 1;

                // Update detection rate
                metrics.detection_rate = metrics.total_frauds_detected as f64 * 1000.0
                    / metrics.total_transactions_analyzed as f64;
            }

            // Update detection time
            if fraud_evidence.is_some() {
                let elapsed_ms = start_time.elapsed().as_millis() as f64;
                let alpha = 0.2; // Smoothing factor for exponential moving average
                metrics.avg_detection_time_ms =
                    alpha * elapsed_ms + (1.0 - alpha) * metrics.avg_detection_time_ms;
            }
        }

        Ok(fraud_evidence)
    }

    /// Process a new block for fraud detection
    pub async fn process_block(&self, block: &Block) -> Result<Vec<FraudEvidence>> {
        let config = self.config.read().await;
        if !config.enabled {
            return Ok(Vec::new());
        }

        let start_time = Instant::now();
        let mut fraud_evidences = Vec::new();

        // Store block header
        {
            let header_info = BlockHeaderInfo {
                hash: block.hash.clone(),
                height: block.height,
                prev_hash: block.prev_hash.clone(),
                timestamp: block.timestamp.unwrap_or(0),
                miner: block.miner.clone().unwrap_or_else(|| "unknown".to_string()),
            };

            let mut headers = self.block_headers.write().await;
            headers.insert(block.hash.clone(), header_info);
        }

        // Check for timestamp manipulation
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if let Some(block_time) = block.timestamp {
            if block_time > current_time + 300 {
                // 5 minutes in the future
                let evidence = FraudEvidence {
                    fraud_type: FraudType::TimeManipulation,
                    tx_hashes: Vec::new(),
                    block_hashes: vec![block.hash.clone()],
                    suspect_node: block.miner.clone(),
                    evidence_data: block_time.to_be_bytes().to_vec(),
                    timestamp: current_time,
                    confidence: 0.9,
                    detection_method: DetectionMethod::RuleBased,
                    description: format!(
                        "Block timestamp too far in the future: {} (current: {})",
                        block_time, current_time
                    ),
                };

                fraud_evidences.push(evidence.clone());

                // Add to detected frauds
                let mut detected = self.detected_fraud.write().await;
                detected.push(evidence);

                // Add to suspicious nodes if miner is known
                if let Some(ref miner) = block.miner {
                    let mut suspicious = self.suspicious_nodes.write().await;
                    suspicious
                        .entry(miner.clone())
                        .or_insert_with(Vec::new)
                        .push(fraud_evidences.last().unwrap().clone());
                }
            }
        }

        // Process all transactions in the block
        for tx in &block.txs {
            if let Some(evidence) = self.process_transaction(tx).await? {
                fraud_evidences.push(evidence);
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_blocks_analyzed += 1;

            // Update detection time if fraud was detected
            if !fraud_evidences.is_empty() {
                let elapsed_ms = start_time.elapsed().as_millis() as f64;
                let alpha = 0.2; // Smoothing factor for exponential moving average
                metrics.avg_detection_time_ms =
                    alpha * elapsed_ms + (1.0 - alpha) * metrics.avg_detection_time_ms;
            }
        }

        Ok(fraud_evidences)
    }

    /// Apply machine learning detection
    #[cfg(feature = "ml_detection")]
    async fn apply_ml_detection(&self, tx: &Transaction) -> Result<Option<FraudEvidence>> {
        let config = self.config.read().await;
        if !config.enable_ml_detection || self.ml_model.is_none() {
            return Ok(None);
        }

        let ml_model = self.ml_model.as_ref().unwrap();

        // Extract features from transaction
        let features = self.extract_transaction_features(tx).await?;

        // Run model inference
        let (is_fraud, confidence, fraud_type) = ml_model.detect_fraud(features).await?;

        if is_fraud && confidence >= config.ml_confidence_threshold {
            let evidence = FraudEvidence {
                fraud_type: fraud_type.unwrap_or(FraudType::UnauthorizedTransaction),
                tx_hashes: vec![tx.hash.clone()],
                block_hashes: Vec::new(),
                suspect_node: None,
                evidence_data: tx.hash.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                confidence,
                detection_method: DetectionMethod::MachineLearning,
                description: format!(
                    "ML model detected potential fraud (confidence: {:.2})",
                    confidence
                ),
            };

            // Record the fraud
            let mut detected = self.detected_fraud.write().await;
            detected.push(evidence.clone());

            return Ok(Some(evidence));
        }

        Ok(None)
    }

    // Non-ML version that always returns None
    #[cfg(not(feature = "ml_detection"))]
    async fn apply_ml_detection(&self, _tx: &Transaction) -> Result<Option<FraudEvidence>> {
        Ok(None)
    }

    /// Extract features from a transaction for ML detection
    #[cfg(feature = "ml_detection")]
    async fn extract_transaction_features(&self, tx: &Transaction) -> Result<Vec<f32>> {
        // Simple feature extraction:
        // 1. Number of inputs
        // 2. Number of outputs
        // 3. Total input value
        // 4. Total output value
        // 5. Fee amount
        // 6. Is coinbase?
        // 7. Transaction size

        let mut features = Vec::new();

        features.push(tx.inputs.len() as f32);
        features.push(tx.outputs.len() as f32);

        // Other features would require access to the UTXO set for input values
        // For now, we'll use placeholders
        features.push(0.0); // Total input value

        let output_value = tx.outputs.iter().map(|o| o.value).sum::<u64>() as f32;
        features.push(output_value);

        features.push(tx.fee.unwrap_or(0) as f32);
        features.push(if tx.is_coinbase { 1.0 } else { 0.0 });
        features.push(tx.size.unwrap_or(0) as f32);

        Ok(features)
    }

    /// Report fraud to the network
    async fn report_fraud(&self, evidence: &FraudEvidence) -> Result<()> {
        // In a real implementation, this would send the evidence to the network
        info!("Reporting fraud to network: {}", evidence.description);

        // Convert to Byzantine evidence for the consensus system
        let byzantine_type = match evidence.fraud_type {
            FraudType::DoubleSpend => ByzantineFaultType::InvalidTransactions,
            FraudType::Equivocation => ByzantineFaultType::DoubleSigning,
            FraudType::BlockWithholding => ByzantineFaultType::BlockWithholding,
            FraudType::TimeManipulation => ByzantineFaultType::MalformedMessages,
            _ => ByzantineFaultType::InvalidTransactions,
        };

        let _byzantine_evidence = ByzantineEvidence {
            fault_type: byzantine_type,
            node_id: evidence
                .suspect_node
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            timestamp: evidence.timestamp,
            related_blocks: evidence.block_hashes.clone(),
            data: evidence.evidence_data.clone(),
            description: evidence.description.clone(),
            reporters: vec!["fraud_detection_engine".to_string()],
            evidence_hash: Vec::new(), // Would be computed by the Byzantine system
        };

        // In a real implementation:
        // 1. Submit evidence to the ByzantineManager
        // 2. Log evidence to persistent storage
        // 3. Optionally submit evidence as an on-chain transaction

        Ok(())
    }

    /// Get all detected fraud
    pub async fn get_all_fraud(&self) -> Vec<FraudEvidence> {
        self.detected_fraud.read().await.clone()
    }

    /// Get fraud by transaction hash
    pub async fn get_fraud_by_tx(&self, tx_hash: &[u8]) -> Vec<FraudEvidence> {
        let detected = self.detected_fraud.read().await;

        detected
            .iter()
            .filter(|e| e.tx_hashes.iter().any(|h| h == tx_hash))
            .cloned()
            .collect()
    }

    /// Get fraud by block hash
    pub async fn get_fraud_by_block(&self, block_hash: &[u8]) -> Vec<FraudEvidence> {
        let detected = self.detected_fraud.read().await;

        detected
            .iter()
            .filter(|e| e.block_hashes.iter().any(|h| h == block_hash))
            .cloned()
            .collect()
    }

    /// Get fraud by suspect node
    pub async fn get_fraud_by_node(&self, node_id: &str) -> Vec<FraudEvidence> {
        let detected = self.detected_fraud.read().await;

        detected
            .iter()
            .filter(|e| e.suspect_node.as_ref().map_or(false, |n| n == node_id))
            .cloned()
            .collect()
    }

    /// Get detection metrics
    pub async fn get_metrics(&self) -> FraudDetectionMetrics {
        self.metrics.read().await.clone()
    }

    /// Update configuration
    pub async fn update_config(&self, config: FraudDetectionConfig) -> Result<()> {
        let mut cfg = self.config.write().await;
        *cfg = config;
        Ok(())
    }

    /// Clear detected fraud
    pub async fn clear_detected_fraud(&self) -> Result<usize> {
        let mut detected = self.detected_fraud.write().await;
        let count = detected.len();
        detected.clear();
        Ok(count)
    }
}

impl Clone for FraudDetectionEngine {
    fn clone(&self) -> Self {
        // This is a partial clone for use in async tasks
        Self {
            config: RwLock::new(self.config.try_read().unwrap_or_default().clone()),
            tx_history: RwLock::new(VecDeque::new()),
            spent_outputs: RwLock::new(HashSet::new()),
            detected_fraud: RwLock::new(Vec::new()),
            metrics: RwLock::new(FraudDetectionMetrics::default()),
            running: RwLock::new(false),
            block_headers: RwLock::new(HashMap::new()),
            suspicious_nodes: RwLock::new(HashMap::new()),
            #[cfg(feature = "ml_detection")]
            ml_model: None,
        }
    }
}
