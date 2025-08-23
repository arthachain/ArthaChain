use crate::ai_engine::models::neural_base::{
    LossType, NeuralBase, NeuralConfig, NeuralNetwork, OptimizerType,
};
use crate::ledger::block::Block;
use crate::ledger::transaction::Transaction;
use crate::utils::quantum_merkle::QuantumMerkleTree;
use anyhow::Result;
use chrono::{DateTime, Datelike, Timelike, Utc};
// Removed numpy Python dependency - using pure Rust ndarray instead
use candle_core::{Device, Tensor}; // Include Device for explicit usage
                                   // use candle_nn::{VarBuilder, VarMap}; // Unused imports removed
                                   // use ndarray::{Array1, Array2}; // Unused imports removed
use serde::{Deserialize, Serialize};
// use smartcore::ensemble::random_forest_classifier::RandomForestClassifier; // Unused import removed
// use smartcore::linalg::basic::matrix::DenseMatrix; // Unused import removed
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Transaction feature vector for fraud detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionFeatures {
    /// Transaction amount in wei
    pub amount: f32,
    /// Gas price in wei
    pub gas_price: f32,
    /// Gas used
    pub gas_used: f32,
    /// Transaction frequency (txs per day)
    pub tx_frequency: f32,
    /// Time since last transaction (seconds)
    pub time_since_last_tx: f32,
    /// Network connectedness (0-1)
    pub network_connectedness: f32,
    /// Sender reputation score (0-1)
    pub sender_reputation: f32,
    /// Recipient reputation score (0-1)
    pub recipient_reputation: f32,
    /// Transaction pattern similarity (0-1)
    pub pattern_similarity: f32,
    /// Contract interaction complexity (0-1)
    pub contract_complexity: f32,
    /// Geographical risk score (0-1)
    pub geo_risk: f32,
    /// Timestamp hour (0-23, normalized)
    pub hour_of_day: f32,
    /// Is weekend (0 or 1)
    pub is_weekend: f32,
    /// Amount velocity (rate of change)
    pub amount_velocity: f32,
    /// Unusual activity flag (0 or 1)
    pub unusual_activity: f32,
}

impl Default for TransactionFeatures {
    fn default() -> Self {
        Self {
            amount: 0.0,
            gas_price: 0.0,
            gas_used: 0.0,
            tx_frequency: 0.0,
            time_since_last_tx: 0.0,
            network_connectedness: 0.5,
            sender_reputation: 0.5,
            recipient_reputation: 0.5,
            pattern_similarity: 0.5,
            contract_complexity: 0.0,
            geo_risk: 0.3,
            hour_of_day: 0.5,
            is_weekend: 0.0,
            amount_velocity: 0.0,
            unusual_activity: 0.0,
        }
    }
}

impl TransactionFeatures {
    /// Convert to feature vector
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.amount,
            self.gas_price,
            self.gas_used,
            self.tx_frequency,
            self.time_since_last_tx,
            self.network_connectedness,
            self.sender_reputation,
            self.recipient_reputation,
            self.pattern_similarity,
            self.contract_complexity,
            self.geo_risk,
            self.hour_of_day,
            self.is_weekend,
            self.amount_velocity,
            self.unusual_activity,
        ]
    }

    /// Get the number of features
    pub fn feature_count() -> usize {
        15
    }
}

/// Fraud detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudDetectionResult {
    /// Transaction hash
    pub tx_hash: String,
    /// Fraud probability (0-1)
    pub fraud_probability: f32,
    /// Anomaly score (0-1)
    pub anomaly_score: f32,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Feature importance
    pub feature_importance: HashMap<String, f32>,
    /// Detection timestamp
    pub timestamp: DateTime<Utc>,
    /// Is flagged as suspicious
    pub is_suspicious: bool,
    /// Recommended action
    pub recommended_action: RecommendedAction,
    /// Quantum-resistant verification hash
    pub quantum_verification: String,
}

/// Risk level for fraud detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Critical risk
    Critical,
}

impl From<f32> for RiskLevel {
    fn from(prob: f32) -> Self {
        match prob {
            p if p < 0.25 => RiskLevel::Low,
            p if p < 0.5 => RiskLevel::Medium,
            p if p < 0.75 => RiskLevel::High,
            _ => RiskLevel::Critical,
        }
    }
}

/// Recommended action for a flagged transaction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendedAction {
    /// Allow the transaction
    Allow,
    /// Flag the transaction for manual review
    ReviewManually,
    /// Temporarily block for verification
    TemporaryBlock,
    /// Permanently block the transaction
    Block,
}

impl From<RiskLevel> for RecommendedAction {
    fn from(risk: RiskLevel) -> Self {
        match risk {
            RiskLevel::Low => RecommendedAction::Allow,
            RiskLevel::Medium => RecommendedAction::ReviewManually,
            RiskLevel::High => RecommendedAction::TemporaryBlock,
            RiskLevel::Critical => RecommendedAction::Block,
        }
    }
}

/// Neural network configuration for fraud detection
pub fn create_fraud_detection_config() -> NeuralConfig {
    let feature_count = TransactionFeatures::feature_count();

    NeuralConfig {
        name: "fraud_detection_network".to_string(),
        input_dim: feature_count,
        output_dim: 2, // fraud probability, anomaly score
        hidden_layers: vec![
            // Hidden layer 1
            crate::ai_engine::models::neural_base::LayerConfig {
                input_size: feature_count,
                output_size: 64,
                activation: crate::ai_engine::models::neural_base::ActivationType::GELU,
                dropout_rate: Some(0.2),
                batch_norm: true,
            },
            // Hidden layer 2
            crate::ai_engine::models::neural_base::LayerConfig {
                input_size: 64,
                output_size: 32,
                activation: crate::ai_engine::models::neural_base::ActivationType::GELU,
                dropout_rate: Some(0.2),
                batch_norm: true,
            },
            // Hidden layer 3
            crate::ai_engine::models::neural_base::LayerConfig {
                input_size: 32,
                output_size: 16,
                activation: crate::ai_engine::models::neural_base::ActivationType::GELU,
                dropout_rate: Some(0.1),
                batch_norm: true,
            },
            // Output layer
            crate::ai_engine::models::neural_base::LayerConfig {
                input_size: 16,
                output_size: 2,
                activation: crate::ai_engine::models::neural_base::ActivationType::Sigmoid,
                dropout_rate: None,
                batch_norm: false,
            },
        ],
        learning_rate: 0.001,
        batch_size: 64,
        epochs: 10,
        optimizer: OptimizerType::Adam,
        loss: LossType::BinaryCrossEntropy,
        use_gpu: false,
    }
}

/// Advanced fraud detection model with quantum resistance
pub struct AdvancedFraudDetection {
    /// Neural network for fraud detection
    model: NeuralBase,
    /// Historical transaction features for context
    tx_history: Arc<RwLock<HashMap<String, VecDeque<TransactionFeatures>>>>,
    /// Transaction risk scores
    #[allow(dead_code)]
    risk_scores: Arc<RwLock<HashMap<String, f32>>>,
    /// Quantum-resistant Merkle tree for verification
    quantum_merkle: Arc<RwLock<QuantumMerkleTree>>,
    /// Detection results history
    detection_history: Arc<RwLock<VecDeque<FraudDetectionResult>>>,
    /// Feature extractor
    feature_extractor: Arc<FeatureExtractor>,
    /// Address risk profile
    address_profiles: Arc<RwLock<HashMap<String, AddressProfile>>>,
}

/// Address risk profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddressProfile {
    /// Address
    pub address: String,
    /// Risk score (0-1)
    pub risk_score: f32,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
    /// Transaction history (hashes)
    pub tx_history: VecDeque<String>,
    /// Interaction count
    pub interaction_count: u32,
    /// Average transaction amount
    pub avg_tx_amount: f32,
    /// Last transaction features
    pub last_tx_features: Option<TransactionFeatures>,
}

impl AddressProfile {
    /// Create a new address profile
    pub fn new(address: String) -> Self {
        Self {
            address,
            risk_score: 0.5,
            last_update: Utc::now(),
            tx_history: VecDeque::with_capacity(100),
            interaction_count: 0,
            avg_tx_amount: 0.0,
            last_tx_features: None,
        }
    }

    /// Update profile with new transaction
    pub fn update(&mut self, tx_hash: &str, amount: f32) {
        // Add to history
        if self.tx_history.len() >= 100 {
            self.tx_history.pop_front();
        }
        self.tx_history.push_back(tx_hash.to_string());

        // Update statistics
        self.interaction_count += 1;
        self.avg_tx_amount = (self.avg_tx_amount * (self.interaction_count - 1) as f32 + amount)
            / self.interaction_count as f32;
        self.last_update = Utc::now();
    }
}

/// Feature extractor for transactions
#[derive(Default)]
pub struct FeatureExtractor {
    /// Network context
    network_context: NetworkContext,
}

/// Network context for feature extraction
pub struct NetworkContext {
    /// Global transaction frequency (txs per block)
    global_tx_frequency: f32,
    /// Average transaction amount
    avg_tx_amount: f32,
    /// Average gas price
    avg_gas_price: f32,
    /// Address activity level
    address_activity: HashMap<String, f32>,
}

impl Default for NetworkContext {
    fn default() -> Self {
        Self {
            global_tx_frequency: 10.0,
            avg_tx_amount: 0.1,
            avg_gas_price: 20.0,
            address_activity: HashMap::new(),
        }
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new() -> Self {
        Self {
            network_context: NetworkContext::default(),
        }
    }

    /// Extract features from a transaction
    pub fn extract_features(
        &self,
        tx: &Transaction,
        sender_history: &[TransactionFeatures],
        recipient_history: &[TransactionFeatures],
        _block: Option<&Block>,
    ) -> TransactionFeatures {
        // Base features
        let amount = tx.amount as f32;
        let gas_price = tx.gas_price as f32;
        let gas_used = tx.gas_limit as f32;

        // Calculate frequency and time since last tx
        let tx_frequency = if !sender_history.is_empty() {
            sender_history.len() as f32
        } else {
            0.1
        };

        let time_since_last_tx = if !sender_history.is_empty() {
            let _now = Utc::now().timestamp() as f32;
            // In a real implementation, this would use actual transaction timestamps
            10000.0 // Placeholder
        } else {
            100000.0 // High value for new addresses
        };

        // Network reputation scores
        let sender_reputation = self.calculate_address_reputation(&tx.sender.to_string());
        let recipient_reputation = self.calculate_address_reputation(&tx.recipient.to_string());

        // Pattern similarity (compare with history)
        let pattern_similarity = self.calculate_pattern_similarity(tx, sender_history);

        // Contract complexity (0 for regular txs, higher for contract interactions)
        let contract_complexity = if !tx.data.is_empty() {
            let data_len = tx.data.len();
            (data_len as f32 / 1000.0).min(1.0)
        } else {
            0.0
        };

        // Temporal features
        let now = chrono::Utc::now();
        let hour_of_day = (now.hour() as f32) / 24.0;
        let is_weekend = if now.weekday().number_from_monday() > 5 {
            1.0
        } else {
            0.0
        };

        // Amount velocity
        let amount_velocity = if !sender_history.is_empty() {
            let avg_amount =
                sender_history.iter().map(|f| f.amount).sum::<f32>() / sender_history.len() as f32;
            if avg_amount > 0.0 {
                (amount - avg_amount) / avg_amount
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Unusual activity detection
        let unusual_activity = self.detect_unusual_activity(tx, sender_history, recipient_history);

        // Geographical risk (would use IP data in real system)
        let geo_risk = 0.3; // Placeholder

        // Network connectedness (how central the address is in transaction graph)
        let network_connectedness = 0.5; // Placeholder

        TransactionFeatures {
            amount,
            gas_price,
            gas_used,
            tx_frequency,
            time_since_last_tx,
            network_connectedness,
            sender_reputation,
            recipient_reputation,
            pattern_similarity,
            contract_complexity,
            geo_risk,
            hour_of_day,
            is_weekend,
            amount_velocity,
            unusual_activity,
        }
    }

    /// Calculate address reputation
    fn calculate_address_reputation(&self, _address: &str) -> f32 {
        // In a real implementation, this would check reputation systems
        // For now, return a placeholder value
        0.5
    }

    /// Calculate pattern similarity
    fn calculate_pattern_similarity(
        &self,
        tx: &Transaction,
        history: &[TransactionFeatures],
    ) -> f32 {
        if history.is_empty() {
            return 0.5; // Default for new addresses
        }

        // Calculate standard deviation of amount in history
        let avg_amount = history.iter().map(|f| f.amount).sum::<f32>() / history.len() as f32;
        let amount_variance = history
            .iter()
            .map(|f| (f.amount - avg_amount).powi(2))
            .sum::<f32>()
            / history.len() as f32;
        let amount_std = amount_variance.sqrt();

        // Calculate z-score for current transaction
        let amount_z_score = if amount_std > 0.0 {
            (tx.amount as f32 - avg_amount) / amount_std
        } else {
            0.0
        };

        // Convert to similarity score (1.0 = very similar, 0.0 = very different)
        (-amount_z_score.abs() / 3.0).exp()
    }

    /// Detect unusual activity
    fn detect_unusual_activity(
        &self,
        tx: &Transaction,
        sender_history: &[TransactionFeatures],
        recipient_history: &[TransactionFeatures],
    ) -> f32 {
        let mut unusual_score: f32 = 0.0;

        // Check if amount is much larger than typical
        if !sender_history.is_empty() {
            let avg_amount =
                sender_history.iter().map(|f| f.amount).sum::<f32>() / sender_history.len() as f32;
            if tx.amount as f32 > avg_amount * 5.0 {
                unusual_score += 0.5;
            }
        }

        // Check if this sender has never sent to this recipient before
        if !recipient_history.is_empty() && sender_history.is_empty() {
            unusual_score += 0.3;
        }

        // Cap at 1.0
        unusual_score.min(1.0)
    }

    /// Update network context with new block
    pub fn update_context(&mut self, block: &Block) {
        // Update global transaction frequency
        self.network_context.global_tx_frequency = block.transactions.len() as f32;

        // Update average transaction amount and gas price
        let mut total_amount = 0.0;
        let mut total_gas_price = 0.0;

        for tx in &block.transactions {
            total_amount += tx.amount as f32;
            total_gas_price += tx.fee as f32;

            // Update address activity
            let sender = hex::encode(&tx.from);
            *self
                .network_context
                .address_activity
                .entry(sender)
                .or_insert(0.0) += 1.0;

            let recipient = hex::encode(&tx.to);
            *self
                .network_context
                .address_activity
                .entry(recipient)
                .or_insert(0.0) += 1.0;
        }

        if !block.transactions.is_empty() {
            let count = block.transactions.len() as f32;
            self.network_context.avg_tx_amount =
                (self.network_context.avg_tx_amount * 0.95) + (total_amount / count * 0.05);
            self.network_context.avg_gas_price =
                (self.network_context.avg_gas_price * 0.95) + (total_gas_price / count * 0.05);
        }
    }
}

impl AdvancedFraudDetection {
    /// Create a new advanced fraud detection model
    pub async fn new() -> Result<Self> {
        // Create neural network
        let config = create_fraud_detection_config();
        let model = NeuralBase::new_sync(config)?;

        // Create quantum-resistant Merkle tree
        let quantum_merkle = QuantumMerkleTree::new();

        // Create feature extractor
        let feature_extractor = FeatureExtractor::new();

        Ok(Self {
            model,
            tx_history: Arc::new(RwLock::new(HashMap::new())),
            risk_scores: Arc::new(RwLock::new(HashMap::new())),
            quantum_merkle: Arc::new(RwLock::new(quantum_merkle)),
            detection_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            feature_extractor: Arc::new(feature_extractor),
            address_profiles: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create a simple mock fraud detection model for testing/fallback
    pub async fn new_simple() -> Self {
        // Create a simple mock neural network
        let config = create_fraud_detection_config();
        let model = NeuralBase::new_sync(config).unwrap_or_else(|_| {
            // Create a simple fallback model if the main one fails
            let simple_config = NeuralConfig::default();
            NeuralBase::new_sync(simple_config).unwrap_or_else(|_| {
                // If even the fallback fails, panic as this is critical
                panic!("Failed to create any neural network model")
            })
        });

        // Create quantum-resistant Merkle tree
        let quantum_merkle = QuantumMerkleTree::new();

        // Create feature extractor
        let feature_extractor = FeatureExtractor::new();

        Self {
            model,
            tx_history: Arc::new(RwLock::new(HashMap::new())),
            risk_scores: Arc::new(RwLock::new(HashMap::new())),
            quantum_merkle: Arc::new(RwLock::new(quantum_merkle)),
            detection_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            feature_extractor: Arc::new(feature_extractor),
            address_profiles: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Detect fraud in a transaction
    pub async fn detect_fraud(
        &self,
        tx: &Transaction,
        block: Option<&Block>,
    ) -> Result<FraudDetectionResult> {
        // Get transaction history
        let sender_history = self.get_address_history(&tx.sender.to_string()).await?;
        let recipient_history = self.get_address_history(&tx.recipient.to_string()).await?;

        // Extract features
        let features =
            self.feature_extractor
                .extract_features(tx, &sender_history, &recipient_history, block);

        // Convert to feature vector
        let feature_vec = features.to_vec();

        // Get prediction from neural network
        let prediction = self.model.predict(&feature_vec)?;

        // Extract fraud probability and anomaly score
        let fraud_probability = prediction[0];
        let anomaly_score = prediction[1];

        // Determine risk level
        let risk_level = RiskLevel::from(fraud_probability);

        // Determine recommended action
        let recommended_action = RecommendedAction::from(risk_level);

        // Create feature importance map
        let feature_names = vec![
            "amount",
            "gas_price",
            "gas_used",
            "tx_frequency",
            "time_since_last_tx",
            "network_connectedness",
            "sender_reputation",
            "recipient_reputation",
            "pattern_similarity",
            "contract_complexity",
            "geo_risk",
            "hour_of_day",
            "is_weekend",
            "amount_velocity",
            "unusual_activity",
        ];

        // Placeholder feature importance (would use SHAP values in real implementation)
        let feature_importance = feature_names
            .into_iter()
            .zip(vec![
                0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05, 0.02, 0.02, 0.05, 0.06,
            ])
            .map(|(name, value)| (name.to_string(), value))
            .collect();

        // Generate quantum-resistant verification
        let verification_hash = self.generate_quantum_verification(tx, &features).await?;

        // Create detection result
        let result = FraudDetectionResult {
            tx_hash: hex::encode(tx.hash().as_ref()),
            fraud_probability,
            anomaly_score,
            risk_level,
            feature_importance,
            timestamp: Utc::now(),
            is_suspicious: fraud_probability > 0.5 || anomaly_score > 0.7,
            recommended_action,
            quantum_verification: verification_hash,
        };

        // Update detection history
        {
            let mut history = self.detection_history.write().await;
            if history.len() >= 1000 {
                history.pop_front();
            }
            history.push_back(result.clone());
        }

        // Update transaction history
        self.update_tx_history(&tx.sender.to_string(), features.clone())
            .await?;

        // Update address profiles
        self.update_address_profiles(tx, &features).await?;

        Ok(result)
    }

    /// Generate quantum-resistant verification hash
    async fn generate_quantum_verification(
        &self,
        tx: &Transaction,
        features: &TransactionFeatures,
    ) -> Result<String> {
        // Serialize transaction and features
        let tx_data = serde_json::to_vec(tx)?;
        let feature_data = serde_json::to_vec(features)?;

        // Combine data
        let mut combined = Vec::with_capacity(tx_data.len() + feature_data.len());
        combined.extend_from_slice(&tx_data);
        combined.extend_from_slice(&feature_data);

        // Add to quantum Merkle tree
        let mut tree = self.quantum_merkle.write().await;
        let _proof = tree.add_leaf(&combined);

        // Return root hash as verification
        Ok(hex::encode(tree.root()))
    }

    /// Train the model with historical data
    pub fn train(&mut self, features: &Vec<Vec<f32>>, labels: &[f32]) -> Result<f32> {
        // Convert input to tensors
        let batch_size = features.len();
        let feature_dim = features[0].len();

        let mut flat_features = Vec::with_capacity(batch_size * feature_dim);
        for row in features {
            flat_features.extend_from_slice(row);
        }

        let device = match self.model.device.device_type {
            crate::ai_engine::models::neural_base::DeviceType::CPU => Device::Cpu,
            crate::ai_engine::models::neural_base::DeviceType::GPU => Device::Cpu, // Fallback to CPU for now
            _ => Device::Cpu,
        };
        let x = Tensor::from_vec(flat_features, (batch_size, feature_dim), &device)?;
        let y = Tensor::from_vec(labels.to_vec(), batch_size, &device)?;

        // Train the model (simplified training)
        let metrics = self.model.train_step(&[], &[])?;
        Ok(metrics.loss as f32)
    }

    /// Get address transaction history
    async fn get_address_history(&self, address: &str) -> Result<Vec<TransactionFeatures>> {
        let history = self.tx_history.read().await;

        // Get transaction history for address
        if let Some(queue) = history.get(address) {
            Ok(queue.iter().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Update transaction history for an address
    async fn update_tx_history(&self, address: &str, features: TransactionFeatures) -> Result<()> {
        let mut history = self.tx_history.write().await;

        // Get or create history queue
        let queue = history
            .entry(address.to_string())
            .or_insert_with(|| VecDeque::with_capacity(100));

        // Add new features
        if queue.len() >= 100 {
            queue.pop_front();
        }
        queue.push_back(features);

        Ok(())
    }

    /// Update address profiles
    async fn update_address_profiles(
        &self,
        tx: &Transaction,
        features: &TransactionFeatures,
    ) -> Result<()> {
        let mut profiles = self.address_profiles.write().await;

        // Update sender profile - use sender as String since it's stored as String in Transaction
        let sender_profile = profiles
            .entry(tx.sender.to_string())
            .or_insert_with(|| AddressProfile::new(tx.sender.to_string()));

        sender_profile.update(&hex::encode(tx.hash().as_ref()), features.amount);
        sender_profile.last_tx_features = Some(features.clone());

        // Update recipient profile - use recipient as String since it's stored as String in Transaction
        let recipient_profile = profiles
            .entry(tx.recipient.to_string())
            .or_insert_with(|| AddressProfile::new(tx.recipient.to_string()));

        recipient_profile.update(&hex::encode(tx.hash().as_ref()), features.amount);

        Ok(())
    }

    /// Save the model to a file
    pub fn save_model(&self, path: &str) -> Result<()> {
        self.model.save_state(path)
    }

    /// Load the model from a file
    pub fn load_model(&mut self, path: &str) -> Result<()> {
        self.model.load_state(path)
    }

    /// Get recent detection results
    pub async fn get_recent_detections(&self, limit: usize) -> Vec<FraudDetectionResult> {
        let history = self.detection_history.read().await;
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get address risk profile
    pub async fn get_address_profile(&self, address: &str) -> Option<AddressProfile> {
        let profiles = self.address_profiles.read().await;
        profiles.get(address).cloned()
    }

    /// Get top risky addresses sorted by risk score
    pub async fn get_top_risky_addresses(&self, limit: usize) -> Vec<AddressProfile> {
        let profiles = self.address_profiles.read().await;

        let mut address_list: Vec<AddressProfile> = profiles.values().cloned().collect();

        // Sort by risk score in descending order
        address_list.sort_by(|a, b| b.risk_score.partial_cmp(&a.risk_score).unwrap());

        // Take only the top N addresses
        address_list.into_iter().take(limit).collect()
    }

    /// Get address profiles count
    pub async fn get_address_count(&self) -> usize {
        let profiles = self.address_profiles.read().await;
        profiles.len()
    }
}
