use anyhow::Result;
use serde::{Deserialize, Serialize};
use statrs::statistics::{OrderStatistics, Statistics};
use std::collections::HashMap;

/// Simple neural network placeholder
pub struct NeuralNetwork {
    weights: Vec<f64>,
}

impl NeuralNetwork {
    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.clone()
    }

    pub fn set_weights(&mut self, weights: Vec<f64>) -> Result<()> {
        self.weights = weights;
        Ok(())
    }
}

/// Model history tracking
#[derive(Default)]
pub struct ModelHistory {
    pub predictions: Vec<f64>,
    pub alerts: Vec<String>,
}

/// Pure Rust Device Health Anomaly Detection Model
pub struct DeviceHealthDetector {
    /// Model parameters
    params: ModelParams,
    /// Feature names
    feature_names: Vec<String>,
    /// Training data statistics
    feature_stats: HashMap<String, FeatureStats>,
    /// Anomaly thresholds
    thresholds: HashMap<String, f32>,
    /// Trained flag
    is_trained: bool,
    /// Model performance metrics
    current_accuracy: f64,
    prediction_window: std::time::Duration,
    alert_threshold: f64,
    /// Feature weights for importance
    feature_weights: HashMap<String, f64>,
    /// Optional neural network component
    neural_network: Option<NeuralNetwork>,
    /// Model history
    history: ModelHistory,
}

/// Model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    /// Contamination factor (expected fraction of anomalies)
    pub contamination: f32,
    /// Number of standard deviations for outlier detection
    pub outlier_threshold: f32,
    /// Threshold for anomaly score
    pub anomaly_threshold: f32,
    /// Minimum samples needed for training
    pub min_samples: usize,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            contamination: 0.1,
            outlier_threshold: 2.5,
            anomaly_threshold: 0.6,
            min_samples: 50,
        }
    }
}

/// Statistical features for each metric
#[derive(Debug, Clone)]
struct FeatureStats {
    mean: f32,
    std_dev: f32,
    min: f32,
    max: f32,
    median: f32,
    iqr: f32, // Interquartile range
}

impl FeatureStats {
    fn new(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                iqr: 0.0,
            };
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();

        let q1 = sorted_data[sorted_data.len() / 4];
        let median = sorted_data[sorted_data.len() / 2];
        let q3 = sorted_data[3 * sorted_data.len() / 4];
        let iqr = q3 - q1;

        Self {
            mean,
            std_dev,
            min: sorted_data[0],
            max: sorted_data[sorted_data.len() - 1],
            median,
            iqr,
        }
    }

    /// Calculate z-score for a value
    fn z_score(&self, value: f32) -> f32 {
        if self.std_dev > 0.0 {
            (value - self.mean) / self.std_dev
        } else {
            0.0
        }
    }

    /// Calculate modified z-score using median and MAD
    fn modified_z_score(&self, value: f32) -> f32 {
        let mad = self.iqr / 1.349; // Convert IQR to MAD approximation
        if mad > 0.0 {
            0.6745 * (value - self.median) / mad
        } else {
            0.0
        }
    }
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Overall anomaly score (0.0 to 1.0)
    pub anomaly_score: f32,
    /// Whether the sample is classified as anomalous
    pub is_anomaly: bool,
    /// Individual feature anomaly scores
    pub feature_scores: HashMap<String, f32>,
    /// Feature contributions to the anomaly score
    pub feature_contributions: HashMap<String, f32>,
    /// Detected anomaly types
    pub anomaly_types: Vec<String>,
}

impl DeviceHealthDetector {
    /// Create a new device health detector
    pub fn new(params: ModelParams) -> Result<Self> {
        let feature_names = vec![
            "cpu_usage".to_string(),
            "memory_usage".to_string(),
            "disk_io".to_string(),
            "network_traffic".to_string(),
            "error_rate".to_string(),
            "response_time".to_string(),
            "temperature".to_string(),
            "power_consumption".to_string(),
        ];

        Ok(Self {
            params,
            feature_names: feature_names.clone(),
            feature_stats: HashMap::new(),
            thresholds: HashMap::new(),
            is_trained: false,
            current_accuracy: 0.85,
            prediction_window: std::time::Duration::from_secs(3600),
            alert_threshold: 0.7,
            feature_weights: feature_names
                .into_iter()
                .map(|name| (name, 1.0 / 8.0)) // Equal weights initially
                .collect(),
            neural_network: None,
            history: ModelHistory::default(),
        })
    }

    /// Train the model with historical data
    pub fn train(&mut self, data: &[Vec<f32>]) -> Result<()> {
        if data.len() < self.params.min_samples {
            return Err(anyhow::anyhow!(
                "Insufficient training data: {} samples, need at least {}",
                data.len(),
                self.params.min_samples
            ));
        }

        if data.is_empty() || data[0].len() != self.feature_names.len() {
            return Err(anyhow::anyhow!(
                "Invalid data dimensions: expected {} features",
                self.feature_names.len()
            ));
        }

        // Calculate statistics for each feature
        for (i, feature_name) in self.feature_names.iter().enumerate() {
            let feature_data: Vec<f32> = data.iter().map(|sample| sample[i]).collect();
            let stats = FeatureStats::new(&feature_data);
            self.feature_stats.insert(feature_name.clone(), stats);
        }

        // Calculate adaptive thresholds based on contamination factor
        self.calculate_thresholds(data)?;

        self.is_trained = true;
        Ok(())
    }

    /// Calculate adaptive thresholds for anomaly detection
    fn calculate_thresholds(&mut self, data: &[Vec<f32>]) -> Result<()> {
        let mut all_scores = Vec::new();

        // Calculate anomaly scores for all training samples
        for sample in data {
            let score = self.calculate_sample_score(sample)?;
            all_scores.push(score);
        }

        // Sort scores and find threshold at contamination percentile
        all_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_index =
            ((1.0 - self.params.contamination) * all_scores.len() as f32) as usize;
        let adaptive_threshold = all_scores
            .get(threshold_index)
            .unwrap_or(&self.params.anomaly_threshold);

        // Set global threshold
        self.thresholds
            .insert("global".to_string(), *adaptive_threshold);

        // Set per-feature thresholds
        for feature_name in &self.feature_names {
            self.thresholds
                .insert(feature_name.clone(), self.params.outlier_threshold);
        }

        Ok(())
    }

    /// Calculate anomaly score for a single sample
    fn calculate_sample_score(&self, sample: &[f32]) -> Result<f32> {
        if sample.len() != self.feature_names.len() {
            return Err(anyhow::anyhow!("Invalid sample dimensions"));
        }

        let mut total_score = 0.0;
        let mut score_count = 0;

        for (i, feature_name) in self.feature_names.iter().enumerate() {
            if let Some(stats) = self.feature_stats.get(feature_name) {
                let value = sample[i];

                // Calculate multiple anomaly indicators
                let z_score = stats.z_score(value).abs();
                let modified_z_score = stats.modified_z_score(value).abs();

                // Combine scores with weights
                let feature_score =
                    (0.6 * z_score + 0.4 * modified_z_score) / self.params.outlier_threshold;
                total_score += feature_score.min(1.0); // Cap at 1.0
                score_count += 1;
            }
        }

        Ok(if score_count > 0 {
            total_score / score_count as f32
        } else {
            0.0
        })
    }

    /// Detect anomalies in device health metrics
    pub fn detect_anomalies(&self, metrics: &HashMap<String, f32>) -> Result<AnomalyResult> {
        if !self.is_trained {
            return Err(anyhow::anyhow!("Model not trained yet"));
        }

        // Convert metrics to feature vector
        let mut sample = Vec::new();
        let mut feature_scores = HashMap::new();
        let mut feature_contributions = HashMap::new();
        let mut anomaly_types = Vec::new();

        for feature_name in &self.feature_names {
            let value = metrics.get(feature_name).unwrap_or(&0.0);
            sample.push(*value);

            if let Some(stats) = self.feature_stats.get(feature_name) {
                let z_score = stats.z_score(*value);
                let modified_z_score = stats.modified_z_score(*value);

                // Calculate feature-specific anomaly score
                let feature_score = (0.6 * z_score.abs() + 0.4 * modified_z_score.abs())
                    / self.params.outlier_threshold;
                let normalized_score = feature_score.min(1.0);

                feature_scores.insert(feature_name.clone(), normalized_score);
                feature_contributions.insert(feature_name.clone(), z_score);

                // Detect specific anomaly types
                if z_score.abs() > self.params.outlier_threshold {
                    if z_score > 0.0 {
                        anomaly_types.push(format!("high_{}", feature_name));
                    } else {
                        anomaly_types.push(format!("low_{}", feature_name));
                    }
                }
            }
        }

        // Calculate overall anomaly score
        let overall_score = self.calculate_sample_score(&sample)?;
        let global_threshold = self
            .thresholds
            .get("global")
            .unwrap_or(&self.params.anomaly_threshold);
        let is_anomaly = overall_score >= *global_threshold;

        Ok(AnomalyResult {
            anomaly_score: overall_score,
            is_anomaly,
            feature_scores,
            feature_contributions,
            anomaly_types,
        })
    }

    /// Update model with new data (incremental learning)
    pub fn update(&mut self, new_data: &[Vec<f32>]) -> Result<()> {
        if !self.is_trained {
            return self.train(new_data);
        }

        // For now, retrain with new data
        // In production, implement proper incremental learning
        self.train(new_data)
    }

    /// Get model performance metrics
    pub fn get_performance_metrics(&self) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();

        metrics.insert("contamination".to_string(), self.params.contamination);
        metrics.insert(
            "outlier_threshold".to_string(),
            self.params.outlier_threshold,
        );
        metrics.insert(
            "is_trained".to_string(),
            if self.is_trained { 1.0 } else { 0.0 },
        );
        metrics.insert("num_features".to_string(), self.feature_names.len() as f32);

        if let Some(threshold) = self.thresholds.get("global") {
            metrics.insert("adaptive_threshold".to_string(), *threshold);
        }

        metrics
    }

    /// Save model state
    pub fn save(&self, path: &str) -> Result<()> {
        use std::fs;

        // Save model state as JSON
        let model_state = serde_json::json!({
            "model_type": "DeviceHealthDetector",
            "version": "1.0",
            "training_date": chrono::Utc::now().to_rfc3339(),
            "metrics": {
                "current_accuracy": self.current_accuracy,
                "prediction_window": self.prediction_window.as_secs(),
                "alert_threshold": self.alert_threshold,
            },
            "history": {
                "total_predictions": self.history.predictions.len(),
                "alerts_generated": self.history.alerts.len(),
            },
            "feature_weights": self.feature_weights,
        });

        let json_data = serde_json::to_string_pretty(&model_state)?;
        fs::write(path, json_data)?;

        // Save neural network weights if available
        let weights_path = format!("{}.nn_weights", path);
        if let Some(network) = &self.neural_network {
            let weights_data = bincode::serialize(&network.get_weights())?;
            fs::write(weights_path, weights_data)?;
        }

        Ok(())
    }

    /// Load model state
    pub fn load(&mut self, path: &str) -> Result<()> {
        use std::fs;
        use std::time::Duration;

        // Load model state from JSON
        let json_data = fs::read_to_string(path)?;
        let model_state: serde_json::Value = serde_json::from_str(&json_data)?;

        // Validate model type
        if model_state["model_type"] != "DeviceHealthDetector" {
            return Err(anyhow::anyhow!("Invalid model type"));
        }

        // Load metrics
        if let Some(metrics) = model_state["metrics"].as_object() {
            self.current_accuracy = metrics
                .get("current_accuracy")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.85);
            self.prediction_window = Duration::from_secs(
                metrics
                    .get("prediction_window")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3600),
            );
            self.alert_threshold = metrics
                .get("alert_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.7);
        }

        // Load feature weights
        if let Some(weights) = model_state["feature_weights"].as_object() {
            self.feature_weights = weights
                .iter()
                .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                .collect();
        }

        // Load neural network weights if they exist
        let weights_path = format!("{}.nn_weights", path);
        if std::path::Path::new(&weights_path).exists() {
            let weights_data = fs::read(weights_path)?;
            let weights: Vec<f64> = bincode::deserialize(&weights_data)?;
            if let Some(network) = &mut self.neural_network {
                network.set_weights(weights)?;
            }
        }

        Ok(())
    }
}
