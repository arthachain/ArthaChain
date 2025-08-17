use anyhow::Result;
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use smartcore::tree::decision_tree_classifier::SplitCriterion;
use std::collections::HashMap;

/// Pure Rust Fraud Detection Model
pub struct FraudDetectionModel {
    /// Random Forest classifier
    model: Option<RandomForestClassifier<f32, i32, DenseMatrix<f32>, Vec<i32>>>,
    /// Feature processor
    feature_processor: FeatureProcessor,
    /// Model parameters
    params: ModelParams,
    /// Model features
    features: Vec<String>,
    /// Model weights (for custom implementations)
    weights: Option<Vec<f32>>,
    /// Model performance metrics
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1_score: f64,
}

/// Model parameters for fraud detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    /// Number of trees in the forest
    pub n_estimators: u16,
    /// Maximum depth of trees
    pub max_depth: Option<u16>,
    /// Minimum samples required to split
    pub min_samples_split: usize,
    /// Minimum samples required at leaf
    pub min_samples_leaf: usize,
    /// Random state for reproducibility
    pub random_state: u64,
    /// Prediction threshold
    pub prediction_threshold: f32,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: Some(10),
            min_samples_split: 2,
            min_samples_leaf: 1,
            random_state: 42,
            prediction_threshold: 0.5,
        }
    }
}

/// Feature processor for preprocessing input data
#[derive(Debug, Clone)]
pub struct FeatureProcessor {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Feature normalizers (min, max) for each feature
    pub normalizers: HashMap<String, (f32, f32)>,
}

impl FeatureProcessor {
    /// Process features for model input
    pub fn process_features(&self, features: &[f32]) -> Result<Vec<f32>> {
        if features.len() != self.feature_names.len() {
            return Err(anyhow::anyhow!(
                "Feature count mismatch: expected {}, got {}",
                self.feature_names.len(),
                features.len()
            ));
        }

        let mut processed = Vec::new();
        for (i, &value) in features.iter().enumerate() {
            let feature_name = &self.feature_names[i];
            if let Some((min_val, max_val)) = self.normalizers.get(feature_name) {
                // Normalize to [0, 1]
                let normalized = if max_val - min_val > 0.0 {
                    (value - min_val) / (max_val - min_val)
                } else {
                    0.0
                };
                processed.push(normalized.clamp(0.0, 1.0));
            } else {
                processed.push(value);
            }
        }

        Ok(processed)
    }

    /// Calculate feature importance (simplified)
    pub fn calculate_feature_importance(&self, _features: &[f32]) -> HashMap<String, f32> {
        // Simplified feature importance - in real implementation this would
        // come from the trained model
        let mut importance = HashMap::new();
        let base_importance = 1.0 / self.feature_names.len() as f32;

        for name in &self.feature_names {
            importance.insert(name.clone(), base_importance);
        }

        importance
    }
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Model accuracy
    pub accuracy: f32,
    /// Feature importance scores
    pub feature_importance: HashMap<String, f32>,
}

/// Fraud prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudPrediction {
    /// Whether the transaction is predicted as fraud
    pub is_fraud: bool,
    /// Probability of fraud (0.0 to 1.0)
    pub probability: f32,
    /// Feature contributions to the prediction
    pub feature_contributions: HashMap<String, f32>,
    /// Threshold used for classification
    pub threshold: f32,
}

impl FraudDetectionModel {
    /// Create a new fraud detection model
    pub fn new(params: ModelParams) -> Result<Self> {
        // Initialize feature processor
        let feature_processor = FeatureProcessor {
            feature_names: vec![
                "transaction_amount".to_string(),
                "transaction_frequency".to_string(),
                "device_reputation".to_string(),
                "network_trust".to_string(),
                "historical_behavior".to_string(),
                "geographical_risk".to_string(),
                "time_pattern".to_string(),
                "peer_reputation".to_string(),
            ],
            normalizers: HashMap::new(),
        };

        Ok(Self {
            model: None,
            feature_processor,
            params,
            features: vec![
                "transaction_amount".to_string(),
                "transaction_frequency".to_string(),
                "device_reputation".to_string(),
                "network_trust".to_string(),
                "historical_behavior".to_string(),
                "geographical_risk".to_string(),
                "time_pattern".to_string(),
                "peer_reputation".to_string(),
            ],
            weights: None,
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
        })
    }

    /// Train the model with historical data
    pub fn train(&mut self, features: &[Vec<f32>], labels: &[bool]) -> Result<TrainingMetrics> {
        if features.is_empty() || labels.is_empty() {
            return Err(anyhow::anyhow!("Training data cannot be empty"));
        }

        if features.len() != labels.len() {
            return Err(anyhow::anyhow!("Features and labels must have same length"));
        }

        // Convert data to smartcore format
        let mut data_matrix = Vec::new();
        for feature_vec in features {
            data_matrix.extend_from_slice(feature_vec);
        }

        let x = DenseMatrix::from_2d_vec(&features.iter().map(|v| v.clone()).collect::<Vec<_>>());
        let y: Vec<i32> = labels.iter().map(|&b| if b { 1 } else { 0 }).collect();

        // Split data for validation
        let (x_train, x_test, y_train, y_test) =
            train_test_split(&x, &y, 0.2, true, Some(self.params.random_state));

        // Train Random Forest model
        let model = RandomForestClassifier::fit(
            &x_train,
            &y_train,
            smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
                .with_n_trees(self.params.n_estimators)
                .with_max_depth(self.params.max_depth.unwrap_or(u16::MAX))
                .with_min_samples_split(self.params.min_samples_split)
                .with_min_samples_leaf(self.params.min_samples_leaf)
                .with_criterion(SplitCriterion::Gini)
                .with_seed(self.params.random_state),
        )?;

        // Evaluate model
        let y_pred = model.predict(&x_test)?;
        let train_accuracy = accuracy(&y_test, &y_pred);

        // Store the trained model
        self.model = Some(model);

        // Update model metrics
        self.accuracy = train_accuracy;
        // Calculate precision, recall, and f1 (simplified for now)
        let true_positives = y_test
            .iter()
            .zip(&y_pred)
            .filter(|(&actual, &predicted)| actual == 1 && predicted == 1)
            .count() as f64;
        let false_positives = y_test
            .iter()
            .zip(&y_pred)
            .filter(|(&actual, &predicted)| actual == 0 && predicted == 1)
            .count() as f64;
        let false_negatives = y_test
            .iter()
            .zip(&y_pred)
            .filter(|(&actual, &predicted)| actual == 1 && predicted == 0)
            .count() as f64;

        self.precision = if true_positives + false_positives > 0.0 {
            true_positives / (true_positives + false_positives)
        } else {
            0.0
        };

        self.recall = if true_positives + false_negatives > 0.0 {
            true_positives / (true_positives + false_negatives)
        } else {
            0.0
        };

        self.f1_score = if self.precision + self.recall > 0.0 {
            2.0 * (self.precision * self.recall) / (self.precision + self.recall)
        } else {
            0.0
        };

        // Calculate feature importance (simplified)
        let feature_importance = self
            .feature_processor
            .calculate_feature_importance(&features[0]);

        Ok(TrainingMetrics {
            accuracy: train_accuracy as f32,
            feature_importance,
        })
    }

    /// Predict fraud probability for new data
    pub fn predict(&self, features: &[f32]) -> Result<FraudPrediction> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not trained yet"))?;

        // Process features
        let processed = self.feature_processor.process_features(features)?;

        // Convert to matrix format
        let x = DenseMatrix::from_2d_vec(&vec![processed.clone()]);

        // Get prediction
        let prediction = model.predict(&x)?;
        let is_fraud = prediction[0] == 1;

        // For probability, we simulate it based on the prediction
        // In a real implementation, this would use predict_proba if available
        let probability = if is_fraud {
            0.7 + (processed[0] * 0.3) // Simplified probability calculation
        } else {
            0.3 - (processed[0] * 0.3)
        }
        .clamp(0.0, 1.0);

        // Calculate feature contributions
        let contributions = self
            .feature_processor
            .calculate_feature_importance(&processed);

        Ok(FraudPrediction {
            is_fraud: probability >= self.params.prediction_threshold,
            probability,
            feature_contributions: contributions,
            threshold: self.params.prediction_threshold,
        })
    }

    /// Update model with new data (incremental learning)
    pub fn update(&mut self, features: &[Vec<f32>], labels: &[bool]) -> Result<()> {
        // For Random Forest, we retrain the entire model
        // In production, you might want to use online learning algorithms
        self.train(features, labels)?;
        Ok(())
    }

    /// Save model to file
    pub fn save(&self, path: &str) -> Result<()> {
        use std::fs;

        // Save model metadata as JSON (since smartcore models aren't directly serializable)
        let model_metadata = serde_json::json!({
            "model_type": "FraudDetectionModel",
            "version": "1.0",
            "training_date": chrono::Utc::now().to_rfc3339(),
            "params": {
                "n_estimators": self.params.n_estimators,
                "max_depth": self.params.max_depth,
                "min_samples_split": self.params.min_samples_split,
                "min_samples_leaf": self.params.min_samples_leaf,
                "random_state": self.params.random_state,
                "prediction_threshold": self.params.prediction_threshold,
            },
            "features": self.features,
            "performance": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
            },
            "weights": self.weights,
            "feature_processor": {
                "feature_names": self.feature_processor.feature_names,
            }
        });

        let json_data = serde_json::to_string_pretty(&model_metadata)?;
        fs::write(path, json_data)?;

        // Save model weights if available
        if let Some(weights) = &self.weights {
            let weights_path = format!("{}.weights", path);
            let weights_data = bincode::serialize(weights)?;
            fs::write(weights_path, weights_data)?;
        }

        Ok(())
    }

    /// Load model from file
    pub fn load(&mut self, path: &str) -> Result<()> {
        use std::fs;

        // Load model metadata from JSON
        let json_data = fs::read_to_string(path)?;
        let model_metadata: serde_json::Value = serde_json::from_str(&json_data)?;

        // Validate model type
        if model_metadata["model_type"] != "FraudDetectionModel" {
            return Err(anyhow::anyhow!("Invalid model type"));
        }

        // Load parameters
        if let Some(params) = model_metadata["params"].as_object() {
            self.params.n_estimators = params
                .get("n_estimators")
                .and_then(|v| v.as_u64())
                .unwrap_or(100) as u16;
            self.params.max_depth = params
                .get("max_depth")
                .and_then(|v| v.as_u64())
                .map(|v| v as u16);
            self.params.min_samples_split = params
                .get("min_samples_split")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            self.params.min_samples_leaf = params
                .get("min_samples_leaf")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            self.params.random_state = params
                .get("random_state")
                .and_then(|v| v.as_u64())
                .unwrap_or(42);
            self.params.prediction_threshold = params
                .get("prediction_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32;
        }

        // Load features
        if let Some(features) = model_metadata["features"].as_array() {
            self.features = features
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }

        // Load performance metrics
        if let Some(performance) = model_metadata["performance"].as_object() {
            self.accuracy = performance
                .get("accuracy")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            self.precision = performance
                .get("precision")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            self.recall = performance
                .get("recall")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            self.f1_score = performance
                .get("f1_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
        }

        // Load weights if they exist
        let weights_path = format!("{}.weights", path);
        if std::path::Path::new(&weights_path).exists() {
            let weights_data = fs::read(weights_path)?;
            let weights: Vec<f32> = bincode::deserialize(&weights_data)?;
            self.weights = Some(weights);
        }

        // Load feature processor names
        if let Some(processor) = model_metadata["feature_processor"].as_object() {
            if let Some(feature_names) = processor.get("feature_names").and_then(|v| v.as_array()) {
                self.feature_processor.feature_names = feature_names
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
            }
        }

        // Note: The actual RandomForest model cannot be deserialized directly
        // In production, you would need to retrain or use a different serialization approach
        self.model = None; // Will need retraining

        Ok(())
    }

    /// Get model performance metrics
    pub fn get_metrics(&self) -> HashMap<String, f32> {
        let mut metrics = HashMap::new();
        metrics.insert("threshold".to_string(), self.params.prediction_threshold);
        metrics.insert("n_estimators".to_string(), self.params.n_estimators as f32);

        if let Some(max_depth) = self.params.max_depth {
            metrics.insert("max_depth".to_string(), max_depth as f32);
        }

        metrics
    }
}
