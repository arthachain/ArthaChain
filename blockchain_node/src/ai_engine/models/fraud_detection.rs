use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2};
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use log::{info, warn, debug};

/// ML-based fraud detection model using LightGBM
pub struct FraudDetectionModel {
    /// Python interpreter
    py_interpreter: Python<'static>,
    /// LightGBM model
    model: PyObject,
    /// Feature processor
    feature_processor: FeatureProcessor,
    /// Model parameters
    params: ModelParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    /// LightGBM parameters
    pub lgb_params: HashMap<String, String>,
    /// Feature importance threshold
    pub importance_threshold: f32,
    /// Prediction threshold
    pub prediction_threshold: f32,
    /// Number of trees
    pub num_trees: i32,
}

impl Default for ModelParams {
    fn default() -> Self {
        let mut lgb_params = HashMap::new();
        lgb_params.insert("objective".to_string(), "binary".to_string());
        lgb_params.insert("metric".to_string(), "auc".to_string());
        lgb_params.insert("boosting_type".to_string(), "gbdt".to_string());
        lgb_params.insert("num_leaves".to_string(), "31".to_string());
        lgb_params.insert("learning_rate".to_string(), "0.05".to_string());
        
        Self {
            lgb_params,
            importance_threshold: 0.05,
            prediction_threshold: 0.5,
            num_trees: 100,
        }
    }
}

pub struct FeatureProcessor {
    /// Feature names
    feature_names: Vec<String>,
    /// Feature normalizers
    normalizers: HashMap<String, Normalizer>,
}

#[derive(Debug, Clone)]
struct Normalizer {
    mean: f32,
    std: f32,
}

impl FraudDetectionModel {
    /// Create a new fraud detection model
    pub fn new(params: ModelParams) -> Result<Self> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // Import LightGBM
        let lgb = py.import("lightgbm")?;

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

        // Create model with parameters
        let kwargs = PyDict::new(py);
        for (key, value) in &params.lgb_params {
            kwargs.set_item(key, value)?;
        }
        
        let model = lgb.getattr("LGBMClassifier")?.call((), Some(kwargs))?;

        Ok(Self {
            py_interpreter: py,
            model: model.into(),
            feature_processor,
            params,
        })
    }

    /// Train the model with historical data
    pub fn train(&self, features: &[Vec<f32>], labels: &[bool]) -> Result<TrainingMetrics> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // Convert data to numpy arrays
        let x_train = PyArray2::from_vec2(py, features)?;
        let y_train = PyArray1::from_slice(py, labels)?;

        // Train model
        self.model.call_method1(
            py,
            "fit",
            (x_train, y_train),
        )?;

        // Get training metrics
        let train_score = self.model.call_method0(py, "score")?.extract::<f32>()?;
        
        // Get feature importance
        let importance = self.model
            .getattr(py, "feature_importances_")?
            .extract::<Vec<f32>>()?;

        Ok(TrainingMetrics {
            accuracy: train_score,
            feature_importance: self.feature_processor.feature_names.iter()
                .zip(importance.iter())
                .map(|(name, &imp)| (name.clone(), imp))
                .collect(),
        })
    }

    /// Predict fraud probability for new data
    pub fn predict(&self, features: &[f32]) -> Result<FraudPrediction> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        // Process features
        let processed = self.feature_processor.process_features(features)?;
        
        // Convert to numpy array
        let x = PyArray2::from_vec2(py, &[processed])?;

        // Get prediction probability
        let proba = self.model
            .call_method1(py, "predict_proba", (x,))?
            .extract::<Vec<Vec<f32>>>()?;

        let fraud_probability = proba[0][1]; // Probability of fraud class
        
        // Get feature contributions
        let contributions = self.calculate_feature_contributions(py, features)?;

        Ok(FraudPrediction {
            is_fraud: fraud_probability >= self.params.prediction_threshold,
            probability: fraud_probability,
            feature_contributions: contributions,
            threshold: self.params.prediction_threshold,
        })
    }

    /// Calculate feature contributions using SHAP values
    fn calculate_feature_contributions(
        &self,
        py: Python,
        features: &[f32],
    ) -> Result<HashMap<String, f32>> {
        let shap = py.import("shap")?;
        
        // Create explainer
        let explainer = shap.call_method1(
            "TreeExplainer",
            (self.model,)
        )?;

        // Get SHAP values
        let x = PyArray2::from_vec2(py, &[features.to_vec()])?;
        let shap_values = explainer.call_method1("shap_values", (x,))?;

        // Convert to feature contributions
        let contributions: Vec<f32> = shap_values
            .extract::<Vec<Vec<f32>>>()?
            .pop()
            .ok_or_else(|| anyhow!("Failed to get SHAP values"))?;

        Ok(self.feature_processor.feature_names.iter()
            .zip(contributions.iter())
            .map(|(name, &value)| (name.clone(), value))
            .collect())
    }
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Model accuracy
    pub accuracy: f32,
    /// Feature importance scores
    pub feature_importance: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct FraudPrediction {
    /// Whether transaction is fraudulent
    pub is_fraud: bool,
    /// Fraud probability
    pub probability: f32,
    /// Feature contributions
    pub feature_contributions: HashMap<String, f32>,
    /// Prediction threshold used
    pub threshold: f32,
}

impl FeatureProcessor {
    /// Process features for model input
    fn process_features(&self, features: &[f32]) -> Result<Vec<f32>> {
        if features.len() != self.feature_names.len() {
            return Err(anyhow!("Invalid feature count"));
        }

        let mut processed = Vec::new();
        for (i, &value) in features.iter().enumerate() {
            if let Some(normalizer) = self.normalizers.get(&self.feature_names[i]) {
                processed.push((value - normalizer.mean) / normalizer.std);
            } else {
                processed.push(value);
            }
        }
        
        Ok(processed)
    }
} 