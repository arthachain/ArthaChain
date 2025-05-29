use anyhow::Result;
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Device health anomaly detection model
pub struct DeviceHealthDetector {
    /// Isolation Forest model
    isolation_forest: PyObject,
    /// Feature names
    feature_names: Vec<String>,
    /// Model parameters
    params: ModelParams,
}

/// Model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParams {
    /// Number of estimators
    pub n_estimators: i32,
    /// Contamination factor
    pub contamination: f32,
    /// Random state
    pub random_state: i32,
    /// Threshold for anomaly score
    pub anomaly_threshold: f32,
}

impl DeviceHealthDetector {
    /// Create a new device health detector
    pub fn new(params: ModelParams) -> Result<Self> {
        let isolation_forest = Python::with_gil(|py| -> PyResult<PyObject> {
            // Import required Python modules
            let pyod = py.import_bound("pyod.models.iforest")?;

            // Initialize Isolation Forest model
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("n_estimators", params.n_estimators)?;
            kwargs.set_item("contamination", params.contamination)?;
            kwargs.set_item("random_state", params.random_state)?;

            let isolation_forest = pyod.getattr("IsolationForest")?.call((), Some(&kwargs))?;
            Ok(isolation_forest.into())
        })?;

        Ok(Self {
            isolation_forest,
            feature_names: vec![
                "cpu_usage".to_string(),
                "memory_usage".to_string(),
                "disk_io".to_string(),
                "network_traffic".to_string(),
                "error_rate".to_string(),
                "response_time".to_string(),
            ],
            params,
        })
    }

    /// Train the model with historical data
    pub fn train(&self, data: &[Vec<f32>]) -> Result<()> {
        Python::with_gil(|py| -> PyResult<()> {
            // Convert data to numpy array
            let data_array = PyArray2::from_vec2_bound(py, data)?;

            // Fit the model
            self.isolation_forest
                .call_method1(py, "fit", (data_array,))?;

            Ok(())
        })?;

        Ok(())
    }

    /// Detect anomalies in device health metrics
    pub fn detect_anomalies(&self, metrics: &HashMap<String, f32>) -> Result<AnomalyResult> {
        let (score, is_anomaly, contributions) =
            Python::with_gil(|py| -> PyResult<(f32, bool, HashMap<String, f32>)> {
                // Prepare input data
                let mut data_vec = Vec::new();
                for name in &self.feature_names {
                    data_vec.push(*metrics.get(name).unwrap_or(&0.0));
                }

                // Convert to numpy array
                let data_array = PyArray2::from_vec2_bound(py, &[data_vec.clone()])?;

                // Get anomaly scores
                let scores = self
                    .isolation_forest
                    .call_method1(py, "decision_function", (&data_array,))?
                    .extract::<Vec<f32>>(py)?;

                // Get predictions (1: normal, -1: anomaly)
                let predictions = self
                    .isolation_forest
                    .call_method1(py, "predict", (&data_array,))?
                    .extract::<Vec<i32>>(py)?;

                let score = scores[0];
                let is_anomaly = predictions[0] == -1;

                // Calculate feature contributions
                let contributions = self.calculate_feature_contributions(py, &data_vec)?;

                Ok((score, is_anomaly, contributions))
            })?;

        Ok(AnomalyResult {
            is_anomaly,
            anomaly_score: score,
            feature_contributions: contributions,
            threshold: self.params.anomaly_threshold,
        })
    }

    /// Calculate feature contributions to anomaly score
    fn calculate_feature_contributions(
        &self,
        py: Python,
        data: &[f32],
    ) -> PyResult<HashMap<String, f32>> {
        let mut contributions = HashMap::new();

        // Get feature importances from the model
        let importances = self
            .isolation_forest
            .call_method0(py, "get_feature_importances")?
            .extract::<Vec<f32>>(py)?;

        // Calculate contribution for each feature
        for (i, name) in self.feature_names.iter().enumerate() {
            let contribution = data[i] * importances[i];
            contributions.insert(name.clone(), contribution);
        }

        Ok(contributions)
    }
}

/// Result of anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Whether an anomaly was detected
    pub is_anomaly: bool,
    /// Anomaly score
    pub anomaly_score: f32,
    /// Contribution of each feature to the anomaly score
    pub feature_contributions: HashMap<String, f32>,
    /// Threshold used for detection
    pub threshold: f32,
}
