use super::neural_base::{NeuralBase, NeuralConfig, NeuralNetwork};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::RwLock;

/// Parameters for BCI signal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalParams {
    /// Sampling rate in Hz
    pub sampling_rate: usize,
    /// Number of EEG channels
    pub num_channels: usize,
    /// Window size in samples
    pub window_size: usize,
    /// Filter parameters
    pub filter_params: FilterParams,
    /// Threshold for spike detection
    pub spike_threshold: f32,
    /// Whether to normalize signals
    pub normalize: bool,
    /// Whether to use wavelet transform
    pub use_wavelet: bool,
}

/// Signal filtering parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterParams {
    /// Low-cut frequency in Hz
    pub low_cut: f32,
    /// High-cut frequency in Hz
    pub high_cut: f32,
    /// Filter order
    pub order: usize,
}

/// Brain-Computer Interface model output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BCIOutput {
    /// Decoded intent
    pub intent: Vec<f32>,
    /// Confidence score
    pub confidence: f32,
    /// Detected spikes
    pub spikes: Vec<Spike>,
    /// Latency in milliseconds
    pub latency: f32,
}

/// Neural spike data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spike {
    /// Channel index
    pub channel: usize,
    /// Timestamp in milliseconds
    pub timestamp: f32,
    /// Amplitude
    pub amplitude: f32,
    /// Waveform shape
    pub waveform: Vec<f32>,
}

/// Brain-Computer Interface model
pub struct BCIModel {
    /// Neural base model
    neural_base: Arc<RwLock<Box<dyn NeuralNetwork>>>,
    /// Signal processing parameters
    signal_params: RefCell<SignalParams>,
    /// Processed signal buffer
    signal_buffer: Vec<Vec<f32>>,
    /// Current state
    current_state: RefCell<BCIState>,
    #[allow(dead_code)]
    neural_net: Arc<RwLock<NeuralBase>>,
    #[allow(dead_code)]
    feature_cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}

/// BCI processing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BCIState {
    /// Current filtering state
    pub filter_state: Vec<f32>,
    /// Spike timestamps
    pub spike_timestamps: Vec<u64>,
    /// Feature vectors
    pub features: Vec<Vec<f32>>,
    /// Classification results
    pub classifications: Vec<usize>,
}

/// Initialization trait
pub trait Initialize {
    fn initialize(&self, config: &NeuralConfig) -> Result<()>;
}

/// Serializable BCIModel state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BCIModelState {
    signal_params: SignalParams,
    current_state: BCIState,
}

impl BCIModel {
    /// Create a new BCI model
    pub async fn new(config: NeuralConfig, signal_params: SignalParams) -> Result<Self> {
        // Create neural base with a clone of the config
        let neural_base = NeuralBase::new(config.clone()).await?;

        Ok(Self {
            neural_base: Arc::new(RwLock::new(Box::new(neural_base))),
            signal_params: RefCell::new(signal_params.clone()),
            signal_buffer: Vec::new(),
            current_state: RefCell::new(BCIState {
                filter_state: vec![0.0; signal_params.num_channels * 4],
                spike_timestamps: Vec::new(),
                features: Vec::new(),
                classifications: Vec::new(),
            }),
            neural_net: Arc::new(RwLock::new(NeuralBase::new(config).await?)),
            feature_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Process a new batch of EEG signals
    pub async fn process_signals(&mut self, signals: &[Vec<f32>]) -> Result<Vec<usize>> {
        // Add signals to buffer
        self.signal_buffer.extend(signals.iter().cloned());

        // Trim buffer to maximum size
        let max_buffer_size = self.signal_params.borrow().window_size * 3;
        if self.signal_buffer.len() > max_buffer_size {
            self.signal_buffer = self
                .signal_buffer
                .split_off(self.signal_buffer.len() - max_buffer_size);
        }

        // Check if we have enough data
        if self.signal_buffer.len() < self.signal_params.borrow().window_size {
            return Ok(Vec::new());
        }

        // Extract features
        let features = self.extract_features()?;

        // Classify features
        let neural_base = self.neural_base.read().await;
        let mut classifications = Vec::new();

        for feature in features.clone() {
            let output = neural_base.forward(&feature);
            let class = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            classifications.push(class);
            self.current_state.borrow_mut().classifications.push(class);
        }

        Ok(classifications)
    }

    /// Extract features from signal buffer
    fn extract_features(&mut self) -> Result<Vec<Vec<f32>>> {
        let window_size = self.signal_params.borrow().window_size;
        let mut features = Vec::new();

        // Process each window
        for window_start in (0..self.signal_buffer.len()).step_by(window_size / 2) {
            if window_start + window_size > self.signal_buffer.len() {
                break;
            }

            // Extract window
            let window: Vec<Vec<f32>> = self.signal_buffer
                [window_start..(window_start + window_size)]
                .iter()
                .cloned()
                .collect();

            // Apply filtering
            let filtered = self.apply_filter(&window)?;

            // Detect spikes
            self.detect_spikes(&filtered, window_start as u64);

            // Extract features from filtered signal
            let feature = self.compute_features(&filtered)?;
            features.push(feature.clone());

            // Store feature
            self.current_state.borrow_mut().features.push(feature);
        }

        Ok(features)
    }

    /// Apply filter to signal window
    fn apply_filter(&mut self, window: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        // Simple passthrough for now
        // In a real implementation, we would apply bandpass filtering here
        let filtered = window.to_vec();

        Ok(filtered)
    }

    /// Detect spikes in filtered signal
    fn detect_spikes(&mut self, filtered: &[Vec<f32>], start_time: u64) {
        for (_i, channel) in filtered.iter().enumerate() {
            for (j, &sample) in channel.iter().enumerate() {
                if sample.abs() > self.signal_params.borrow().spike_threshold {
                    let timestamp = start_time + j as u64;
                    self.current_state
                        .borrow_mut()
                        .spike_timestamps
                        .push(timestamp);
                }
            }
        }
    }

    /// Compute features from filtered signal
    fn compute_features(&self, filtered: &[Vec<f32>]) -> Result<Vec<f32>> {
        if filtered.is_empty() {
            return Err(anyhow!("Empty filtered signal"));
        }

        // Calculate basic features
        let mut features = Vec::new();

        // For each channel
        for channel in filtered.iter() {
            // Mean
            let mean = channel.iter().sum::<f32>() / channel.len() as f32;
            features.push(mean);

            // Standard deviation
            let var =
                channel.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / channel.len() as f32;
            let std_dev = var.sqrt();
            features.push(std_dev);

            // Max amplitude
            let max_amp = channel.iter().fold(0.0f32, |max, &x| max.max(x.abs()));
            features.push(max_amp);
        }

        Ok(features)
    }

    /// Train model on feedback data
    pub async fn train(&mut self, data: Vec<(Vec<f32>, usize)>) -> Result<()> {
        let mut neural_base = self.neural_base.write().await;

        // Convert data to format expected by neural_base
        let training_data: Vec<(Vec<f32>, Vec<f32>)> = data
            .into_iter()
            .map(|(input, target)| {
                let mut target_vec = vec![0.0; 10]; // Assuming 10 classes
                if target < target_vec.len() {
                    target_vec[target] = 1.0;
                }
                (input, target_vec)
            })
            .collect();

        // Train neural base
        neural_base.train(&training_data)?;

        Ok(())
    }

    /// Save model to file
    pub async fn save(&self, path: &str) -> Result<()> {
        // Create directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Save neural base
        {
            let neural_base = self.neural_base.read().await;
            neural_base.save(&format!("{}/neural_base.pt", path))?;
        }

        // Save signal params and current state
        let state = BCIModelState {
            signal_params: self.signal_params.borrow().clone(),
            current_state: self.current_state.borrow().clone(),
        };

        let serialized = serde_json::to_string_pretty(&state)?;
        std::fs::write(&format!("{}/state.json", path), serialized)?;

        Ok(())
    }

    /// Load model from file
    pub async fn load(&self, path: &str) -> Result<()> {
        // Load neural base
        {
            let mut neural_base = self.neural_base.write().await;
            neural_base.load(&format!("{}/neural_base.pt", path))?;
        }

        // Load signal params and current state
        let state_path = format!("{}/state.json", path);
        if std::path::Path::new(&state_path).exists() {
            let serialized = std::fs::read_to_string(&state_path)?;
            let state: BCIModelState = serde_json::from_str(&serialized)?;

            // Update state using interior mutability via RefCell
            *self.signal_params.borrow_mut() = state.signal_params;
            *self.current_state.borrow_mut() = state.current_state;
        }

        Ok(())
    }

    /// Get a serializable state for persistence
    pub fn get_serializable_state(&self) -> BCIModelState {
        BCIModelState {
            signal_params: self.signal_params.borrow().clone(),
            current_state: self.current_state.borrow().clone(),
        }
    }

    /// Restore from a serializable state
    pub fn restore_from_state(&mut self, state: BCIModelState) {
        *self.signal_params.borrow_mut() = state.signal_params;
        *self.current_state.borrow_mut() = state.current_state;
    }
}
