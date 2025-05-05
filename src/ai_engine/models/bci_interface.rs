use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2};
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use super::neural_base::{NeuralBase, NeuralConfig};

/// Brain-Computer Interface model inspired by Neuralink
pub struct BCIModel {
    /// Base neural network
    neural_base: NeuralBase,
    /// Signal processing parameters
    signal_params: SignalParams,
    /// Spike detection model
    spike_detector: PyObject,
    /// Neural decoder
    decoder: PyObject,
}

/// Signal processing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalParams {
    /// Sampling rate in Hz
    pub sampling_rate: usize,
    /// Number of channels
    pub num_channels: usize,
    /// Filter parameters
    pub filter_params: FilterParams,
    /// Spike detection threshold
    pub spike_threshold: f32,
    /// Temporal window size
    pub window_size: usize,
}

/// Filter parameters for signal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterParams {
    /// Low-pass cutoff frequency
    pub low_cutoff: f32,
    /// High-pass cutoff frequency
    pub high_cutoff: f32,
    /// Filter order
    pub order: usize,
}

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

impl BCIModel {
    /// Create a new BCI model
    pub async fn new(config: NeuralConfig, signal_params: SignalParams) -> Result<Self> {
        // Create base neural network
        let neural_base = NeuralBase::new(config.clone()).await?;

        Python::with_gil(|py| {
            // Import required Python modules
            let torch = py.import("torch")?;
            let nn = py.import("torch.nn")?;

            // Create model instances
            let locals = PyDict::new(py);
            
            // Use include_str instead of CString for code
            let spike_detector_code = include_str!("spike_detector.py");
            let decoder_code = include_str!("decoder.py");
            
            py.run(spike_detector_code, None, Some(locals))?;
            py.run(decoder_code, None, Some(locals))?;

            let spike_detector = locals.get_item("SpikeDetector")
                .ok_or_else(|| anyhow!("Failed to get SpikeDetector class"))?
                .call1((
                    signal_params.num_channels,
                    signal_params.window_size,
                ))?;

            let decoder = locals.get_item("NeuralDecoder")
                .ok_or_else(|| anyhow!("Failed to get NeuralDecoder class"))?
                .call1((
                    signal_params.num_channels * signal_params.window_size,
                    config.output_dim,
                ))?;

            Ok(Self {
                neural_base,
                signal_params,
                spike_detector: spike_detector.into_py(py),
                decoder: decoder.into_py(py),
            })
        })
    }

    /// Process neural signals
    pub fn process(&self, signals: &[f32]) -> Result<BCIOutput> {
        Python::with_gil(|py| {
            // Convert signals to PyTorch tensor
            let x = PyArray1::from_slice(py, signals)?;
            
            // Process with spike detector
            let _spikes = self.spike_detector.call_method1(py, "forward", (x.clone(),))?;
            
            // Process with decoder
            let intent = self.decoder.call_method1(py, "forward", (x,))?;
            let intent: Vec<f32> = intent.extract(py)?;
            
            // Calculate confidence (placeholder)
            let confidence = 0.95;
            
            Ok(BCIOutput {
                intent,
                confidence,
                spikes: vec![],
                latency: 5.0,
            })
        })
    }

    /// Train the model
    pub async fn train(&mut self, training_data: &[(Vec<f32>, Vec<f32>)]) -> Result<f32> {
        // First train the neural base
        let metrics = self.neural_base.train(training_data.len()).await?;
        
        Python::with_gil(|py| {
            // Prepare training data
            let inputs: Vec<_> = training_data.iter()
                .map(|(x, _)| x.clone())
                .collect();
            let targets: Vec<_> = training_data.iter()
                .map(|(_, y)| y.clone())
                .collect();

            // Convert to PyTorch tensors
            let x = PyArray2::from_vec2(py, &inputs)?;
            let y = PyArray2::from_vec2(py, &targets)?;

            // Train spike detector
            let spike_loss: f32 = self.spike_detector.call_method1(
                py,
                "train",
                (x.clone(), y.clone())
            )?.extract(py)?;

            // Train decoder
            let decoder_loss: f32 = self.decoder.call_method1(
                py,
                "train",
                (x, y)
            )?.extract(py)?;

            Ok((spike_loss + decoder_loss) / 2.0)
        })
    }

    /// Save model state
    pub async fn save(&self, path: &str) -> Result<()> {
        // Save the neural base first
        self.neural_base.save(path)?;
        
        Python::with_gil(|py| {
            let torch = py.import("torch")?;
            
            // Save spike detector
            torch.call_method1(
                "save",
                (
                    self.spike_detector.call_method0(py, "state_dict")?,
                    format!("{}_spike_detector.pt", path)
                )
            )?;

            // Save decoder
            torch.call_method1(
                "save",
                (
                    self.decoder.call_method0(py, "state_dict")?,
                    format!("{}_decoder.pt", path)
                )
            )?;

            Ok(())
        })
    }

    /// Load model state
    pub async fn load(&mut self, path: &str) -> Result<()> {
        // Load the neural base first
        self.neural_base.load(path)?;
        
        Python::with_gil(|py| {
            let torch = py.import("torch")?;
            
            // Load spike detector
            let state_dict = torch.call_method1("load", (format!("{}_spike_detector.pt", path),))?;
            self.spike_detector.call_method1(py, "load_state_dict", (state_dict,))?;
            
            // Load decoder
            let state_dict = torch.call_method1("load", (format!("{}_decoder.pt", path),))?;
            self.decoder.call_method1(py, "load_state_dict", (state_dict,))?;
            
            Ok(())
        })
    }
} 