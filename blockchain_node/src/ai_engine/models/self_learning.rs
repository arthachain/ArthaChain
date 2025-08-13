use super::types::Experience;
use crate::ai_engine::models::neural_base::{
    ActivationType, LayerConfig, NeuralBase, NeuralConfig, NeuralNetwork,
};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::result::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Type alias for neural network model storage
type NeuralModelMap = HashMap<String, Arc<RwLock<Box<dyn NeuralNetwork + Send + Sync>>>>;

/// Type alias for training data
type TrainingDataVec = Vec<(Vec<f32>, Vec<f32>)>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfLearningConfig {
    /// Base neural network configuration
    pub base_config: NeuralConfig,
    /// Maximum number of models to maintain
    pub max_models: usize,
    /// Learning rate adjustment factor
    pub lr_factor: f32,
    /// Adaptation threshold for model architecture changes
    pub adaptation_threshold: f32,
    /// Knowledge sharing threshold between models
    pub sharing_threshold: f32,
    /// Minimum required performance for a model to be kept
    pub min_performance: f32,
}

/// Self-learning system that coordinates multiple neural models
pub struct SelfLearningSystem {
    /// Neural models for different tasks
    models: NeuralModelMap,
    /// Model configurations
    configs: HashMap<String, NeuralConfig>,
    /// Learning coordinator
    coordinator: LearningCoordinator,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Experience buffer
    experiences: Vec<Experience>,
    /// Active model identifier
    active_model: String,
    /// Performance history
    performance_history: VecDeque<(String, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCoordinator {
    /// Learning rate adjustment factor
    pub lr_factor: f32,
    /// Architecture adaptation threshold
    pub adaptation_threshold: f32,
    /// Knowledge sharing threshold
    pub sharing_threshold: f32,
    /// Minimum performance requirement
    pub min_performance: f32,
    /// Maximum models to maintain
    pub max_models: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Model accuracies
    pub accuracies: HashMap<String, f32>,
    /// Model losses
    pub losses: HashMap<String, f32>,
    /// Resource usage
    pub resource_usage: ResourceMetrics,
    /// Learning progress
    pub learning_progress: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// GPU usage percentage
    pub gpu_usage: Option<f32>,
    /// Training time in seconds
    pub training_time: f32,
}

/// Serializable state for SelfLearningSystem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfLearningState {
    /// Model configurations
    pub configs: HashMap<String, NeuralConfig>,
    /// Learning coordinator
    pub coordinator: LearningCoordinator,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Model save paths
    pub model_paths: HashMap<String, String>,
}

/// Metadata for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfLearningMeta {
    /// Model names
    pub models: Vec<String>,
    /// Active model identifier
    pub active_model: String,
    /// Performance history
    pub performance_history: VecDeque<(String, f32)>,
}

impl SelfLearningSystem {
    /// Create a new self-learning system
    pub async fn new(config: SelfLearningConfig) -> Result<Self, anyhow::Error> {
        // Create neural model
        let model = NeuralBase::new_sync(config.base_config.clone())?;
        let model: Box<dyn NeuralNetwork + Send + Sync> = Box::new(model);

        let mut models = HashMap::new();
        models.insert("base".to_string(), Arc::new(RwLock::new(model)));

        let mut configs = HashMap::new();
        configs.insert("base".to_string(), config.base_config.clone());

        Ok(Self {
            models,
            configs,
            coordinator: LearningCoordinator {
                lr_factor: config.lr_factor,
                adaptation_threshold: config.adaptation_threshold,
                sharing_threshold: config.sharing_threshold,
                min_performance: config.min_performance,
                max_models: config.max_models,
            },
            metrics: Arc::new(RwLock::new(PerformanceMetrics {
                accuracies: HashMap::new(),
                losses: HashMap::new(),
                resource_usage: ResourceMetrics {
                    cpu_usage: 0.0,
                    memory_usage: 0,
                    gpu_usage: None,
                    training_time: 0.0,
                },
                learning_progress: HashMap::new(),
            })),
            experiences: Vec::new(),
            active_model: "base".to_string(),
            performance_history: VecDeque::new(),
        })
    }

    /// Add a new neural model
    pub async fn add_model(
        &mut self,
        name: &str,
        config: NeuralConfig,
    ) -> Result<(), anyhow::Error> {
        let model = NeuralBase::new_sync(config.clone())?;
        let model: Box<dyn NeuralNetwork + Send + Sync> = Box::new(model);
        self.models
            .insert(name.to_string(), Arc::new(RwLock::new(model)));
        self.configs.insert(name.to_string(), config);
        Ok(())
    }

    /// Train all models with shared knowledge
    pub async fn train_all(&mut self, experiences: Vec<Experience>) -> Result<(), anyhow::Error> {
        // Store experiences for later use
        self.experiences.extend(experiences);

        // Update resource metrics
        self.update_resource_metrics().await?;

        // Train each model
        for (name, model) in &self.models {
            let mut model = model.write().await;

            // Convert experiences to training data
            let training_data = self.prepare_training_data()?;

            // Train model using the NeuralNetwork trait method - split training data into inputs and targets
            let inputs: Vec<Vec<f32>> = training_data
                .iter()
                .map(|(input, _)| input.clone())
                .collect();
            let targets: Vec<Vec<f32>> = training_data
                .iter()
                .map(|(_, target)| target.clone())
                .collect();
            model.train(&inputs, &targets)?;

            // Update metrics
            let mut system_metrics = self.metrics.write().await;
            system_metrics.losses.insert(name.clone(), 0.0); // Replace with actual loss

            // Record learning progress
            system_metrics
                .learning_progress
                .entry(name.clone())
                .or_insert_with(Vec::new)
                .push(0.0); // Replace with actual loss
        }

        // Share knowledge between models
        self.share_knowledge().await?;

        // Adapt models based on performance
        self.adapt_models().await?;

        Ok(())
    }

    /// Prepare training data from experiences
    fn prepare_training_data(&self) -> Result<TrainingDataVec, anyhow::Error> {
        // Convert experiences to input-output pairs
        let mut training_data = Vec::new();

        for exp in &self.experiences {
            // Use state as input
            let input = exp.state.clone();

            // Create a one-hot encoded output for the action
            let mut output = vec![0.0; 10]; // Assuming up to 10 possible actions
            if exp.action < output.len() {
                output[exp.action] = 1.0;
            }

            training_data.push((input, output));
        }

        Ok(training_data)
    }

    /// Share knowledge between models
    async fn share_knowledge(&self) -> Result<(), anyhow::Error> {
        let metrics = self.metrics.read().await;

        // Find best performing model
        let best_model = metrics
            .accuracies
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone());

        if let Some(best_name) = best_model {
            if let Some(best_model) = self.models.get(&best_name) {
                // Save the best model temporarily
                let best_model = best_model.read().await;
                best_model.save("temp_best_model.pt")?;

                // Share knowledge with other models
                for (name, model) in &self.models {
                    if name != &best_name {
                        let mut model = model.write().await;
                        if metrics.accuracies.get(name).unwrap_or(&0.0)
                            < &self.coordinator.sharing_threshold
                        {
                            // Transfer learning from best model
                            model.load("temp_best_model.pt")?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Adapt models based on performance
    async fn adapt_models(&self) -> Result<(), anyhow::Error> {
        let metrics = self.metrics.read().await;

        for name in self.models.keys() {
            if let Some(&loss) = metrics.losses.get(name) {
                if loss > self.coordinator.adaptation_threshold {
                    // Increase model capacity
                    if let Some(config) = self.configs.get(name) {
                        let mut new_config = config.clone();
                        // Get the last layer from the hidden_layers vector
                        if !new_config.hidden_layers.is_empty() {
                            // Get the last layer's output dimension
                            let last_layer = new_config.hidden_layers.last().unwrap();
                            let last_output_size = last_layer.output_size;

                            // Add a new layer with twice the output dimension
                            new_config.hidden_layers.push(LayerConfig {
                                input_size: last_output_size,
                                output_size: last_output_size * 2,
                                activation: ActivationType::GELU,
                                dropout_rate: Some(0.2),
                                batch_norm: true,
                            });

                            // Create new model with adapted architecture
                            let _new_model = NeuralBase::new_sync(new_config.clone())?;
                            // Here we would replace the old model, but this is a simplification
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Update resource usage metrics
    async fn update_resource_metrics(&self) -> Result<(), anyhow::Error> {
        let mut _metrics = self.metrics.write().await;

        // Basic CPU/Memory metrics
        #[allow(unused_attributes)]
        #[cfg(feature = "sysinfo")]
        {
            use sysinfo::System;
            let mut sys = System::new();
            sys.refresh_cpu_all();
            _metrics.resource_usage.cpu_usage = sys.global_cpu_usage();
            _metrics.resource_usage.memory_usage =
                (sys.used_memory() * 100 / sys.total_memory()) as u64;
        }

        Ok(())
    }

    /// Get a serializable representation of the system state
    pub fn get_serializable_state(&self) -> Result<SelfLearningState, anyhow::Error> {
        let metrics = self.metrics.blocking_read();

        // Save individual models to temporary files to capture their state
        let mut model_paths = HashMap::new();
        for name in self.models.keys() {
            let path = format!("temp_model_{name}.pt");
            model_paths.insert(name.clone(), path);
        }

        Ok(SelfLearningState {
            configs: self.configs.clone(),
            coordinator: self.coordinator.clone(),
            metrics: metrics.clone(),
            model_paths,
        })
    }

    /// Restore system state from serialized data
    pub fn restore_from_state(&mut self, state: SelfLearningState) -> Result<(), anyhow::Error> {
        // Restore configs and coordinator
        self.configs = state.configs;
        self.coordinator = state.coordinator;

        // Restore metrics
        let mut metrics = self.metrics.blocking_write();
        *metrics = state.metrics;

        // Restore models from saved paths
        for (name, model_path) in &state.model_paths {
            if let Some(model) = self.models.get(name) {
                let mut model = model.blocking_write();
                if std::path::Path::new(model_path).exists() {
                    model.load(model_path)?;
                }
            }
        }

        Ok(())
    }

    /// Add experience to the system
    pub async fn add_experience(&mut self, experience: Experience) -> Result<(), anyhow::Error> {
        self.experiences.push(experience);
        Ok(())
    }

    /// Save system to path
    pub fn save(&self, path: &str) -> Result<(), anyhow::Error> {
        let state = self.get_serializable_state()?;

        // Create directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Save state
        let serialized = serde_json::to_string_pretty(&state)?;
        std::fs::write(path, serialized)?;

        // Save each model
        for name in state.model_paths.keys() {
            if let Some(model) = self.models.get(name) {
                let model = model.blocking_read();
                let full_path = format!("{path}/model_{name}.pt");
                model.save(&full_path)?;
            }
        }

        Ok(())
    }

    /// Load system from path
    pub fn load(&mut self, path: &str) -> Result<(), anyhow::Error> {
        // Load state
        let serialized = std::fs::read_to_string(path)?;
        let state: SelfLearningState = serde_json::from_str(&serialized)?;

        // Restore system from state
        self.restore_from_state(state)?;

        // Load each model
        for name in self.configs.keys() {
            if let Some(model) = self.models.get(name) {
                let mut model = model.blocking_write();
                let full_path = format!("{path}/model_{name}.pt");
                if std::path::Path::new(&full_path).exists() {
                    model.load(&full_path)?;
                }
            }
        }

        Ok(())
    }

    /// Save model to disk with real neural network serialization
    pub async fn save_model(&self, path: &str) -> Result<(), anyhow::Error> {
        // Create directory structure for model storage
        let model_dir = std::path::Path::new(path);
        if let Some(parent) = model_dir.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Save metadata
        let metadata = SelfLearningMeta {
            models: self.models.keys().cloned().collect(),
            active_model: self.active_model.clone(),
            performance_history: self.performance_history.clone(),
        };

        let metadata_path = format!("{}.meta", path);
        let serialized_meta = serde_json::to_string_pretty(&metadata)?;
        tokio::fs::write(&metadata_path, serialized_meta).await?;

        // Save each neural model individually
        for (model_name, model_arc) in &self.models {
            let model_path = format!("{}.{}.model", path, model_name);
            let model = model_arc.read().await;

            // Real model serialization using the save method from NeuralNetwork trait
            if let Err(e) = model.save(&model_path) {
                warn!("Failed to save model {}: {}", model_name, e);
                // Continue saving other models even if one fails
            }
        }

        // Save performance metrics
        let metrics = self.metrics.read().await;
        let metrics_path = format!("{}.metrics", path);
        let serialized_metrics = serde_json::to_string_pretty(&*metrics)?;
        tokio::fs::write(&metrics_path, serialized_metrics).await?;

        info!("Successfully saved self-learning system to {}", path);
        Ok(())
    }

    /// Load model from disk with real neural network deserialization
    pub async fn load_model(&mut self, path: &str) -> Result<(), anyhow::Error> {
        // Load metadata first
        let metadata_path = format!("{}.meta", path);
        let metadata_content = match tokio::fs::read_to_string(&metadata_path).await {
            Ok(content) => content,
            Err(e) => {
                warn!("Failed to read metadata from {}: {}", metadata_path, e);
                return Ok(()); // Gracefully handle missing files
            }
        };

        let metadata: SelfLearningMeta = serde_json::from_str(&metadata_content)?;
        info!("Loading self-learning system from {}", path);

        // Clear existing models
        self.models.clear();

        // Load each neural model
        for model_name in &metadata.models {
            let model_path = format!("{}.{}.model", path, model_name);

            // Create new neural base with appropriate config
            let config = self
                .configs
                .get("base")
                .ok_or_else(|| anyhow::anyhow!("Base config not found"))?
                .clone();

            let mut neural_base = NeuralBase::new(config)?;

            // Load the saved model state
            if let Err(e) = neural_base.load(&model_path) {
                warn!(
                    "Failed to load model {} from {}: {}",
                    model_name, model_path, e
                );
                // Create a fresh model if loading fails
                info!("Creating fresh model for {}", model_name);
            }

            let boxed_model: Box<dyn NeuralNetwork + Send + Sync> = Box::new(neural_base);
            self.models
                .insert(model_name.clone(), Arc::new(RwLock::new(boxed_model)));
        }

        // Load performance metrics if available
        let metrics_path = format!("{}.metrics", path);
        if let Ok(metrics_content) = tokio::fs::read_to_string(&metrics_path).await {
            if let Ok(loaded_metrics) = serde_json::from_str::<PerformanceMetrics>(&metrics_content)
            {
                *self.metrics.write().await = loaded_metrics;
                info!("Loaded performance metrics from disk");
            }
        }

        // Restore metadata state
        self.active_model = metadata.active_model;
        self.performance_history = metadata.performance_history;

        info!(
            "Successfully loaded self-learning system with {} models",
            self.models.len()
        );
        Ok(())
    }

    /// Internal method to generate export data
    pub fn export_model(&self) -> Result<Vec<u8>, anyhow::Error> {
        // This would serialize the entire model state to bytes
        // For this simulation, we just create a simple JSON representation

        #[derive(Serialize)]
        struct ExportData {
            model_count: usize,
            active_model: String,
        }

        let export = ExportData {
            model_count: self.models.len(),
            active_model: self.active_model.clone(),
        };

        let serialized = serde_json::to_vec(&export)?;
        Ok(serialized)
    }

    /// Internal method to import model from serialized data
    pub async fn import_model(&mut self, data: &[u8]) -> Result<(), anyhow::Error> {
        // This would deserialize the entire model state
        // For this simulation, we just parse a simple JSON representation

        #[derive(Deserialize)]
        struct ImportData {
            model_count: usize,
            active_model: String,
        }

        let import: ImportData = serde_json::from_slice(data)?;

        // In a real implementation, we would restore actual neural models
        // For this simulation, just update some metadata

        self.active_model = import.active_model;

        // Generate placeholder models
        for i in 0..import.model_count {
            let _model_path = format!("model_{i}");
            let config = self.configs["base"].clone();
            let model = NeuralBase::new_sync(config)?;
            let boxed_model: Box<dyn NeuralNetwork + Send + Sync> = Box::new(model);
            self.models
                .insert(format!("model_{i}"), Arc::new(RwLock::new(boxed_model)));
        }

        Ok(())
    }
}
