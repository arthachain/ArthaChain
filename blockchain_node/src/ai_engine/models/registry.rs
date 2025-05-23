use super::{
    bci_interface::{BCIModel, SignalParams},
    neural_base::{NeuralBase, NeuralConfig, NeuralNetwork},
    self_learning::{SelfLearningConfig, SelfLearningSystem},
};
use anyhow::{anyhow, Result};
use log;
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

/// Central registry for all AI models
pub struct ModelRegistry {
    /// Neural network models
    neural_models: Arc<RwLock<HashMap<String, Arc<RwLock<Box<dyn NeuralNetwork>>>>>>,
    /// Self-learning systems
    learning_systems: Arc<RwLock<HashMap<String, Arc<RwLock<SelfLearningSystem>>>>>,
    /// BCI models
    bci_models: Arc<RwLock<HashMap<String, Arc<RwLock<BCIModel>>>>>,
    /// Model metadata
    metadata: Arc<RwLock<HashMap<String, ModelMetadata>>>,
    /// Registry configuration
    config: RegistryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model type
    pub model_type: ModelType,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last updated timestamp
    pub updated_at: SystemTime,
    /// Training iterations
    pub training_iterations: u64,
    /// Performance metrics
    pub metrics: HashMap<String, f32>,
    /// Model version
    pub version: String,
    /// Model description
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    NeuralNetwork,
    BCI,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Maximum models per type
    pub max_models: usize,
    /// Auto-cleanup threshold
    pub cleanup_threshold: usize,
    /// Model versioning strategy
    pub versioning: VersioningStrategy,
    /// Storage configuration
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersioningStrategy {
    Timestamp,
    Semantic,
    Incremental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Base path for model storage
    pub base_path: String,
    /// Storage format
    pub format: StorageFormat,
    /// Compression level
    pub compression: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageFormat {
    PyTorch,
    ONNX,
    TorchScript,
}

#[derive(Debug, Clone)]
pub struct ModelRegistryStats {
    /// Total number of models
    pub total_models: usize,
    /// Number of neural network models
    pub neural_models: usize,
    /// Number of BCI models
    pub bci_models: usize,
    /// Number of self-learning systems
    pub self_learning_systems: usize,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            neural_models: Arc::new(RwLock::new(HashMap::new())),
            learning_systems: Arc::new(RwLock::new(HashMap::new())),
            bci_models: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Register a new neural model
    pub async fn register_neural_model(&self, name: &str, config: NeuralConfig) -> Result<()> {
        let model = NeuralBase::new(config).await?;

        self.neural_models
            .write()
            .await
            .insert(name.to_string(), Arc::new(RwLock::new(Box::new(model))));

        Ok(())
    }

    /// Register a new self-learning system
    pub async fn register_self_learning_system(
        &self,
        name: &str,
        config: SelfLearningConfig,
    ) -> Result<()> {
        let system = SelfLearningSystem::new(config).await?;

        self.learning_systems
            .write()
            .await
            .insert(name.to_string(), Arc::new(RwLock::new(system)));

        Ok(())
    }

    /// Register a new BCI model
    pub async fn register_bci_model(
        &self,
        name: &str,
        config: NeuralConfig,
        signal_params: SignalParams,
    ) -> Result<()> {
        let model = BCIModel::new(config, signal_params).await?;

        self.bci_models
            .write()
            .await
            .insert(name.to_string(), Arc::new(RwLock::new(model)));

        Ok(())
    }

    /// Get a neural base model
    pub async fn get_neural_model(
        &self,
        name: &str,
    ) -> Result<Arc<RwLock<Box<dyn NeuralNetwork>>>> {
        self.neural_models
            .read()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Neural model not found: {}", name))
    }

    /// Get a self-learning system
    pub async fn get_learning_system(&self, name: &str) -> Result<Arc<RwLock<SelfLearningSystem>>> {
        self.learning_systems
            .read()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Learning system not found: {}", name))
    }

    /// Get a BCI model
    pub async fn get_bci_model(&self, name: &str) -> Result<Arc<RwLock<BCIModel>>> {
        self.bci_models
            .read()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("BCI model not found: {}", name))
    }

    /// Update model metadata
    pub async fn update_metadata(
        &mut self,
        name: &str,
        metrics: HashMap<String, f32>,
    ) -> Result<()> {
        // Use a let binding to extend the lifetime of the write guard
        let mut metadata_guard = self.metadata.write().await;
        let metadata = metadata_guard
            .get_mut(name)
            .ok_or_else(|| anyhow!("Model not found: {}", name))?;

        metadata.updated_at = SystemTime::now();
        metadata.training_iterations += 1;
        metadata.metrics.extend(metrics);

        Ok(())
    }

    /// Save all models
    pub async fn save_all(&self) -> Result<()> {
        let base_path = &self.config.storage.base_path;

        // Save neural models
        for (name, model) in self.neural_models.read().await.iter() {
            let model = model.read().await;
            let path = format!("{}/neural_{}", base_path, name);
            model.save(&path)?;
        }

        // Save learning systems
        for (name, system) in self.learning_systems.read().await.iter() {
            // For SelfLearningSystem, we need a different approach
            let path = format!("{}/learning_{}", base_path, name);
            let guard = system.read().await;
            if let Err(e) = self.save_learning_system(&guard, &path).await {
                log::warn!("Failed to save learning system {}: {}", name, e);
            }
        }

        // Save BCI models
        for (name, model) in self.bci_models.read().await.iter() {
            let model = model.read().await;
            let path = format!("{}/bci_{}", base_path, name);
            model.save(&path).await?;
        }

        // Save metadata - clone the HashMap from the guard to avoid serializing the guard itself
        let metadata_path = format!("{}/metadata.json", base_path);
        let metadata_clone = {
            let guard = self.metadata.read().await;
            guard.clone()
        };

        std::fs::write(
            &metadata_path,
            serde_json::to_string_pretty(&metadata_clone)?,
        )?;

        Ok(())
    }

    /// Helper method to save a learning system
    async fn save_learning_system(&self, system: &SelfLearningSystem, path: &str) -> Result<()> {
        // Instead of trying to serialize the entire SelfLearningSystem, we serialize its parts
        // This example assumes SelfLearningSystem has a method to get its serializable state
        let system_state = system.get_serializable_state()?;
        let serialized = serde_json::to_string(&system_state)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Load all models
    pub async fn load_all(&mut self) -> Result<()> {
        let base_path = &self.config.storage.base_path;

        // Load metadata first
        let metadata_path = format!("{}/metadata.json", base_path);
        if std::path::Path::new(&metadata_path).exists() {
            let metadata_str = std::fs::read_to_string(&metadata_path)?;
            let loaded_metadata: HashMap<String, ModelMetadata> =
                serde_json::from_str(&metadata_str)?;
            // Update the metadata
            let mut metadata_guard = self.metadata.write().await;
            *metadata_guard = loaded_metadata;
        }

        // Load models based on metadata
        {
            // Use a let binding to extend the lifetime of the read guard
            let metadata_guard = self.metadata.read().await;

            for (name, metadata) in metadata_guard.iter() {
                match metadata.model_type {
                    ModelType::NeuralNetwork => {
                        if let Ok(model) = self.get_neural_model(name).await {
                            let mut model = model.write().await;
                            let path = format!("{}/neural_{}", base_path, name);
                            model.load(&path)?;
                        }
                    }
                    ModelType::Custom(ref custom_name) => {
                        if let Ok(system) = self.get_learning_system(custom_name).await {
                            let path = format!("{}/learning_{}", base_path, name);
                            // Use a helper method to load the system
                            self.load_learning_system(&system, &path).await?;
                        }
                    }
                    ModelType::BCI => {
                        if let Ok(model) = self.get_bci_model(name).await {
                            let model = model.write().await;
                            let path = format!("{}/bci_{}", base_path, name);
                            // Call the load method if available or implement an alternative
                            if let Err(e) = self.load_bci_model(&model, &path).await {
                                log::warn!("Failed to load BCI model {}: {}", name, e);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Helper method to load a learning system
    async fn load_learning_system(
        &self,
        system: &Arc<RwLock<SelfLearningSystem>>,
        path: &str,
    ) -> Result<()> {
        // Custom approach to load the SelfLearningSystem
        if std::path::Path::new(path).exists() {
            let serialized = std::fs::read_to_string(path)?;
            let system_state = serde_json::from_str(&serialized)?;
            let mut guard = system.write().await;
            guard.restore_from_state(system_state)?;
        }
        Ok(())
    }

    /// Helper method to load a BCI model
    async fn load_bci_model(&self, model: &BCIModel, path: &str) -> Result<()> {
        // Implement a method to load BCIModel
        if std::path::Path::new(path).exists() {
            // Create a mutable copy of the model path to pass to load
            let path_str = path.to_string();
            // Use the method on BCIModel that takes &self
            model.load(&path_str).await?;
        }
        Ok(())
    }

    /// Load default models when none exist
    async fn load_default_models(&self) -> Result<(), anyhow::Error> {
        // Create basic configurations
        let basic_neural_config = NeuralConfig {
            layers: vec![], // You need to fill in appropriate layer configs
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            optimizer: "Adam".to_string(),
            loss: "MSE".to_string(),
        };

        // Register a basic neural model
        self.register_neural_model("default_neural", basic_neural_config.clone())
            .await?;

        // Register metadata
        let mut metadata_guard = self.metadata.write().await;
        metadata_guard.insert(
            "default_neural".to_string(),
            ModelMetadata {
                model_type: ModelType::NeuralNetwork,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                training_iterations: 0,
                metrics: HashMap::new(),
                version: "1.0.0".to_string(),
                description: "Default neural model".to_string(),
            },
        );

        Ok(())
    }

    /// Create a copy of models and metadata for temporary use
    pub async fn get_models_snapshot(&self) -> Vec<ModelMetadata> {
        let mut models: Vec<_> = {
            let metadata_guard = self.metadata.read().await;
            metadata_guard.iter().map(|(_, v)| v.clone()).collect()
        };
        models.sort_by_key(|m| m.updated_at);
        models
    }

    /// Ensure all models are loaded
    pub async fn ensure_models_loaded(&self) -> Result<(), anyhow::Error> {
        let models = self.metadata.read().await;

        if models.is_empty() {
            // No models found, load defaults
            drop(models); // Release the read lock

            // Load default models
            self.load_default_models().await?;

            info!("Loaded default models");
        }

        Ok(())
    }

    /// Get statistics about registered models
    pub async fn get_statistics(&self) -> ModelRegistryStats {
        let models = self.metadata.read().await;

        ModelRegistryStats {
            total_models: models.len(),
            neural_models: models
                .values()
                .filter(|&model| matches!(model.model_type, ModelType::NeuralNetwork))
                .count(),
            bci_models: models
                .values()
                .filter(|&model| matches!(model.model_type, ModelType::BCI))
                .count(),
            self_learning_systems: models
                .values()
                .filter(|&model| matches!(model.model_type, ModelType::Custom(_)))
                .count(),
        }
    }
}
