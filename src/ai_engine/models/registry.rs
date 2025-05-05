use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::{Result, anyhow};
use super::neural_base::{NeuralBase, NeuralConfig};
use super::bci_interface::{BCIModel, SignalParams};
use super::self_learning::{SelfLearningSystem, SelfLearningConfig};
use serde::{Serialize, Deserialize};

/// Neural model registry for managing all AI models
pub struct ModelRegistry {
    /// Neural networks
    neural_models: Arc<RwLock<HashMap<String, Arc<RwLock<NeuralBase>>>>>,
    /// BCI models
    bci_models: Arc<RwLock<HashMap<String, Arc<RwLock<BCIModel>>>>>,
    /// Self-learning systems
    learning_systems: Arc<RwLock<HashMap<String, Arc<RwLock<SelfLearningSystem>>>>>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            neural_models: Arc::new(RwLock::new(HashMap::new())),
            bci_models: Arc::new(RwLock::new(HashMap::new())),
            learning_systems: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a new neural model
    pub async fn register_neural_model(&self, name: &str, config: NeuralConfig) -> Result<()> {
        let model = NeuralBase::new(config).await?;

        self.neural_models
            .write()
            .await
            .insert(name.to_string(), Arc::new(RwLock::new(model)));

        Ok(())
    }
    
    /// Get a neural model by name
    pub async fn get_neural_model(&self, name: &str) -> Result<Arc<RwLock<NeuralBase>>> {
        self.neural_models
            .read()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Neural model not found: {}", name))
    }
    
    /// Get a learning system by name
    pub async fn get_learning_system(&self, name: &str) -> Result<Arc<RwLock<SelfLearningSystem>>> {
        self.learning_systems
            .read()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("Learning system not found: {}", name))
    }
    
    /// Get a BCI model by name
    pub async fn get_bci_model(&self, name: &str) -> Result<Arc<RwLock<BCIModel>>> {
        self.bci_models
            .read()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("BCI model not found: {}", name))
    }
    
    /// Get any model by name and type
    pub async fn get_model(&self, name: &str, model_type: ModelType) -> Result<Arc<RwLock<dyn std::any::Any + Send + Sync>>> {
        match model_type {
            ModelType::Neural => {
                let model = self.get_neural_model(name).await?;
                Ok(model as Arc<RwLock<dyn std::any::Any + Send + Sync>>)
            }
            ModelType::BCI => {
                let model = self.get_bci_model(name).await?;
                Ok(model as Arc<RwLock<dyn std::any::Any + Send + Sync>>)
            }
            ModelType::SelfLearning => {
                let model = self.get_learning_system(name).await?;
                Ok(model as Arc<RwLock<dyn std::any::Any + Send + Sync>>)
            }
        }
    }
    
    /// Register a new BCI model
    pub async fn register_bci_model(&self, name: &str, config: NeuralConfig, signal_params: SignalParams) -> Result<()> {
        let model = BCIModel::new(config, signal_params).await?;

        self.bci_models
            .write()
            .await
            .insert(name.to_string(), Arc::new(RwLock::new(model)));

        Ok(())
    }
    
    /// Register a new self-learning system
    pub async fn register_self_learning_system(&self, name: &str, config: SelfLearningConfig) -> Result<()> {
        let system = SelfLearningSystem::new(config)?;

        self.learning_systems
            .write()
            .await
            .insert(name.to_string(), Arc::new(RwLock::new(system)));

        Ok(())
    }
    
    /// Save a model to disk
    pub async fn save_model(&self, name: &str, model_type: ModelType, path: &str) -> Result<()> {
        match model_type {
            ModelType::Neural => {
                let model = self.get_neural_model(name).await?;
                let model = model.read().await;
                model.save(path)?;
            }
            ModelType::BCI => {
                let model = self.get_bci_model(name).await?;
                let model = model.read().await;
                model.save(path).await?;
            }
            ModelType::SelfLearning => {
                let system = self.get_learning_system(name).await?;
                let system = system.read().await;
                // Implement SelfLearningSystem::save method
                // system.save(path)?;
            }
        }
        Ok(())
    }
    
    /// Load a model from disk
    pub async fn load_model(&self, name: &str, model_type: ModelType, path: &str) -> Result<()> {
        match model_type {
            ModelType::Neural => {
                let model = self.get_neural_model(name).await?;
                let mut model = model.write().await;
                model.load(path)?;
            }
            ModelType::BCI => {
                let model = self.get_bci_model(name).await?;
                let mut model = model.write().await;
                model.load(path).await?;
            }
            ModelType::SelfLearning => {
                let system = self.get_learning_system(name).await?;
                let mut system = system.write().await;
                // Implement SelfLearningSystem::load method
                // system.load(path)?;
            }
        }
        Ok(())
    }
}

/// Model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// Neural base model
    Neural,
    /// Brain-Computer Interface model
    BCI,
    /// Self-learning system
    SelfLearning,
}

impl Default for ModelType {
    fn default() -> Self {
        Self::Neural
    }
} 