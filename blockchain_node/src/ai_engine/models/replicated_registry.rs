use super::neural_base::NeuralNetwork;
use super::registry::{ModelMetadata, ModelRegistry, ModelType, RegistryConfig};

use anyhow::{anyhow, Result};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::time::interval;

/// Configuration for replicated model registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicatedRegistryConfig {
    /// Primary registry path
    pub primary_path: PathBuf,
    /// Replica registry paths
    pub replica_paths: Vec<PathBuf>,
    /// Synchronization interval in seconds
    pub sync_interval_secs: u64,
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Model integrity check interval
    pub integrity_check_interval_secs: u64,
    /// Maximum model size for replication (MB)
    pub max_model_size_mb: u64,
    /// Enable compression for replication
    pub compression_enabled: bool,
    /// Replication timeout in seconds
    pub replication_timeout_secs: u64,
    /// Number of replicas required for quorum
    pub replication_quorum: usize,
}

impl Default for ReplicatedRegistryConfig {
    fn default() -> Self {
        Self {
            primary_path: PathBuf::from("models/primary"),
            replica_paths: vec![
                PathBuf::from("models/replica1"),
                PathBuf::from("models/replica2"),
            ],
            sync_interval_secs: 300, // 5 minutes
            auto_failover: true,
            integrity_check_interval_secs: 3600, // 1 hour
            max_model_size_mb: 1024,             // 1GB
            compression_enabled: true,
            replication_timeout_secs: 60,
            replication_quorum: 2,
        }
    }
}

/// Model replication status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationStatus {
    /// Model name
    pub model_name: String,
    /// Primary checksum
    pub primary_checksum: String,
    /// Replica checksums
    pub replica_checksums: HashMap<String, String>,
    /// Last sync time
    pub last_sync: SystemTime,
    /// Is synchronized
    pub is_synchronized: bool,
    /// Sync errors
    pub sync_errors: Vec<String>,
}

/// Registry health status
#[derive(Debug, Clone, PartialEq)]
pub enum RegistryHealth {
    Healthy,
    Degraded,
    Failed,
}

/// Replicated model registry
pub struct ReplicatedModelRegistry {
    /// Configuration
    config: ReplicatedRegistryConfig,
    /// Primary registry
    primary_registry: Arc<RwLock<ModelRegistry>>,
    /// Replica registries
    replica_registries: Arc<RwLock<HashMap<String, Arc<RwLock<ModelRegistry>>>>>,
    /// Replication status
    replication_status: Arc<RwLock<HashMap<String, ReplicationStatus>>>,
    /// Registry health
    health: Arc<RwLock<RegistryHealth>>,
    /// Active registry (for failover)
    active_registry: Arc<RwLock<String>>,
    /// Sync scheduler handle
    sync_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Integrity checker handle
    integrity_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Event broadcaster
    event_sender: broadcast::Sender<RegistryEvent>,
    /// Model cache for fast access
    model_cache: Arc<RwLock<HashMap<String, CachedModel>>>,
}

#[derive(Debug, Clone)]
pub enum RegistryEvent {
    ModelReplicated(String),
    ModelFailedReplication(String, String),
    RegistryFailover(String, String),
    IntegrityCheckFailed(String),
    RegistryHealthChanged(RegistryHealth),
}

#[derive(Clone)]
struct CachedModel {
    /// Model data
    data: Vec<u8>,
    /// Model metadata
    metadata: ModelMetadata,
    /// Cache timestamp
    cached_at: SystemTime,
    /// Access count
    access_count: u64,
}

impl ReplicatedModelRegistry {
    /// Create new replicated registry
    pub async fn new(config: ReplicatedRegistryConfig) -> Result<Self> {
        // Create directories
        std::fs::create_dir_all(&config.primary_path)?;
        for path in &config.replica_paths {
            std::fs::create_dir_all(path)?;
        }

        // Initialize primary registry
        let primary_config = RegistryConfig::default();
        let primary_registry = Arc::new(RwLock::new(ModelRegistry::new(primary_config)));

        // Initialize replica registries
        let mut replicas = HashMap::new();
        for (i, _path) in config.replica_paths.iter().enumerate() {
            let replica_config = RegistryConfig::default();
            let replica = ModelRegistry::new(replica_config);
            replicas.insert(format!("replica_{}", i), Arc::new(RwLock::new(replica)));
        }

        let (event_sender, _) = broadcast::channel(1000);

        let registry = Self {
            config,
            primary_registry,
            replica_registries: Arc::new(RwLock::new(replicas)),
            replication_status: Arc::new(RwLock::new(HashMap::new())),
            health: Arc::new(RwLock::new(RegistryHealth::Healthy)),
            active_registry: Arc::new(RwLock::new("primary".to_string())),
            sync_handle: Arc::new(Mutex::new(None)),
            integrity_handle: Arc::new(Mutex::new(None)),
            event_sender,
            model_cache: Arc::new(RwLock::new(HashMap::new())),
        };

        Ok(registry)
    }

    /// Start replication services
    pub async fn start(&self) -> Result<()> {
        // Start synchronization
        self.start_sync_scheduler().await?;

        // Start integrity checking
        self.start_integrity_checker().await?;

        info!("Replicated model registry started");
        Ok(())
    }

    /// Start synchronization scheduler
    async fn start_sync_scheduler(&self) -> Result<()> {
        let config = self.config.clone();
        let primary_registry = self.primary_registry.clone();
        let replica_registries = self.replica_registries.clone();
        let replication_status = self.replication_status.clone();
        let event_sender = self.event_sender.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.sync_interval_secs));

            loop {
                interval.tick().await;

                // Get all models from primary
                let models = {
                    let primary = primary_registry.read().await;
                    primary.list_all_models().await
                };

                // Sync each model
                for (model_name, metadata) in models {
                    if let Err(e) = Self::sync_model(
                        &model_name,
                        &metadata,
                        &primary_registry,
                        &replica_registries,
                        &replication_status,
                        &event_sender,
                        &config,
                    )
                    .await
                    {
                        error!("Failed to sync model {}: {}", model_name, e);
                    }
                }
            }
        });

        *self.sync_handle.lock().await = Some(handle);
        Ok(())
    }

    /// Sync a single model
    async fn sync_model(
        model_name: &str,
        metadata: &ModelMetadata,
        primary_registry: &Arc<RwLock<ModelRegistry>>,
        replica_registries: &Arc<RwLock<HashMap<String, Arc<RwLock<ModelRegistry>>>>>,
        replication_status: &Arc<RwLock<HashMap<String, ReplicationStatus>>>,
        event_sender: &broadcast::Sender<RegistryEvent>,
        config: &ReplicatedRegistryConfig,
    ) -> Result<()> {
        info!("Syncing model: {}", model_name);

        // Export model from primary
        let model_data = {
            let primary = primary_registry.read().await;
            primary.export_model(model_name).await?
        };

        // Check size limit
        if model_data.len() > (config.max_model_size_mb * 1024 * 1024) as usize {
            return Err(anyhow!("Model exceeds size limit for replication"));
        }

        // Compress if enabled
        let data_to_replicate = if config.compression_enabled {
            zstd::encode_all(std::io::Cursor::new(model_data.as_slice()), 3)?
        } else {
            model_data.clone()
        };

        // Calculate checksum
        let primary_checksum = blake3::hash(&model_data).to_hex().to_string();

        // Replicate to each replica
        let replicas = replica_registries.read().await;
        let mut replica_checksums = HashMap::new();
        let mut sync_errors = Vec::new();
        let mut successful_replicas = 0;

        for (replica_name, replica_registry) in replicas.iter() {
            match tokio::time::timeout(
                Duration::from_secs(config.replication_timeout_secs),
                Self::replicate_to_replica(
                    model_name,
                    metadata,
                    &data_to_replicate,
                    replica_registry,
                    config.compression_enabled,
                ),
            )
            .await
            {
                Ok(Ok(checksum)) => {
                    replica_checksums.insert(replica_name.clone(), checksum);
                    successful_replicas += 1;
                }
                Ok(Err(e)) => {
                    let error = format!("Replication to {} failed: {}", replica_name, e);
                    sync_errors.push(error.clone());
                    warn!("{}", error);
                }
                Err(_) => {
                    let error = format!("Replication to {} timed out", replica_name);
                    sync_errors.push(error.clone());
                    warn!("{}", error);
                }
            }
        }

        // Check quorum
        let is_synchronized = successful_replicas >= config.replication_quorum;

        // Update status
        let status = ReplicationStatus {
            model_name: model_name.to_string(),
            primary_checksum,
            replica_checksums,
            last_sync: SystemTime::now(),
            is_synchronized,
            sync_errors: sync_errors.clone(),
        };

        replication_status
            .write()
            .await
            .insert(model_name.to_string(), status);

        // Send event
        if is_synchronized {
            let _ = event_sender.send(RegistryEvent::ModelReplicated(model_name.to_string()));
        } else {
            let _ = event_sender.send(RegistryEvent::ModelFailedReplication(
                model_name.to_string(),
                sync_errors.join("; "),
            ));
        }

        Ok(())
    }

    /// Replicate model to a single replica
    async fn replicate_to_replica(
        model_name: &str,
        metadata: &ModelMetadata,
        data: &[u8],
        replica_registry: &Arc<RwLock<ModelRegistry>>,
        is_compressed: bool,
    ) -> Result<String> {
        // Decompress if needed
        let model_data = if is_compressed {
            zstd::decode_all(std::io::Cursor::new(data))?
        } else {
            data.to_vec()
        };

        // Import to replica
        let mut replica = replica_registry.write().await;
        replica
            .import_model(model_name, metadata.clone(), model_data.clone())
            .await?;

        // Calculate and return checksum
        Ok(blake3::hash(&model_data).to_hex().to_string())
    }

    /// Start integrity checker
    async fn start_integrity_checker(&self) -> Result<()> {
        let config = self.config.clone();
        let primary_registry = self.primary_registry.clone();
        let replica_registries = self.replica_registries.clone();
        let replication_status = self.replication_status.clone();
        let health = self.health.clone();
        let event_sender = self.event_sender.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.integrity_check_interval_secs));

            loop {
                interval.tick().await;

                let mut failed_checks = 0;
                let mut total_checks = 0;

                // Check each model
                let statuses: Vec<_> =
                    { replication_status.read().await.values().cloned().collect() };

                for status in statuses {
                    total_checks += 1;

                    // Verify checksums match
                    let mut checksum_mismatch = false;
                    for (replica_name, replica_checksum) in &status.replica_checksums {
                        if *replica_checksum != status.primary_checksum {
                            checksum_mismatch = true;
                            warn!(
                                "Checksum mismatch for model {} in replica {}",
                                status.model_name, replica_name
                            );
                            let _ = event_sender.send(RegistryEvent::IntegrityCheckFailed(
                                status.model_name.clone(),
                            ));
                            break;
                        }
                    }

                    if checksum_mismatch || !status.is_synchronized {
                        failed_checks += 1;
                    }
                }

                // Update health status
                let new_health = if failed_checks == 0 {
                    RegistryHealth::Healthy
                } else if failed_checks < total_checks / 2 {
                    RegistryHealth::Degraded
                } else {
                    RegistryHealth::Failed
                };

                let mut current_health = health.write().await;
                if *current_health != new_health {
                    *current_health = new_health.clone();
                    let _ = event_sender.send(RegistryEvent::RegistryHealthChanged(new_health));
                }
            }
        });

        *self.integrity_handle.lock().await = Some(handle);
        Ok(())
    }

    /// Register a neural model with replication
    pub async fn register_neural_model(
        &self,
        name: &str,
        config: crate::ai_engine::models::neural_base::NeuralConfig,
    ) -> Result<()> {
        // Register in active registry
        let active = self.active_registry.read().await;

        if *active == "primary" {
            self.primary_registry
                .write()
                .await
                .register_neural_model(name, config)
                .await?;
        } else {
            // Failover mode - register in replica
            let replicas = self.replica_registries.read().await;
            if let Some(replica) = replicas.get(&*active) {
                replica
                    .write()
                    .await
                    .register_neural_model(name, config)
                    .await?;
            }
        }

        // Trigger immediate sync
        self.sync_model_immediate(name).await?;

        Ok(())
    }

    /// Get neural model with caching
    pub async fn get_neural_model(
        &self,
        name: &str,
    ) -> Result<Arc<RwLock<Box<dyn NeuralNetwork + Send + Sync>>>> {
        // Check cache first
        if let Some(cached) = self.get_from_cache(name).await {
            return self.deserialize_neural_model(&cached.data).await;
        }

        // Get from active registry
        let active = self.active_registry.read().await;

        let model = if *active == "primary" {
            self.primary_registry
                .read()
                .await
                .get_neural_model(name)
                .await?
        } else {
            let replicas = self.replica_registries.read().await;
            if let Some(replica) = replicas.get(&*active) {
                replica.read().await.get_neural_model(name).await?
            } else {
                return Err(anyhow!("Active registry not found"));
            }
        };

        // Cache for future access
        self.cache_model(name, &model).await?;

        Ok(model)
    }

    /// Force failover to replica
    pub async fn force_failover(&self, replica_name: &str) -> Result<()> {
        let replicas = self.replica_registries.read().await;
        if !replicas.contains_key(replica_name) {
            return Err(anyhow!("Replica not found"));
        }

        let old_active = self.active_registry.read().await.clone();
        *self.active_registry.write().await = replica_name.to_string();

        let _ = self.event_sender.send(RegistryEvent::RegistryFailover(
            old_active,
            replica_name.to_string(),
        ));

        info!("Forced failover to replica: {}", replica_name);
        Ok(())
    }

    /// Get replication status
    pub async fn get_replication_status(&self) -> HashMap<String, ReplicationStatus> {
        self.replication_status.read().await.clone()
    }

    /// Get registry health
    pub async fn get_health(&self) -> RegistryHealth {
        self.health.read().await.clone()
    }

    /// Sync specific model immediately
    async fn sync_model_immediate(&self, model_name: &str) -> Result<()> {
        let primary = self.primary_registry.read().await;
        let models = primary.list_all_models().await;

        if let Some(metadata) = models.get(model_name) {
            Self::sync_model(
                model_name,
                metadata,
                &self.primary_registry,
                &self.replica_registries,
                &self.replication_status,
                &self.event_sender,
                &self.config,
            )
            .await?;
        }

        Ok(())
    }

    /// Get model from cache
    async fn get_from_cache(&self, name: &str) -> Option<CachedModel> {
        let mut cache = self.model_cache.write().await;
        if let Some(cached) = cache.get_mut(name) {
            cached.access_count += 1;
            return Some(cached.clone());
        }
        None
    }

    /// Cache a model
    async fn cache_model(
        &self,
        name: &str,
        model: &Arc<RwLock<Box<dyn NeuralNetwork + Send + Sync>>>,
    ) -> Result<()> {
        // Serialize model
        let data = self.serialize_neural_model(model).await?;

        let cached = CachedModel {
            data,
            metadata: ModelMetadata {
                model_type: ModelType::NeuralNetwork,
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                training_iterations: 0,
                metrics: HashMap::new(),
                version: "1.0".to_string(),
                description: String::new(),
            },
            cached_at: SystemTime::now(),
            access_count: 1,
        };

        self.model_cache
            .write()
            .await
            .insert(name.to_string(), cached);
        Ok(())
    }

    /// Serialize neural model (placeholder)
    async fn serialize_neural_model(
        &self,
        model: &Arc<RwLock<Box<dyn NeuralNetwork + Send + Sync>>>,
    ) -> Result<Vec<u8>> {
        // In a real implementation, this would serialize the model
        Ok(vec![])
    }

    /// Deserialize neural model (placeholder)
    async fn deserialize_neural_model(
        &self,
        data: &[u8],
    ) -> Result<Arc<RwLock<Box<dyn NeuralNetwork + Send + Sync>>>> {
        // In a real implementation, this would deserialize the model
        Err(anyhow!("Deserialization not implemented"))
    }

    /// Stop replication services
    pub async fn stop(&self) {
        if let Some(handle) = self.sync_handle.lock().await.take() {
            handle.abort();
        }
        if let Some(handle) = self.integrity_handle.lock().await.take() {
            handle.abort();
        }
        info!("Replicated model registry stopped");
    }
}

// Extension methods for ModelRegistry to support import/export
impl ModelRegistry {
    /// Export a model for replication
    pub async fn export_model(&self, name: &str) -> Result<Vec<u8>> {
        // This would export the model data
        // Placeholder implementation
        Ok(vec![])
    }

    /// Import a model from replication
    pub async fn import_model(
        &mut self,
        name: &str,
        metadata: ModelMetadata,
        data: Vec<u8>,
    ) -> Result<()> {
        // This would import the model data
        // Placeholder implementation
        Ok(())
    }

    /// List all models with metadata
    pub async fn list_all_models(&self) -> HashMap<String, ModelMetadata> {
        // This would return all models with their metadata
        // Placeholder implementation
        HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_replicated_registry() {
        let config = ReplicatedRegistryConfig {
            primary_path: PathBuf::from("/tmp/test_primary_models"),
            replica_paths: vec![
                PathBuf::from("/tmp/test_replica1_models"),
                PathBuf::from("/tmp/test_replica2_models"),
            ],
            sync_interval_secs: 1,
            ..Default::default()
        };

        let registry = ReplicatedModelRegistry::new(config).await.unwrap();
        registry.start().await.unwrap();

        // Test health
        assert_eq!(registry.get_health().await, RegistryHealth::Healthy);

        // Clean up
        registry.stop().await;
        let _ = std::fs::remove_dir_all("/tmp/test_primary_models");
        let _ = std::fs::remove_dir_all("/tmp/test_replica1_models");
        let _ = std::fs::remove_dir_all("/tmp/test_replica2_models");
    }
}
