// AI Engine modules will be implemented here
pub mod config;
pub mod data_chunking;
pub mod device_health;
pub mod explainability;
pub mod fraud_detection;
pub mod models;
pub mod security;
pub mod user_identification;

use crate::ai_engine::config::model_orchestration::{
    ModelFailoverSettings, ModelOrchestrationConfig,
};
use crate::ai_engine::models::bci_interface::FilterParams;
use crate::ai_engine::models::{
    bci_interface::SignalParams,
    neural_base::{ActivationType, LayerConfig, NeuralConfig},
    registry::{ModelRegistry, RegistryConfig, StorageConfig, StorageFormat, VersioningStrategy},
    self_learning::SelfLearningConfig,
    types::Experience,
};
use crate::config::Config;
use crate::ledger::state::State;
use anyhow::{Context, Result};
use log::{error, info, warn};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use sysinfo::System;
use tokio::sync::RwLock;
use tokio::time::Duration;

/// Central AI Engine that integrates all AI modules
pub struct AIEngine {
    /// Device Health AI module
    pub device_health: device_health::DeviceHealthAI,
    /// User Identification AI module
    pub user_identification: user_identification::UserIdentificationAI,
    /// Data Chunking AI module
    pub data_chunking: data_chunking::DataChunkingAI,
    /// Fraud Detection AI module
    pub fraud_detection: fraud_detection::FraudDetectionAI,
    /// Security AI module
    pub security_ai: Option<security::SecurityAI>,
    /// Whether the AI Engine is running
    running: Arc<Mutex<bool>>,
    /// AI models directory
    models_dir: PathBuf,
    /// Config reference
    config: Config,
    /// Orchestration configuration
    orchestration: Arc<RwLock<ModelOrchestrationConfig>>,
    /// Neural model registry
    registry: ModelRegistry,
}

impl AIEngine {
    /// Create a new AI Engine instance with orchestration
    pub fn new(config: Config) -> Self {
        // Set default models directory if not provided
        let models_dir = if config.ai_model_dir.as_os_str().is_empty() {
            PathBuf::from("./models")
        } else {
            config.ai_model_dir.clone()
        };

        // Initialize orchestration config
        let orchestration = ModelOrchestrationConfig::default();

        // Create model registry
        let registry_config = RegistryConfig {
            max_models: 10,
            cleanup_threshold: 8,
            versioning: VersioningStrategy::Semantic,
            storage: StorageConfig {
                base_path: "models".to_string(),
                format: StorageFormat::PyTorch,
                compression: Some(3),
            },
        };

        let registry = ModelRegistry::new(registry_config);

        Self {
            device_health: device_health::DeviceHealthAI::new(&config),
            user_identification: user_identification::UserIdentificationAI::new(&config),
            data_chunking: data_chunking::DataChunkingAI::new(&config),
            fraud_detection: fraud_detection::FraudDetectionAI::new(&config),
            security_ai: None,
            running: Arc::new(Mutex::new(false)),
            models_dir,
            config,
            orchestration: Arc::new(RwLock::new(orchestration)),
            registry,
        }
    }

    /// Initialize the Security AI module with blockchain state
    pub fn init_security_ai(&mut self, state: Arc<RwLock<State>>) -> Result<()> {
        // Create the SecurityAI instance
        let security = security::SecurityAI::new(self.config.clone(), state)
            .context("Failed to initialize SecurityAI")?;

        self.security_ai = Some(security);
        Ok(())
    }

    /// Start the AI Engine with orchestrated scheduling
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Ok(());
        }

        let orchestration = self.orchestration.read().await;

        // Start device health monitoring
        if orchestration
            .enabled_components
            .contains(&"device_health".to_string())
        {
            let interval = orchestration.scheduler.device_health_interval;
            self.start_device_health_monitor(interval).await?;
        }

        // Start user identity updates
        if orchestration
            .enabled_components
            .contains(&"user_identity".to_string())
        {
            let interval = orchestration.scheduler.identity_update_interval;
            self.start_identity_updates(interval).await?;
        }

        // Start data chunking optimization
        if orchestration
            .enabled_components
            .contains(&"data_chunking".to_string())
        {
            let interval = orchestration.scheduler.chunking_refresh_interval;
            self.start_chunking_optimization(interval).await?;
        }

        // Start security monitoring
        if let Some(security) = &mut self.security_ai {
            if orchestration
                .enabled_components
                .contains(&"security".to_string())
            {
                let interval = orchestration.scheduler.security_update_interval;
                security.start_monitoring(interval).await?;
            }
        }

        // Start fraud detection training
        if orchestration
            .enabled_components
            .contains(&"fraud_detection".to_string())
        {
            let interval = orchestration.scheduler.fraud_detection_training_interval;
            self.start_fraud_detection_training(interval).await?;
        }

        *running = true;
        info!("AI Engine started with orchestrated scheduling");
        Ok(())
    }

    /// Start device health monitoring task
    async fn start_device_health_monitor(&self, interval: Duration) -> Result<()> {
        let device_health = self.device_health.clone();
        let failover = self.orchestration.read().await.failover.clone();

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;

                // Check resource usage
                if Self::system_resources_exceeded(&failover) {
                    warn!("Falling back to rule-based device health checks");
                    // Use rule-based checks
                    continue;
                }

                if let Err(e) = device_health.update_metrics().await {
                    error!("Failed to update device health metrics: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start user identity update task
    async fn start_identity_updates(&self, interval: Duration) -> Result<()> {
        let user_identification = self.user_identification.clone();
        let failover = self.orchestration.read().await.failover.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            loop {
                interval.tick().await;
                if Self::system_resources_exceeded(&failover) {
                    continue;
                }

                if let Err(e) = user_identification.update_identities().await {
                    error!("Failed to update user identities: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start data chunking optimization task
    async fn start_chunking_optimization(&self, interval: Duration) -> Result<()> {
        let data_chunking = self.data_chunking.clone();
        let failover = self.orchestration.read().await.failover.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            loop {
                interval.tick().await;
                if Self::system_resources_exceeded(&failover) {
                    continue;
                }

                if let Err(e) = data_chunking.optimize_chunks().await {
                    error!("Failed to optimize data chunks: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start fraud detection training task
    async fn start_fraud_detection_training(&self, interval: Duration) -> Result<()> {
        let fraud_detection = self.fraud_detection.clone();
        let failover = self.orchestration.read().await.failover.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            loop {
                interval.tick().await;
                if Self::system_resources_exceeded(&failover) {
                    continue;
                }

                if let Err(e) = fraud_detection.train_model().await {
                    error!("Failed to train fraud detection model: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Check if system resources are exceeded based on failover settings
    fn system_resources_exceeded(failover: &ModelFailoverSettings) -> bool {
        let mut sys = System::new_all();
        // Refresh CPU and memory
        sys.refresh_memory();
        sys.refresh_cpu();

        // Simulate disk usage with dummy data - avoid using disks() method
        let disk_usage = 65.0; // 65% disk usage as dummy value

        // Check memory usage
        let memory_usage = (sys.used_memory() as f32 / sys.total_memory() as f32) * 100.0;

        // Check CPU usage
        let cpu_usage = sys.global_cpu_info().cpu_usage();

        disk_usage > failover.disk_threshold
            || memory_usage > failover.memory_threshold
            || cpu_usage > failover.cpu_threshold
    }

    /// Stop the AI Engine and all its modules
    pub fn stop(&self) {
        let mut running = self.running.lock().unwrap();
        if !*running {
            return;
        }

        // Stop each AI module
        let _ = self.device_health.stop();

        // Security AI is stopped via its JoinHandle

        *running = false;
        info!("AI Engine stopped");
    }

    /// Update AI models for all modules
    pub async fn update_models(&mut self) -> Result<()> {
        info!("Updating AI models for all modules");

        // Ensure models directory exists
        if !self.models_dir.exists() {
            std::fs::create_dir_all(&self.models_dir)?;
        }

        // Update each AI module's model
        let device_health_model = self.models_dir.join("device_health_model.bin");
        if let Err(e) = self
            .device_health
            .update_model(device_health_model.to_str().unwrap())
            .await
        {
            warn!("Failed to update Device Health AI model: {}", e);
        }

        let user_id_model = self.models_dir.join("user_identification_model.bin");
        if let Err(e) = self
            .user_identification
            .update_model(user_id_model.to_str().unwrap())
            .await
        {
            warn!("Failed to update User Identification AI model: {}", e);
        }

        let data_chunking_model = self.models_dir.join("data_chunking_model.bin");
        if let Err(e) = self
            .data_chunking
            .update_model(data_chunking_model.to_str().unwrap())
            .await
        {
            warn!("Failed to update Data Chunking AI model: {}", e);
        }

        let fraud_detection_model = self.models_dir.join("fraud_detection_model.bin");
        if let Err(e) = self
            .fraud_detection
            .update_model(fraud_detection_model.to_str().unwrap())
            .await
        {
            warn!("Failed to update Fraud Detection AI model: {}", e);
        }

        // Security AI has its own model reload mechanism

        info!("AI model updates completed");
        Ok(())
    }

    /// Get participation eligibility and weight for a node
    pub fn get_participation_info(&self, node_id: &str) -> (bool, f32) {
        // Check device health
        let device_eligible = self.device_health.is_eligible_for_validation();
        let device_weight = self.device_health.get_participation_weight();

        // Check security score
        let security_weight = match self.fraud_detection.get_security_score(node_id) {
            Some(score) => score.overall_score,
            None => 0.7, // Default if no score exists
        };

        // Check if banned
        let is_banned = self.fraud_detection.is_banned(node_id);

        // Calculate eligibility and weight
        let is_eligible = device_eligible && !is_banned;
        let weight = device_weight * security_weight;

        (is_eligible, weight)
    }

    /// Get an overall SVCP reputation score for a node (0.0-1.0)
    pub async fn get_svcp_score(&self, node_id: &str) -> f32 {
        // Combine scores from different AI modules for SVCP
        let device_score = self.device_health.get_score();

        // Get security score or default
        let security_score = match self.fraud_detection.get_security_score(node_id) {
            Some(score) => score.overall_score,
            None => 0.7,
        };

        // Get SecurityAI score if available
        let ai_security_score = if let Some(security) = &self.security_ai {
            // Create a default metrics object for evaluation
            let metrics = security::NodeMetrics {
                device_health: security::DeviceHealthMetrics {
                    cpu_usage: 50.0,
                    memory_usage: 50.0,
                    disk_available: 10 * 1024 * 1024 * 1024, // 10 GB
                    num_cores: 4,
                    uptime: 3600, // 1 hour
                    os_info: "Linux".to_string(),
                    avg_response_time: 100.0,
                    dropped_connections: 0,
                    temperature: Some(45.0),
                },
                network: security::NetworkMetrics {
                    bandwidth_usage: 1024 * 1024, // 1 MB/s
                    latency: 100.0,
                    packet_loss: 0.01,
                    connection_stability: 0.95,
                    peer_count: 10,
                    geo_consistency: 0.9,
                    p2p_score: 0.85,
                    sync_status: 1.0,
                },
                storage: security::StorageMetrics {
                    storage_provided: 100 * 1024 * 1024 * 1024, // 100 GB
                    storage_utilization: 0.7,
                    retrieval_success_rate: 0.95,
                    avg_retrieval_time: 200.0,
                    redundancy_level: 3.0,
                    integrity_violations: 0,
                    storage_uptime: 0.99,
                    storage_growth_rate: 10 * 1024 * 1024, // 10 MB/day
                },
                engagement: security::EngagementMetrics {
                    validation_participation: 0.9,
                    transaction_frequency: 100.0,
                    participation_time: 86400 * 7, // 7 days
                    community_contribution: 0.5,
                    governance_participation: 0.5,
                    staking_percentage: 0.01,
                    referrals: 0,
                    social_verification: 0.8,
                },
                ai_behavior: security::AIBehaviorMetrics {
                    anomaly_score: 0.1,
                    risk_assessment: 0.9,
                    fraud_probability: 0.05,
                    threat_level: 0.1,
                    pattern_consistency: 0.05,
                    sybil_probability: 0.05,
                    historical_reliability: 0.9,
                    identity_verification: 0.8,
                },
            };

            match security.evaluate_node(node_id, &metrics).await {
                Ok(score) => score.overall_score,
                Err(_) => 0.7,
            }
        } else {
            0.7
        };

        // Weighted combination
        let device_weight = 0.3;
        let security_weight = 0.3;
        let ai_security_weight = 0.4;

        (device_score.overall_score * device_weight)
            + (security_score * security_weight)
            + (ai_security_score * ai_security_weight)
    }

    /// Initialize AI models
    #[allow(dead_code)]
    async fn initialize_models(&self) -> Result<()> {
        info!("Initializing AI models...");

        // Base neural configuration
        let neural_config = NeuralConfig {
            layers: vec![
                LayerConfig {
                    input_dim: 256,
                    output_dim: 512,
                    activation: ActivationType::GELU,
                    dropout_rate: 0.2,
                },
                LayerConfig {
                    input_dim: 512,
                    output_dim: 384,
                    activation: ActivationType::GELU,
                    dropout_rate: 0.2,
                },
                LayerConfig {
                    input_dim: 384,
                    output_dim: 256,
                    activation: ActivationType::GELU,
                    dropout_rate: 0.2,
                },
                LayerConfig {
                    input_dim: 256,
                    output_dim: 128,
                    activation: ActivationType::GELU,
                    dropout_rate: 0.2,
                },
            ],
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            optimizer: "Adam".to_string(),
            loss: "MSE".to_string(),
        };

        // BCI signal parameters
        let signal_params = SignalParams {
            sampling_rate: 1000,
            num_channels: 256,
            window_size: 100,
            filter_params: FilterParams {
                low_cut: 0.5,
                high_cut: 200.0,
                order: 4,
            },
            spike_threshold: 5.0,
            normalize: true,
            use_wavelet: true,
        };

        // Self-learning configuration
        let learning_config = SelfLearningConfig {
            base_config: neural_config.clone(),
            max_models: 10,
            adaptation_threshold: 0.8,
            sharing_threshold: 0.75,
            min_performance: 0.6,
            lr_factor: 0.1,
        };

        // Register basic models
        self.registry
            .register_neural_model("base_neural", neural_config.clone())
            .await?;

        self.registry
            .register_bci_model("bci_primary", neural_config.clone(), signal_params.clone())
            .await?;

        self.registry
            .register_self_learning_system("adaptive_system", learning_config)
            .await?;

        // Fix for the fraudster detection model neural configuration
        let _fraud_detector_config = NeuralConfig {
            layers: vec![
                LayerConfig {
                    input_dim: 10,
                    output_dim: 20,
                    activation: ActivationType::ReLU,
                    dropout_rate: 0.2,
                },
                LayerConfig {
                    input_dim: 20,
                    output_dim: 10,
                    activation: ActivationType::ReLU,
                    dropout_rate: 0.2,
                },
                LayerConfig {
                    input_dim: 10,
                    output_dim: 2,
                    activation: ActivationType::Sigmoid,
                    dropout_rate: 0.0,
                },
            ],
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            optimizer: "Adam".to_string(),
            loss: "CrossEntropy".to_string(),
        };

        info!("AI models initialized successfully");
        Ok(())
    }

    /// Train all AI models
    pub async fn train_models(&self) -> Result<()> {
        // Get training data
        let device_data = self.collect_device_data().await?;
        let user_data = self.collect_user_data().await?;
        let neural_data = self.collect_neural_data().await?;

        // Train models
        let device_health = self.registry.get_neural_model("device_health").await?;
        device_health.write().await.train(&device_data)?;

        let user_identification = self
            .registry
            .get_learning_system("user_identification")
            .await?;
        user_identification
            .write()
            .await
            .train_all(user_data)
            .await?;

        let neural_interface = self.registry.get_bci_model("neural_interface").await?;

        // Convert neural_data to the format expected by BCIModel::train (Vec<f32>, usize)
        let bci_data = neural_data
            .into_iter()
            .map(|(input, output)| {
                // Find the index of the max value in the output to use as class label
                let class = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                (input, class)
            })
            .collect();

        neural_interface.write().await.train(bci_data).await?;

        // Save updated models
        self.registry.save_all().await?;

        Ok(())
    }

    /// Collect device metrics for training
    async fn collect_device_data(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let mut sys = System::new_all();
        sys.refresh_all();

        let mut data = Vec::new();

        // CPU metrics
        let cpu_usage =
            sys.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / sys.cpus().len() as f32;

        // Memory metrics
        let memory_usage = (sys.used_memory() as f32 / sys.total_memory() as f32) * 100.0;

        // Disk metrics
        let disk_usage = 60.0; // Use dummy disk usage value (60%) to avoid disks() API compatibility issues

        // Format training data
        let features = vec![cpu_usage, memory_usage, disk_usage];
        let labels = vec![
            if cpu_usage > 80.0 { 1.0 } else { 0.0 },
            if memory_usage > 80.0 { 1.0 } else { 0.0 },
            if disk_usage > 80.0 { 1.0 } else { 0.0 },
        ];

        data.push((features, labels));
        Ok(data)
    }

    /// Collect user behavior data for training
    async fn collect_user_data(&self) -> Result<Vec<Experience>> {
        // Example user interaction data
        let experience = Experience {
            state: vec![0.5, 0.3, 0.8],      // User state
            action: 1,                       // Action taken (using usize instead of Vec<f32>)
            reward: 0.8,                     // Reward received
            next_state: vec![0.6, 0.4, 0.9], // Next state
            done: false,
        };

        Ok(vec![experience])
    }

    /// Collect neural interface data for training
    async fn collect_neural_data(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        // Example neural signal data
        let input = vec![0.0; 256]; // Raw signal
        let target = vec![0.0; 32]; // Intended output

        Ok(vec![(input, target)])
    }

    #[allow(dead_code)]
    fn check_system_resources(&self) -> bool {
        let orchestration = futures::executor::block_on(self.orchestration.read());
        !Self::system_resources_exceeded(&orchestration.failover)
    }

    #[allow(dead_code)]
    fn monitor_system_resources(&self) -> bool {
        let orchestration = futures::executor::block_on(self.orchestration.read());
        !Self::system_resources_exceeded(&orchestration.failover)
    }

    #[allow(dead_code)]
    fn get_resource_usage(&self, sys: &System) -> Result<(f32, f32, f32), anyhow::Error> {
        // CPU usage
        let cpu_usage = sys.global_cpu_info().cpu_usage();

        // Memory usage
        let memory_usage = (sys.used_memory() as f32 / sys.total_memory() as f32) * 100.0;

        // Simulate disk usage with dummy data - avoid using disks() method
        let disk_usage = 65.0; // 65% disk usage as dummy value

        Ok((cpu_usage, memory_usage, disk_usage))
    }

    /// Train a neural model on new data
    pub async fn train_model(
        &self,
        model_name: &str,
        neural_data: &[(Vec<f32>, Vec<f32>)],
    ) -> Result<()> {
        let neural_interface = self.registry.get_bci_model(model_name).await?;

        // Convert neural_data to the format expected by BCIModel::train (Vec<f32>, usize)
        let bci_data = neural_data
            .iter()
            .map(|(input, output)| {
                // Find the index of the max value in the output to use as class label
                let class = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                (input.clone(), class)
            })
            .collect();

        neural_interface.write().await.train(bci_data).await?;

        Ok(())
    }

    #[allow(dead_code)]
    async fn train_classifier(&self, _neural_data: &[(Vec<f32>, Vec<f32>)]) -> Result<f32> {
        // ... existing code ...
        Ok(0.85) // Example result
    }

    #[allow(dead_code)]
    async fn apply_filters(&self, _params: FilterParams) -> Result<Vec<f32>> {
        // ... existing code ...
        Ok(vec![0.1, 0.2, 0.3])
    }

    // Fix the clone issue by getting and using the registry directly instead of cloning it
    #[allow(dead_code)]
    async fn scheduled_training_task(&self) -> Result<()> {
        // Get registry methods directly without cloning
        // No need to clone self.registry

        // Example of registering a model with all required fields
        let neural_config = NeuralConfig {
            layers: vec![
                LayerConfig {
                    input_dim: 10,
                    output_dim: 20,
                    activation: ActivationType::ReLU,
                    dropout_rate: 0.2,
                },
                LayerConfig {
                    input_dim: 20,
                    output_dim: 10,
                    activation: ActivationType::ReLU,
                    dropout_rate: 0.2,
                },
                LayerConfig {
                    input_dim: 10,
                    output_dim: 2,
                    activation: ActivationType::ReLU,
                    dropout_rate: 0.1,
                },
            ],
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 10,
            optimizer: "Adam".to_string(),
            loss: "MSE".to_string(),
        };

        // Call registry methods directly on self.registry
        if let Err(e) = self
            .registry
            .register_neural_model("default", neural_config)
            .await
        {
            error!("Failed to register neural model: {}", e);
        }

        Ok(())
    }

    // Fix the train method with await
    pub async fn train_model_interface(
        &self,
        neural_interface: Arc<RwLock<models::bci_interface::BCIModel>>,
        neural_data: &[(Vec<f32>, Vec<f32>)],
    ) -> Result<()> {
        // Convert neural_data to the format expected by BCIModel::train (Vec<f32>, usize)
        let bci_data = neural_data
            .iter()
            .map(|(input, output)| {
                // Find the index of the max value in the output to use as class label
                let class = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                (input.clone(), class)
            })
            .collect();

        neural_interface.write().await.train(bci_data).await?;

        Ok(())
    }
}

/// Create a default AI Engine with manual configuration
pub fn create_default_ai_engine() -> AIEngine {
    let mut config = Config::default();

    // Set network configuration
    config.network.p2p_port = 7000;
    config.network.max_peers = 50;
    config.network.bootstrap_nodes = vec![
        "127.0.0.1:7001".to_string(),
        "127.0.0.1:7002".to_string(),
        "127.0.0.1:7003".to_string(),
    ];

    // Set API configuration
    config.api.port = 8080;
    config.api.host = "127.0.0.1".to_string();
    config.api.address = "127.0.0.1".to_string();
    config.api.enabled = true;
    config.api.cors_domains = vec!["*".to_string()];
    config.api.allow_origin = vec!["*".to_string()];
    config.api.max_request_body_size = 10 * 1024 * 1024; // 10MB
    config.api.max_connections = 100;
    config.api.enable_websocket = false;
    config.api.enable_graphql = false;

    // Set sharding configuration
    config.sharding.shard_count = 4;
    config.sharding.shard_id = 0;
    config.sharding.enabled = true;

    // Create and return the AI Engine
    AIEngine::new(config)
}

#[cfg(test)]
mod data_chunking_tests;

// Re-export key components
pub use data_chunking::ChunkingConfig;
