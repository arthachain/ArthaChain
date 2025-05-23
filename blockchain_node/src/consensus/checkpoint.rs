use crate::ledger::block::Block;
use crate::ledger::state::State;
use crate::network::types::NodeId;
use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};

/// Configuration for the checkpoint system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Checkpoint interval by block count
    pub checkpoint_interval_blocks: u64,
    /// Maximum checkpoints to keep
    pub max_checkpoints: usize,
    /// Checkpoint storage directory
    pub storage_dir: String,
    /// Minimum number of signatures required for validation
    pub min_signatures: usize,
    /// Enable automatic pruning of old checkpoints
    pub enable_pruning: bool,
    /// Enable compression
    pub enable_compression: bool,
    /// Maximum size of a checkpoint in bytes
    pub max_checkpoint_size_bytes: usize,
    /// Storage format
    pub storage_format: CheckpointFormat,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval_blocks: 1000,
            max_checkpoints: 10,
            storage_dir: "data/checkpoints".to_string(),
            min_signatures: 3,
            enable_pruning: true,
            enable_compression: true,
            max_checkpoint_size_bytes: 1024 * 1024 * 100, // 100 MB
            storage_format: CheckpointFormat::Binary,
        }
    }
}

/// Checkpoint storage format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointFormat {
    /// Binary format
    Binary,
    /// JSON format
    Json,
    /// CBOR format
    Cbor,
}

/// Checkpoint data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint ID
    pub id: u64,
    /// Block hash at checkpoint
    pub block_hash: Vec<u8>,
    /// Block height at checkpoint
    pub block_height: u64,
    /// Timestamp of creation
    pub timestamp: u64,
    /// Serialized state data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_data: Option<Vec<u8>>,
    /// Hash of state data
    pub state_hash: Vec<u8>,
    /// Signatures of validators
    pub signatures: HashMap<NodeId, Vec<u8>>,
    /// Creation metadata
    pub metadata: CheckpointMetadata,
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Creator node ID
    pub creator: NodeId,
    /// Network identifier
    pub network_id: String,
    /// Software version
    pub version: String,
    /// Size of state in bytes
    pub state_size_bytes: usize,
    /// Additional info
    pub additional_info: HashMap<String, String>,
}

/// Manager for blockchain state checkpoints
pub struct CheckpointManager {
    /// Configuration
    config: RwLock<CheckpointConfig>,
    /// Latest checkpoints
    checkpoints: RwLock<HashMap<u64, Checkpoint>>,
    /// Active validators
    validators: Arc<RwLock<HashSet<NodeId>>>,
    /// Blockchain state
    state: Arc<RwLock<State>>,
    /// Next checkpoint ID
    next_checkpoint_id: RwLock<u64>,
    /// Running flag
    running: RwLock<bool>,
    /// Channel for receiving new blocks
    block_receiver: Option<mpsc::Receiver<Block>>,
    /// Node ID
    node_id: NodeId,
    /// Last checkpoint time
    last_checkpoint_time: RwLock<Instant>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(
        config: CheckpointConfig,
        validators: Arc<RwLock<HashSet<NodeId>>>,
        state: Arc<RwLock<State>>,
        node_id: NodeId,
        block_receiver: Option<mpsc::Receiver<Block>>,
    ) -> Self {
        Self {
            config: RwLock::new(config),
            checkpoints: RwLock::new(HashMap::new()),
            validators,
            state,
            next_checkpoint_id: RwLock::new(0),
            running: RwLock::new(false),
            block_receiver,
            node_id,
            last_checkpoint_time: RwLock::new(Instant::now()),
        }
    }

    /// Start the checkpoint manager
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(anyhow!("Checkpoint manager already running"));
        }

        *running = true;

        // Initialize the checkpoint directory
        self.initialize_storage().await?;

        // Load existing checkpoints
        self.load_checkpoints().await?;

        // Start the checkpoint creation task if we have a block receiver
        if let Some(receiver) = self.block_receiver.take() {
            self.start_checkpoint_task(receiver);
        }

        info!("Checkpoint manager started");
        Ok(())
    }

    /// Stop the checkpoint manager
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(anyhow!("Checkpoint manager not running"));
        }

        *running = false;
        info!("Checkpoint manager stopped");
        Ok(())
    }

    /// Initialize the checkpoint storage directory
    async fn initialize_storage(&self) -> Result<()> {
        let config = self.config.read().await;
        let dir = PathBuf::from(&config.storage_dir);

        if !dir.exists() {
            tokio::fs::create_dir_all(&dir).await?;
        }

        Ok(())
    }

    /// Load existing checkpoints from storage
    async fn load_checkpoints(&self) -> Result<()> {
        let config = self.config.read().await;
        let dir = PathBuf::from(&config.storage_dir);

        let mut entries = tokio::fs::read_dir(&dir).await?;
        let mut loaded_checkpoints = HashMap::new();
        let mut highest_id = 0;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "checkpoint") {
                // Parse the checkpoint ID from filename
                if let Some(file_stem) = path.file_stem() {
                    if let Some(file_name) = file_stem.to_str() {
                        if let Ok(id) = file_name.parse::<u64>() {
                            // Load the checkpoint
                            let data = tokio::fs::read(&path).await?;
                            let checkpoint: Checkpoint = match config.storage_format {
                                CheckpointFormat::Binary => bincode::deserialize(&data)?,
                                CheckpointFormat::Json => serde_json::from_slice(&data)?,
                                CheckpointFormat::Cbor => serde_cbor::from_slice(&data)?,
                            };

                            loaded_checkpoints.insert(id, checkpoint);
                            highest_id = highest_id.max(id);
                        }
                    }
                }
            }
        }

        // Update the next checkpoint ID
        let mut next_id = self.next_checkpoint_id.write().await;
        *next_id = highest_id + 1;

        // Store the loaded checkpoints
        let mut checkpoints = self.checkpoints.write().await;
        *checkpoints = loaded_checkpoints;

        info!("Loaded {} checkpoints from storage", checkpoints.len());

        // Prune old checkpoints if necessary
        if config.enable_pruning {
            self.prune_old_checkpoints().await?;
        }

        Ok(())
    }

    /// Start the checkpoint creation task
    fn start_checkpoint_task(&self, mut block_receiver: mpsc::Receiver<Block>) {
        let self_clone = Arc::new(self.clone());

        tokio::spawn(async move {
            let mut last_checkpoint_height = 0;

            while let Some(block) = block_receiver.recv().await {
                let is_running = *self_clone.running.read().await;
                if !is_running {
                    break;
                }

                let should_checkpoint = {
                    let config = self_clone.config.read().await;
                    block.height > 0
                        && block.height - last_checkpoint_height
                            >= config.checkpoint_interval_blocks
                };

                if should_checkpoint {
                    match self_clone.create_checkpoint(block.clone()).await {
                        Ok(checkpoint) => {
                            info!(
                                "Created checkpoint {} at block height {}",
                                checkpoint.id, checkpoint.block_height
                            );
                            last_checkpoint_height = block.height;
                        }
                        Err(e) => {
                            warn!("Failed to create checkpoint: {}", e);
                        }
                    }
                }
            }
        });
    }

    /// Create a new checkpoint
    pub async fn create_checkpoint(&self, block: Block) -> Result<Checkpoint> {
        // Get the current state
        let state = self.state.read().await;

        // Serialize the state
        let state_data = state.serialize().await?;
        let state_hash = compute_state_hash(&state_data);

        // Create the checkpoint
        let id = {
            let mut next_id = self.next_checkpoint_id.write().await;
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Create metadata
        let metadata = CheckpointMetadata {
            creator: self.node_id.clone(),
            network_id: state.network_id.clone(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            state_size_bytes: state_data.len(),
            additional_info: HashMap::new(),
        };

        // Create checkpoint
        let mut checkpoint = Checkpoint {
            id,
            block_hash: block.hash.clone(),
            block_height: block.height,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            state_data: Some(state_data),
            state_hash,
            signatures: HashMap::new(),
            metadata,
        };

        // Sign the checkpoint with our node ID
        let signature = self.sign_checkpoint(&checkpoint)?;
        checkpoint
            .signatures
            .insert(self.node_id.clone(), signature);

        // Save the checkpoint
        self.save_checkpoint(&checkpoint).await?;

        // Store in memory
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.insert(id, checkpoint.clone());

        // Update the last checkpoint time
        let mut last_time = self.last_checkpoint_time.write().await;
        *last_time = Instant::now();

        // Prune old checkpoints if necessary
        drop(checkpoints); // Release lock before pruning
        let config = self.config.read().await;
        if config.enable_pruning {
            self.prune_old_checkpoints().await?;
        }

        Ok(checkpoint)
    }

    /// Sign a checkpoint
    fn sign_checkpoint(&self, checkpoint: &Checkpoint) -> Result<Vec<u8>> {
        // In a real implementation, this would use a proper signature scheme
        // For now, we'll just create a dummy signature
        let mut data_to_sign = Vec::new();
        data_to_sign.extend_from_slice(&checkpoint.id.to_be_bytes());
        data_to_sign.extend_from_slice(&checkpoint.block_hash);
        data_to_sign.extend_from_slice(&checkpoint.state_hash);

        // Example signature (in a real implementation, this would use proper crypto)
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data_to_sign);
        let signature = hasher.finalize().to_vec();

        Ok(signature)
    }

    /// Save a checkpoint to storage
    async fn save_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
        let config = self.config.read().await;
        let dir = PathBuf::from(&config.storage_dir);
        let path = dir.join(format!("{}.checkpoint", checkpoint.id));

        // Create a checkpoint with or without state data based on configuration
        let checkpoint_to_save = if config.max_checkpoint_size_bytes > 0
            && checkpoint.state_data.as_ref().map_or(0, |d| d.len())
                > config.max_checkpoint_size_bytes
        {
            // Remove state data if it exceeds the maximum size
            let mut cp = checkpoint.clone();
            cp.state_data = None;
            cp
        } else {
            checkpoint.clone()
        };

        // Serialize the checkpoint
        let data = match config.storage_format {
            CheckpointFormat::Binary => bincode::serialize(&checkpoint_to_save)?,
            CheckpointFormat::Json => serde_json::to_vec(&checkpoint_to_save)?,
            CheckpointFormat::Cbor => serde_cbor::to_vec(&checkpoint_to_save)?,
        };

        // Compress if enabled
        let final_data = if config.enable_compression {
            use flate2::{write::GzEncoder, Compression};
            use std::io::Write;

            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&data)?;
            encoder.finish()?
        } else {
            data
        };

        // Write to disk
        tokio::fs::write(&path, &final_data).await?;

        debug!("Saved checkpoint {} to {}", checkpoint.id, path.display());
        Ok(())
    }

    /// Prune old checkpoints to stay within max_checkpoints limit
    async fn prune_old_checkpoints(&self) -> Result<()> {
        let config = self.config.read().await;
        let max_checkpoints = config.max_checkpoints;

        let checkpoints_to_prune = {
            let checkpoints = self.checkpoints.read().await;

            if checkpoints.len() <= max_checkpoints {
                return Ok(()); // No pruning needed
            }

            // Identify the oldest checkpoints to prune
            let mut checkpoint_ids: Vec<u64> = checkpoints.keys().cloned().collect();
            checkpoint_ids.sort();

            let num_to_prune = checkpoints.len() - max_checkpoints;
            checkpoint_ids
                .into_iter()
                .take(num_to_prune)
                .collect::<Vec<u64>>()
        };

        if checkpoints_to_prune.is_empty() {
            return Ok(());
        }

        // Remove from storage
        let dir = PathBuf::from(&config.storage_dir);
        for id in &checkpoints_to_prune {
            let path = dir.join(format!("{}.checkpoint", id));
            if tokio::fs::metadata(&path).await.is_ok() {
                tokio::fs::remove_file(&path).await?;
            }
        }

        // Remove from memory
        let mut checkpoints = self.checkpoints.write().await;
        for id in &checkpoints_to_prune {
            checkpoints.remove(id);
        }

        info!("Pruned {} old checkpoints", checkpoints_to_prune.len());
        Ok(())
    }

    /// Verify a checkpoint's signatures
    pub async fn verify_checkpoint(&self, checkpoint: &Checkpoint) -> Result<bool> {
        let validators = self.validators.read().await;
        let config = self.config.read().await;

        // Check that we have enough signatures
        let valid_signatures = checkpoint
            .signatures
            .iter()
            .filter(|(node_id, _)| validators.contains(*node_id))
            .count();

        if valid_signatures < config.min_signatures {
            return Ok(false);
        }

        // In a real implementation, we would verify each signature

        Ok(true)
    }

    /// Apply a checkpoint to restore state
    pub async fn apply_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
        // Verify the checkpoint signatures
        if !self.verify_checkpoint(checkpoint).await? {
            return Err(anyhow!("Invalid checkpoint signatures"));
        }

        // Check if we have state data
        let state_data = if let Some(data) = &checkpoint.state_data {
            data.clone()
        } else {
            // Try to load from storage
            let config = self.config.read().await;
            let dir = PathBuf::from(&config.storage_dir);
            let path = dir.join(format!("{}.checkpoint", checkpoint.id));

            if !path.exists() {
                return Err(anyhow!("Checkpoint state data not available"));
            }

            let data = tokio::fs::read(&path).await?;

            // Decompress if needed
            let decompressed = if config.enable_compression {
                use flate2::read::GzDecoder;
                use std::io::Read;

                let mut decoder = GzDecoder::new(&data[..]);
                let mut decompressed_data = Vec::new();
                decoder.read_to_end(&mut decompressed_data)?;
                decompressed_data
            } else {
                data
            };

            // Deserialize to get the checkpoint
            let loaded_checkpoint: Checkpoint = match config.storage_format {
                CheckpointFormat::Binary => bincode::deserialize(&decompressed)?,
                CheckpointFormat::Json => serde_json::from_slice(&decompressed)?,
                CheckpointFormat::Cbor => serde_cbor::from_slice(&decompressed)?,
            };

            if let Some(data) = loaded_checkpoint.state_data {
                data
            } else {
                return Err(anyhow!("Checkpoint state data not available"));
            }
        };

        // Verify state hash
        let computed_hash = compute_state_hash(&state_data);
        if computed_hash != checkpoint.state_hash {
            return Err(anyhow!("State hash mismatch"));
        }

        // Apply the state
        let mut state = self.state.write().await;
        state.deserialize(&state_data).await?;

        info!(
            "Applied checkpoint {} (block height: {})",
            checkpoint.id, checkpoint.block_height
        );
        Ok(())
    }

    /// Get a checkpoint by ID
    pub async fn get_checkpoint(&self, id: u64) -> Option<Checkpoint> {
        let checkpoints = self.checkpoints.read().await;
        checkpoints.get(&id).cloned()
    }

    /// Get all checkpoints
    pub async fn get_all_checkpoints(&self) -> Vec<Checkpoint> {
        let checkpoints = self.checkpoints.read().await;
        let mut result: Vec<_> = checkpoints.values().cloned().collect();
        result.sort_by_key(|c| c.id);
        result
    }

    /// Get the latest checkpoint
    pub async fn get_latest_checkpoint(&self) -> Option<Checkpoint> {
        let checkpoints = self.checkpoints.read().await;
        if checkpoints.is_empty() {
            return None;
        }

        let latest_id = *checkpoints.keys().max().unwrap();
        checkpoints.get(&latest_id).cloned()
    }

    /// Add a signature to a checkpoint
    pub async fn add_signature(
        &self,
        checkpoint_id: u64,
        node_id: NodeId,
        signature: Vec<u8>,
    ) -> Result<()> {
        let mut checkpoints = self.checkpoints.write().await;

        if let Some(checkpoint) = checkpoints.get_mut(&checkpoint_id) {
            checkpoint.signatures.insert(node_id, signature);

            // Save the updated checkpoint
            drop(checkpoints); // Release lock before saving
            let updated = self.get_checkpoint(checkpoint_id).await.unwrap();
            self.save_checkpoint(&updated).await?;

            Ok(())
        } else {
            Err(anyhow!("Checkpoint {} not found", checkpoint_id))
        }
    }

    /// Update configuration
    pub async fn update_config(&self, config: CheckpointConfig) -> Result<()> {
        let mut cfg = self.config.write().await;

        // Check if storage directory changed
        let dir_changed = cfg.storage_dir != config.storage_dir;

        // Update config
        *cfg = config;

        // If storage directory changed, reinitialize
        if dir_changed {
            drop(cfg); // Release lock
            self.initialize_storage().await?;
            self.load_checkpoints().await?;
        }

        Ok(())
    }
}

impl Clone for CheckpointManager {
    fn clone(&self) -> Self {
        // Partial clone for use in async tasks
        Self {
            config: RwLock::new(self.config.try_read().unwrap_or_default().clone()),
            checkpoints: RwLock::new(HashMap::new()),
            validators: self.validators.clone(),
            state: self.state.clone(),
            next_checkpoint_id: RwLock::new(0),
            running: RwLock::new(false),
            block_receiver: None,
            node_id: self.node_id.clone(),
            last_checkpoint_time: RwLock::new(Instant::now()),
        }
    }
}

/// Compute a hash of the state data
fn compute_state_hash(data: &[u8]) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}
