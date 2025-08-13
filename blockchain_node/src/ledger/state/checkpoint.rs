use crate::ledger::state::State;
use crate::storage::Storage;
use crate::types::Hash;
use anyhow::{anyhow, Result};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use tokio::time::interval;

/// Configuration for state checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Directory to store checkpoints
    pub checkpoint_dir: PathBuf,
    /// Interval between checkpoints in seconds
    pub checkpoint_interval_secs: u64,
    /// Maximum number of checkpoints to retain
    pub max_checkpoints: usize,
    /// Enable automatic checkpointing
    pub auto_checkpoint: bool,
    /// Checkpoint on every N blocks
    pub checkpoint_block_interval: u64,
    /// Enable compression for checkpoints
    pub compression_enabled: bool,
    /// Verify checkpoint integrity on save
    pub verify_on_save: bool,
    /// Verify checkpoint integrity on load
    pub verify_on_load: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("data/checkpoints"),
            checkpoint_interval_secs: 3600, // 1 hour
            max_checkpoints: 24,            // Keep 24 checkpoints
            auto_checkpoint: true,
            checkpoint_block_interval: 1000, // Every 1000 blocks
            compression_enabled: true,
            verify_on_save: true,
            verify_on_load: true,
        }
    }
}

/// Metadata for a state checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Unique checkpoint ID
    pub id: String,
    /// Timestamp when checkpoint was created
    pub timestamp: SystemTime,
    /// Block height at checkpoint
    pub block_height: u64,
    /// State root hash
    pub state_root: Hash,
    /// Size of checkpoint in bytes
    pub size: u64,
    /// Checksum for integrity verification
    pub checksum: String,
    /// Previous checkpoint ID (for chain verification)
    pub previous_checkpoint: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// State checkpoint manager
pub struct CheckpointManager {
    /// Configuration
    config: CheckpointConfig,
    /// Checkpoint metadata storage
    checkpoints: Arc<RwLock<VecDeque<CheckpointMetadata>>>,
    /// Active checkpoint in progress
    active_checkpoint: Arc<Mutex<Option<String>>>,
    /// Checkpoint scheduler handle
    scheduler_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Last checkpoint time
    last_checkpoint: Arc<RwLock<SystemTime>>,
    /// Checkpoint verification cache
    verification_cache: Arc<RwLock<HashMap<String, bool>>>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: CheckpointConfig) -> Result<Self> {
        // Create checkpoint directory
        std::fs::create_dir_all(&config.checkpoint_dir)?;

        // Load existing checkpoint metadata
        let checkpoints = Self::load_checkpoint_metadata(&config.checkpoint_dir)?;

        Ok(Self {
            config,
            checkpoints: Arc::new(RwLock::new(checkpoints)),
            active_checkpoint: Arc::new(Mutex::new(None)),
            scheduler_handle: Arc::new(Mutex::new(None)),
            last_checkpoint: Arc::new(RwLock::new(SystemTime::now())),
            verification_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Start automatic checkpointing
    pub async fn start_auto_checkpoint(&self, state: Arc<State>) -> Result<()> {
        if !self.config.auto_checkpoint {
            return Ok(());
        }

        let config = self.config.clone();
        let checkpoints = self.checkpoints.clone();
        let active_checkpoint = self.active_checkpoint.clone();
        let last_checkpoint = self.last_checkpoint.clone();
        let manager_self = Arc::new(self.clone_internal());

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.checkpoint_interval_secs));

            loop {
                interval.tick().await;

                // Check if checkpoint is needed
                let should_checkpoint = {
                    let last = last_checkpoint.read().await;
                    let elapsed = SystemTime::now()
                        .duration_since(*last)
                        .unwrap_or(Duration::from_secs(0));
                    elapsed.as_secs() >= config.checkpoint_interval_secs
                };

                if should_checkpoint {
                    if let Err(e) = manager_self.create_checkpoint(&state).await {
                        error!("Failed to create automatic checkpoint: {}", e);
                    }
                }
            }
        });

        *self.scheduler_handle.lock().await = Some(handle);
        info!("Started automatic checkpoint scheduler");
        Ok(())
    }

    /// Create a checkpoint of the current state
    pub async fn create_checkpoint(&self, state: &Arc<State>) -> Result<CheckpointMetadata> {
        // Check if another checkpoint is in progress
        let mut active = self.active_checkpoint.lock().await;
        if active.is_some() {
            return Err(anyhow!("Another checkpoint is already in progress"));
        }

        let checkpoint_id = format!(
            "checkpoint_{}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()
        );
        *active = Some(checkpoint_id.clone());

        // Create checkpoint
        let result = self.create_checkpoint_internal(state, &checkpoint_id).await;

        // Clear active checkpoint
        *active = None;

        result
    }

    /// Internal checkpoint creation
    async fn create_checkpoint_internal(
        &self,
        state: &Arc<State>,
        checkpoint_id: &str,
    ) -> Result<CheckpointMetadata> {
        info!("Creating checkpoint: {}", checkpoint_id);

        let checkpoint_path = self.config.checkpoint_dir.join(checkpoint_id);
        std::fs::create_dir_all(&checkpoint_path)?;

        // Get state data
        let block_height = state.get_height().unwrap_or(0);
        let state_root = state.get_state_root()?;

        // Serialize state components
        let mut total_size = 0u64;
        let mut hasher = blake3::Hasher::new();

        // Save accounts
        let accounts_data = state.export_accounts().await?;
        let accounts_path = checkpoint_path.join("accounts.dat");
        if self.config.compression_enabled {
            let compressed = zstd::encode_all(std::io::Cursor::new(accounts_data.as_slice()), 3)?;
            std::fs::write(&accounts_path, &compressed)?;
            total_size += compressed.len() as u64;
            hasher.update(&compressed);
        } else {
            std::fs::write(&accounts_path, &accounts_data)?;
            total_size += accounts_data.len() as u64;
            hasher.update(&accounts_data);
        }

        // Save storage
        let storage_data = state.export_storage().await?;
        let storage_path = checkpoint_path.join("storage.dat");
        if self.config.compression_enabled {
            let compressed = zstd::encode_all(std::io::Cursor::new(storage_data.as_slice()), 3)?;
            std::fs::write(&storage_path, &compressed)?;
            total_size += compressed.len() as u64;
            hasher.update(&compressed);
        } else {
            std::fs::write(&storage_path, &storage_data)?;
            total_size += storage_data.len() as u64;
            hasher.update(&storage_data);
        }

        // Save processed transactions
        let tx_data = state.export_processed_transactions().await?;
        let tx_path = checkpoint_path.join("transactions.dat");
        if self.config.compression_enabled {
            let compressed = zstd::encode_all(std::io::Cursor::new(tx_data.as_slice()), 3)?;
            std::fs::write(&tx_path, &compressed)?;
            total_size += compressed.len() as u64;
            hasher.update(&compressed);
        } else {
            std::fs::write(&tx_path, &tx_data)?;
            total_size += tx_data.len() as u64;
            hasher.update(&tx_data);
        }

        let checksum = hex::encode(hasher.finalize().as_bytes());

        // Get previous checkpoint
        let previous_checkpoint = {
            let checkpoints = self.checkpoints.read().await;
            checkpoints.back().map(|cp| cp.id.clone())
        };

        // Create metadata
        let metadata = CheckpointMetadata {
            id: checkpoint_id.to_string(),
            timestamp: SystemTime::now(),
            block_height,
            state_root,
            size: total_size,
            checksum: checksum.clone(),
            previous_checkpoint,
            metadata: HashMap::new(),
        };

        // Save metadata
        let metadata_path = checkpoint_path.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(metadata_path, metadata_json)?;

        // Verify if enabled
        if self.config.verify_on_save {
            self.verify_checkpoint(&metadata).await?;
        }

        // Add to checkpoint list
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.push_back(metadata.clone());

        // Remove old checkpoints
        while checkpoints.len() > self.config.max_checkpoints {
            if let Some(old_checkpoint) = checkpoints.pop_front() {
                let old_path = self.config.checkpoint_dir.join(&old_checkpoint.id);
                if let Err(e) = std::fs::remove_dir_all(&old_path) {
                    warn!("Failed to remove old checkpoint: {}", e);
                }
            }
        }

        // Update last checkpoint time
        *self.last_checkpoint.write().await = SystemTime::now();

        info!(
            "Checkpoint created: {} (height: {}, size: {} bytes)",
            checkpoint_id, block_height, total_size
        );

        Ok(metadata)
    }

    /// Restore state from checkpoint
    pub async fn restore_checkpoint(&self, checkpoint_id: &str, state: &Arc<State>) -> Result<()> {
        info!("Restoring from checkpoint: {}", checkpoint_id);

        // Find checkpoint metadata
        let metadata = {
            let checkpoints = self.checkpoints.read().await;
            checkpoints
                .iter()
                .find(|cp| cp.id == checkpoint_id)
                .cloned()
                .ok_or_else(|| anyhow!("Checkpoint not found: {}", checkpoint_id))?
        };

        // Verify checkpoint if enabled
        if self.config.verify_on_load {
            self.verify_checkpoint(&metadata).await?;
        }

        let checkpoint_path = self.config.checkpoint_dir.join(checkpoint_id);

        // Restore accounts
        let accounts_path = checkpoint_path.join("accounts.dat");
        let accounts_data = if self.config.compression_enabled {
            let compressed = std::fs::read(&accounts_path)?;
            zstd::decode_all(std::io::Cursor::new(compressed.as_slice()))?
        } else {
            std::fs::read(&accounts_path)?
        };
        state.import_accounts(accounts_data).await?;

        // Restore storage
        let storage_path = checkpoint_path.join("storage.dat");
        let storage_data = if self.config.compression_enabled {
            let compressed = std::fs::read(&storage_path)?;
            zstd::decode_all(std::io::Cursor::new(compressed.as_slice()))?
        } else {
            std::fs::read(&storage_path)?
        };
        state.import_storage(storage_data).await?;

        // Restore transactions
        let tx_path = checkpoint_path.join("transactions.dat");
        let tx_data = if self.config.compression_enabled {
            let compressed = std::fs::read(&tx_path)?;
            zstd::decode_all(std::io::Cursor::new(compressed.as_slice()))?
        } else {
            std::fs::read(&tx_path)?
        };
        state.import_processed_transactions(tx_data).await?;

        info!(
            "State restored from checkpoint: {} (height: {})",
            checkpoint_id, metadata.block_height
        );

        Ok(())
    }

    /// Verify checkpoint integrity
    pub async fn verify_checkpoint(&self, metadata: &CheckpointMetadata) -> Result<()> {
        // Check cache first
        if let Some(&verified) = self.verification_cache.read().await.get(&metadata.id) {
            if verified {
                return Ok(());
            }
        }

        let checkpoint_path = self.config.checkpoint_dir.join(&metadata.id);
        let mut hasher = blake3::Hasher::new();
        let mut total_size = 0u64;

        // Verify accounts file
        let accounts_path = checkpoint_path.join("accounts.dat");
        let accounts_data = std::fs::read(&accounts_path)?;
        total_size += accounts_data.len() as u64;
        hasher.update(&accounts_data);

        // Verify storage file
        let storage_path = checkpoint_path.join("storage.dat");
        let storage_data = std::fs::read(&storage_path)?;
        total_size += storage_data.len() as u64;
        hasher.update(&storage_data);

        // Verify transactions file
        let tx_path = checkpoint_path.join("transactions.dat");
        let tx_data = std::fs::read(&tx_path)?;
        total_size += tx_data.len() as u64;
        hasher.update(&tx_data);

        // Verify checksum
        let calculated_checksum = hex::encode(hasher.finalize().as_bytes());
        if calculated_checksum != metadata.checksum {
            return Err(anyhow!("Checkpoint verification failed: checksum mismatch"));
        }

        // Verify size
        if total_size != metadata.size {
            return Err(anyhow!("Checkpoint verification failed: size mismatch"));
        }

        // Cache successful verification
        self.verification_cache
            .write()
            .await
            .insert(metadata.id.clone(), true);

        Ok(())
    }

    /// List available checkpoints
    pub async fn list_checkpoints(&self) -> Vec<CheckpointMetadata> {
        self.checkpoints.read().await.iter().cloned().collect()
    }

    /// Get latest checkpoint
    pub async fn get_latest_checkpoint(&self) -> Option<CheckpointMetadata> {
        self.checkpoints.read().await.back().cloned()
    }

    /// Load checkpoint metadata from disk
    fn load_checkpoint_metadata(checkpoint_dir: &Path) -> Result<VecDeque<CheckpointMetadata>> {
        let mut checkpoints = VecDeque::new();

        if let Ok(entries) = std::fs::read_dir(checkpoint_dir) {
            for entry in entries.flatten() {
                if entry.file_type()?.is_dir() {
                    let metadata_path = entry.path().join("metadata.json");
                    if metadata_path.exists() {
                        if let Ok(metadata_json) = std::fs::read_to_string(&metadata_path) {
                            if let Ok(metadata) =
                                serde_json::from_str::<CheckpointMetadata>(&metadata_json)
                            {
                                checkpoints.push_back(metadata);
                            }
                        }
                    }
                }
            }
        }

        // Sort by timestamp
        let mut sorted: Vec<_> = checkpoints.into_iter().collect();
        sorted.sort_by_key(|cp| cp.timestamp);

        Ok(sorted.into_iter().collect())
    }

    /// Clone internal state for async tasks
    fn clone_internal(&self) -> Self {
        Self {
            config: self.config.clone(),
            checkpoints: self.checkpoints.clone(),
            active_checkpoint: self.active_checkpoint.clone(),
            scheduler_handle: Arc::new(Mutex::new(None)),
            last_checkpoint: self.last_checkpoint.clone(),
            verification_cache: self.verification_cache.clone(),
        }
    }

    /// Stop automatic checkpointing
    pub async fn stop(&self) {
        if let Some(handle) = self.scheduler_handle.lock().await.take() {
            handle.abort();
            info!("Stopped checkpoint scheduler");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[tokio::test]
    async fn test_checkpoint_manager() {
        let config = CheckpointConfig {
            checkpoint_dir: PathBuf::from("/tmp/test_checkpoints"),
            checkpoint_interval_secs: 1,
            max_checkpoints: 2,
            ..Default::default()
        };

        let manager = CheckpointManager::new(config).unwrap();
        let state = Arc::new(State::new(&Config::default()).unwrap());

        // Create checkpoint
        let checkpoint = manager.create_checkpoint(&state).await.unwrap();
        assert!(!checkpoint.id.is_empty());

        // Verify checkpoint
        manager.verify_checkpoint(&checkpoint).await.unwrap();

        // List checkpoints
        let checkpoints = manager.list_checkpoints().await;
        assert_eq!(checkpoints.len(), 1);

        // Clean up
        let _ = std::fs::remove_dir_all("/tmp/test_checkpoints");
    }
}
