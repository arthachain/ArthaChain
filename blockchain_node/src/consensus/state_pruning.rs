use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use rocksdb::{DB, Options};
use bincode;
use crate::utils::crypto::Hash;
use std::path::PathBuf;
use log::debug;
use crate::utils::proofs::verify_proof;
use tokio::fs;

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Minimum blocks to keep
    pub min_blocks: u64,
    /// Maximum blocks to keep
    pub max_blocks: u64,
    /// Pruning interval in blocks
    pub pruning_interval: u64,
    /// Archive interval in blocks
    pub archive_interval: u64,
    /// Archive path
    pub archive_path: PathBuf,
}

/// State transition record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Block height
    pub height: u64,
    /// State root before transition
    pub prev_state_root: Hash,
    /// State root after transition
    pub new_state_root: Hash,
    /// Transition proof
    pub proof: Vec<u8>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Pruning metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningMetrics {
    /// Number of states pruned
    pub pruned_states: u64,
    /// Number of states archived
    pub archived_states: u64,
    /// Total size of pruned states in bytes
    pub pruned_size: u64,
    /// Total size of archived states in bytes
    pub archived_size: u64,
    /// Last pruning duration in milliseconds
    pub last_pruning_duration: u64,
    /// Last archive duration in milliseconds
    pub last_archive_duration: u64,
}

/// Pruning manager state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningState {
    /// Current block height
    pub current_height: u64,
    /// Last pruning height
    pub last_pruning_height: u64,
    /// Last archive height
    pub last_archive_height: u64,
    /// State transitions
    pub transitions: HashMap<u64, StateTransition>,
    /// Configuration
    pub config: PruningConfig,
    /// Metrics
    pub metrics: PruningMetrics,
}

#[derive(Debug, thiserror::Error)]
pub enum PruningError {
    #[error("Database error: {0}")]
    Database(#[from] rocksdb::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Internal error: {0}")]
    Internal(String),
    
    #[error("Invalid proof")]
    InvalidProof,

    #[error("Invalid height")]
    InvalidHeight,

    #[error("Invalid state root")]
    InvalidStateRoot,
}

impl From<anyhow::Error> for PruningError {
    fn from(err: anyhow::Error) -> Self {
        PruningError::Internal(err.to_string())
    }
}

/// Pruning manager
pub struct PruningManager {
    state: Arc<RwLock<PruningState>>,
    db: Arc<DB>,
    archive_db: Arc<DB>,
}

impl PruningManager {
    /// Create a new pruning manager
    pub fn new(config: PruningConfig) -> Result<Self, PruningError> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);

        let db = DB::open(&db_opts, "state_db")
            .map_err(|e| PruningError::Database(e))?;

        let mut archive_opts = Options::default();
        archive_opts.create_if_missing(true);

        let archive_db = DB::open(&archive_opts, &config.archive_path)
            .map_err(|e| PruningError::Database(e))?;

        Ok(Self {
            state: Arc::new(RwLock::new(PruningState {
                current_height: 0,
                last_pruning_height: 0,
                last_archive_height: 0,
                transitions: HashMap::new(),
                config,
                metrics: PruningMetrics {
                    pruned_states: 0,
                    archived_states: 0,
                    pruned_size: 0,
                    archived_size: 0,
                    last_pruning_duration: 0,
                    last_archive_duration: 0,
                },
            })),
            db: Arc::new(db),
            archive_db: Arc::new(archive_db),
        })
    }

    /// Record a state transition
    pub async fn record_transition(
        &self,
        height: u64,
        prev_state_root: Hash,
        new_state_root: Hash,
        proof: Vec<u8>,
    ) -> Result<(), PruningError> {
        let mut state = self.state.write().await;
        
        let transition = StateTransition {
            height,
            prev_state_root,
            new_state_root,
            proof,
            timestamp: Utc::now(),
        };

        // Store transition in memory
        state.transitions.insert(height, transition.clone());

        // Store transition in database
        self.db.put(
            format!("transition:{}", height).as_bytes(),
            bincode::serialize(&transition)?,
        )?;

        debug!("Recorded state transition at height {}", height);
        Ok(())
    }

    /// Verify a state transition
    pub async fn verify_transition(&self, height: u64) -> Result<bool, PruningError> {
        let state = self.state.read().await;
        
        // Validate height
        if height == 0 || height > state.current_height {
            return Err(PruningError::InvalidHeight);
        }
        
        let transition = state.transitions.get(&height)
            .ok_or_else(|| PruningError::Internal(format!("No transition found at height {}", height)))?;
        
        // Get previous state root
        let prev_root: Hash = bincode::deserialize(
            &self.db.get(format!("state:{}", height - 1).as_bytes())?
                .ok_or_else(|| PruningError::Internal(format!("Missing state root at height {}", height - 1)))?
        )?;
        
        // Get new state root
        let new_root: Hash = bincode::deserialize(
            &self.db.get(format!("state:{}", height).as_bytes())?
                .ok_or_else(|| PruningError::Internal(format!("Missing state root at height {}", height)))?
        )?;

        // Compare state roots
        if prev_root != transition.prev_state_root {
            return Err(PruningError::InvalidStateRoot);
        }

        if new_root != transition.new_state_root {
            return Err(PruningError::InvalidStateRoot);
        }

        // Verify transition proof
        self.verify_merkle_proof(
            &transition.prev_state_root,
            &transition.new_state_root,
            &transition.proof
        ).await
    }

    /// Verify a merkle proof for state transition
    async fn verify_merkle_proof(
        &self,
        prev_root: &Hash,
        new_root: &Hash,
        proof: &[u8]
    ) -> Result<bool, PruningError> {
        verify_proof(prev_root, new_root, proof)
            .map_err(|_e| PruningError::InvalidProof)
    }

    /// Prune old states
    pub async fn prune_old_states(&mut self) -> Result<(), PruningError> {
        let current_height = self.state.read().await.current_height;
        if current_height < self.state.read().await.config.min_blocks {
            return Ok(());
        }

        let prune_height = current_height - self.state.read().await.config.min_blocks;
        
        // Archive states if needed
        if self.should_archive(prune_height) {
            self.archive_states(prune_height).await?;
        }

        // Remove states older than prune_height
        self.remove_old_states(prune_height).await?;
        
        Ok(())
    }

    async fn archive_states(&self, height: u64) -> Result<(), PruningError> {
        let archive_path = self.state.read().await.config.archive_path.join(format!("state_{}", height));
        fs::create_dir_all(&archive_path).await?;
        
        // Archive logic here
        Ok(())
    }

    async fn remove_old_states(&mut self, height: u64) -> Result<(), PruningError> {
        // Remove states older than height
        self.state.write().await.transitions.retain(|&key, _value| key > height);
        Ok(())
    }

    /// Update current block height
    pub async fn update_height(&mut self, height: u64) -> Result<(), PruningError> {
        let mut state = self.state.write().await;
        
        // Validate new height
        if height < state.current_height {
            return Err(PruningError::InvalidHeight);
        }
        
        state.current_height = height;
        
        // Drop the state lock before calling other methods
        drop(state);
        
        // Perform pruning and archiving
        self.prune_old_states().await?;
        
        Ok(())
    }

    /// Get state transition for a block
    pub async fn get_transition(&self, height: u64) -> Option<StateTransition> {
        let state = self.state.read().await;
        state.transitions.get(&height).cloned()
    }

    /// Get archived state transition
    pub async fn get_archived_transition(&self, height: u64) -> Result<Option<StateTransition>, PruningError> {
        let state = self.state.read().await;
        
        // Validate height is within archived range
        if height >= state.current_height.saturating_sub(state.config.max_blocks) {
            return Ok(None);
        }
        
        match self.archive_db.get(format!("transition:{}", height).as_bytes())? {
            Some(data) => {
                match bincode::deserialize(&data) {
                    Ok(transition) => Ok(Some(transition)),
                    Err(_e) => {
                        // Log corrupted data but don't fail
                        debug!("Failed to deserialize archived transition at height {}: {}", height, _e);
                        Ok(None)
                    }
                }
            }
            None => Ok(None)
        }
    }

    /// Check if archiving is needed
    fn should_archive(&self, height: u64) -> bool {
        let config = self.state.try_read().unwrap();
        height % config.config.archive_interval == 0
    }
} 