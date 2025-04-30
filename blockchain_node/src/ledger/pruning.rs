use std::sync::{Arc, RwLock};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{self, Write, BufReader, BufWriter};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use log::{info, debug, warn};
use tokio::sync::mpsc;
use crate::ledger::block::Block;
use crate::ledger::state::State;
use crate::types::Address;
use crate::utils::crypto::Hash;

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Minimum number of blocks to keep
    pub min_blocks: u64,
    /// Maximum number of blocks to keep
    pub max_blocks: u64,
    /// Pruning interval (in blocks)
    pub pruning_interval: u64,
    /// Archive interval (in blocks)
    pub archive_interval: u64,
    /// Archive directory
    pub archive_dir: PathBuf,
    /// Maximum archive size (in bytes)
    pub max_archive_size: u64,
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Keep state for these accounts
    pub keep_accounts: HashSet<Address>,
    /// Keep state for these contracts
    pub keep_contracts: HashSet<Address>,
}

/// Archive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveMetadata {
    /// Start block height
    pub start_height: u64,
    /// End block height
    pub end_height: u64,
    /// Archive timestamp
    pub timestamp: u64,
    /// Archive size
    pub size: u64,
    /// State root hash
    pub state_root: Vec<u8>,
    /// Block count
    pub block_count: u64,
    /// Transaction count
    pub tx_count: u64,
    /// Account count
    pub account_count: u64,
    /// Contract count
    pub contract_count: u64,
}

/// State pruning manager
pub struct PruningManager {
    /// Configuration
    config: PruningConfig,
    /// Current state
    state: Arc<RwLock<State>>,
    /// Archive metadata
    archives: Arc<RwLock<HashMap<u64, ArchiveMetadata>>>,
    /// Last pruning height
    last_pruning: u64,
    /// Last archive height
    last_archive: u64,
    /// Channel for receiving pruning requests
    pruning_rx: mpsc::Receiver<PruningRequest>,
    /// Channel for sending pruning responses
    pruning_tx: mpsc::Sender<PruningResponse>,
}

/// Pruning request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningRequest {
    /// Prune state up to a specific height
    PruneToHeight {
        height: u64,
        timestamp: u64,
    },
    /// Archive state up to a specific height
    ArchiveToHeight {
        height: u64,
        timestamp: u64,
    },
    /// Restore state from archive
    RestoreFromArchive {
        archive_height: u64,
        timestamp: u64,
    },
}

/// Pruning response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningResponse {
    /// Pruning completed
    PruningCompleted {
        height: u64,
        pruned_blocks: u64,
        pruned_size: u64,
    },
    /// Archiving completed
    ArchivingCompleted {
        height: u64,
        archive_path: PathBuf,
        archive_size: u64,
    },
    /// Restoration completed
    RestorationCompleted {
        height: u64,
        restored_blocks: u64,
        restored_size: u64,
    },
    /// Error occurred
    Error {
        message: String,
    },
}

impl PruningManager {
    /// Create a new pruning manager
    pub fn new(
        config: PruningConfig,
        state: Arc<RwLock<State>>,
        pruning_rx: mpsc::Receiver<PruningRequest>,
        pruning_tx: mpsc::Sender<PruningResponse>,
    ) -> Self {
        Self {
            config,
            state,
            archives: Arc::new(RwLock::new(HashMap::new())),
            last_pruning: 0,
            last_archive: 0,
            pruning_rx,
            pruning_tx,
        }
    }

    /// Start the pruning manager
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting pruning manager");
        
        // Create archive directory if it doesn't exist
        fs::create_dir_all(&self.config.archive_dir)?;
        
        while let Some(request) = self.pruning_rx.recv().await {
            match request {
                PruningRequest::PruneToHeight { height, timestamp } => {
                    self.handle_prune_request(height, timestamp).await?;
                },
                PruningRequest::ArchiveToHeight { height, timestamp } => {
                    self.handle_archive_request(height, timestamp).await?;
                },
                PruningRequest::RestoreFromArchive { archive_height, timestamp } => {
                    self.handle_restore_request(archive_height, timestamp).await?;
                },
            }
        }
        
        Ok(())
    }

    /// Handle a prune request
    async fn handle_prune_request(&mut self, height: u64, timestamp: u64) -> Result<()> {
        info!("Processing prune request to height {}", height);
        
        let mut state = self.state.write().await;
        let mut pruned_blocks = 0;
        let mut pruned_size = 0;
        
        // Prune blocks
        while state.height > height {
            if let Some(block) = state.remove_block() {
                pruned_blocks += 1;
                pruned_size += block.size() as u64;
            }
        }
        
        // Prune state
        state.prune_state(height, &self.config.keep_accounts, &self.config.keep_contracts)?;
        
        self.last_pruning = height;
        
        // Send response
        self.pruning_tx.send(PruningResponse::PruningCompleted {
            height,
            pruned_blocks,
            pruned_size,
        }).await?;
        
        info!("Pruning completed: {} blocks, {} bytes", pruned_blocks, pruned_size);
        
        Ok(())
    }

    /// Handle an archive request
    async fn handle_archive_request(&mut self, height: u64, timestamp: u64) -> Result<()> {
        info!("Processing archive request to height {}", height);
        
        let state = self.state.read().await;
        let archive_path = self.config.archive_dir.join(format!("archive_{}.bin", height));
        
        // Create archive file
        let file = File::create(&archive_path)?;
        let writer = BufWriter::new(file);
        
        // Write archive metadata
        let metadata = ArchiveMetadata {
            start_height: self.last_archive,
            end_height: height,
            timestamp,
            size: 0, // Will be updated after writing
            state_root: state.get_state_root()?,
            block_count: height - self.last_archive,
            tx_count: state.get_total_transactions()?,
            account_count: state.get_account_count()?,
            contract_count: state.get_contract_count()?,
        };
        
        // Write state to archive
        let mut size = 0;
        for block_height in self.last_archive..=height {
            if let Some(block) = state.get_block(block_height)? {
                let block_data = bincode::serialize(&block)?;
                size += block_data.len() as u64;
                writer.write_all(&block_data)?;
            }
        }
        
        // Update metadata with final size
        let mut metadata = metadata;
        metadata.size = size;
        
        // Write metadata to separate file
        let meta_path = archive_path.with_extension("meta");
        let meta_file = File::create(meta_path)?;
        bincode::serialize_into(meta_file, &metadata)?;
        
        // Update archives
        self.archives.write().await.insert(height, metadata.clone());
        
        self.last_archive = height;
        
        // Send response
        self.pruning_tx.send(PruningResponse::ArchivingCompleted {
            height,
            archive_path: archive_path.clone(),
            archive_size: size,
        }).await?;
        
        info!("Archiving completed: {} bytes", size);
        
        Ok(())
    }

    /// Handle a restore request
    async fn handle_restore_request(&mut self, archive_height: u64, timestamp: u64) -> Result<()> {
        info!("Processing restore request from archive height {}", archive_height);
        
        let archive_path = self.config.archive_dir.join(format!("archive_{}.bin", archive_height));
        let meta_path = archive_path.with_extension("meta");
        
        // Read metadata
        let meta_file = File::open(meta_path)?;
        let metadata: ArchiveMetadata = bincode::deserialize_from(meta_file)?;
        
        // Read archive
        let file = File::open(archive_path)?;
        let reader = BufReader::new(file);
        
        let mut state = self.state.write().await;
        let mut restored_blocks = 0;
        let mut restored_size = 0;
        
        // Restore blocks
        for block_height in metadata.start_height..=metadata.end_height {
            if let Ok(block) = bincode::deserialize_from::<_, Block>(reader) {
                state.add_block(block)?;
                restored_blocks += 1;
                restored_size += block.size() as u64;
            }
        }
        
        // Send response
        self.pruning_tx.send(PruningResponse::RestorationCompleted {
            height: archive_height,
            restored_blocks,
            restored_size,
        }).await?;
        
        info!("Restoration completed: {} blocks, {} bytes", restored_blocks, restored_size);
        
        Ok(())
    }

    /// Get archive metadata
    pub async fn get_archive_metadata(&self, height: u64) -> Option<ArchiveMetadata> {
        self.archives.read().await.get(&height).cloned()
    }

    /// List available archives
    pub async fn list_archives(&self) -> Vec<(u64, ArchiveMetadata)> {
        self.archives.read().await
            .iter()
            .map(|(height, metadata)| (*height, metadata.clone()))
            .collect()
    }

    /// Clean up old archives
    pub async fn cleanup_archives(&mut self) -> Result<()> {
        let mut archives = self.archives.write().await;
        let mut total_size = 0;
        
        // Calculate total size
        for metadata in archives.values() {
            total_size += metadata.size;
        }
        
        // Remove oldest archives if total size exceeds limit
        while total_size > self.config.max_archive_size {
            if let Some((height, metadata)) = archives.iter()
                .min_by_key(|(_, meta)| meta.timestamp)
            {
                let archive_path = self.config.archive_dir.join(format!("archive_{}.bin", height));
                let meta_path = archive_path.with_extension("meta");
                
                // Remove files
                fs::remove_file(archive_path)?;
                fs::remove_file(meta_path)?;
                
                total_size -= metadata.size;
                archives.remove(height);
                
                info!("Removed archive at height {}", height);
            } else {
                break;
            }
        }
        
        Ok(())
    }
} 