//! Lightweight Sync Manager for Mobile Clients

use crate::mobile::MobileConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

/// Lightweight sync manager for mobile clients
pub struct LightweightSyncManager {
    /// Configuration
    config: MobileConfig,
    /// Recent block headers cache
    header_cache: VecDeque<BlockHeader>,
    /// Current sync height
    current_height: u64,
    /// Target height
    target_height: u64,
    /// Sync start time
    sync_start: Option<Instant>,
}

/// Simplified block header for mobile clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockHeader {
    pub height: u64,
    pub hash: String,
    pub parent_hash: String,
    pub timestamp: u64,
    pub tx_count: u32,
}

/// Sync status information
#[derive(Debug, Serialize)]
pub struct SyncStatus {
    pub current_height: u64,
    pub target_height: u64,
    pub progress_percentage: f32,
    pub is_syncing: bool,
    pub sync_speed: f32, // blocks per second
    pub cached_headers: usize,
}

impl LightweightSyncManager {
    /// Create a new lightweight sync manager
    pub fn new(config: &MobileConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            header_cache: VecDeque::with_capacity(1000),
            current_height: 0,
            target_height: 0,
            sync_start: None,
        })
    }

    /// Start lightweight sync process
    pub async fn start_lightweight_sync(&mut self) -> Result<()> {
        self.sync_start = Some(Instant::now());

        // In a real implementation, this would:
        // 1. Connect to light nodes or full nodes
        // 2. Request block headers only (not full blocks)
        // 3. Verify header chain
        // 4. Download minimal transaction data for user's addresses

        // Simulate getting network height
        self.target_height = 1000; // Would come from network

        // Start header sync
        self.sync_headers().await?;

        Ok(())
    }

    /// Sync block headers
    async fn sync_headers(&mut self) -> Result<()> {
        // Simulate header sync process
        for height in self.current_height..=(self.current_height + 10).min(self.target_height) {
            let header = BlockHeader {
                height,
                hash: format!("0x{:064x}", height * 12345),
                parent_hash: format!("0x{:064x}", (height - 1) * 12345),
                timestamp: 1640995200 + height * 12, // ~12 second blocks
                tx_count: (height % 50) as u32 + 1,
            };

            self.header_cache.push_back(header);

            // Keep cache size reasonable
            if self.header_cache.len() > 1000 {
                self.header_cache.pop_front();
            }
        }

        self.current_height = (self.current_height + 10).min(self.target_height);
        Ok(())
    }

    /// Sync latest blocks (for background sync)
    pub async fn sync_latest_blocks(&mut self, max_blocks: u32) -> Result<()> {
        // Simulate syncing latest blocks
        let blocks_to_sync = max_blocks.min((self.target_height - self.current_height) as u32);

        for _ in 0..blocks_to_sync {
            if self.current_height >= self.target_height {
                break;
            }

            // Simulate downloading block header
            let header = BlockHeader {
                height: self.current_height + 1,
                hash: format!("0x{:064x}", (self.current_height + 1) * 12345),
                parent_hash: format!("0x{:064x}", self.current_height * 12345),
                timestamp: 1640995200 + (self.current_height + 1) * 12,
                tx_count: ((self.current_height + 1) % 50) as u32 + 1,
            };

            self.header_cache.push_back(header);
            self.current_height += 1;

            // Simulate network delay for mobile
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        Ok(())
    }

    /// Get current sync status
    pub async fn get_sync_status(&self) -> Result<SyncStatus> {
        let progress_percentage = if self.target_height > 0 {
            (self.current_height as f32 / self.target_height as f32) * 100.0
        } else {
            0.0
        };

        let sync_speed = if let Some(start_time) = self.sync_start {
            let elapsed = start_time.elapsed().as_secs_f32();
            if elapsed > 0.0 {
                self.current_height as f32 / elapsed
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(SyncStatus {
            current_height: self.current_height,
            target_height: self.target_height,
            progress_percentage,
            is_syncing: self.current_height < self.target_height,
            sync_speed,
            cached_headers: self.header_cache.len(),
        })
    }

    /// Get recent block headers
    pub fn get_recent_headers(&self, count: usize) -> Vec<&BlockHeader> {
        self.header_cache.iter().rev().take(count).collect()
    }

    /// Check if block exists in cache
    pub fn has_block(&self, height: u64) -> bool {
        self.header_cache
            .iter()
            .any(|header| header.height == height)
    }
}
