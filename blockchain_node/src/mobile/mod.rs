//! Mobile Optimization Module
//!
//! This module provides optimizations and utilities for mobile blockchain clients,
//! including lightweight sync, battery optimization, and bandwidth management.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

pub mod battery;
pub mod network;
pub mod sync;

/// Mobile client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileConfig {
    /// Enable lightweight sync mode
    pub lightweight_sync: bool,
    /// Battery optimization level (0-3)
    pub battery_optimization: u8,
    /// Maximum bandwidth usage in KB/s
    pub max_bandwidth: u32,
    /// Background sync interval in seconds
    pub background_sync_interval: u64,
    /// Enable push notifications
    pub push_notifications: bool,
    /// Data compression level (0-9)
    pub compression_level: u8,
}

impl Default for MobileConfig {
    fn default() -> Self {
        Self {
            lightweight_sync: true,
            battery_optimization: 2,
            max_bandwidth: 100,            // 100 KB/s
            background_sync_interval: 300, // 5 minutes
            push_notifications: true,
            compression_level: 6,
        }
    }
}

/// Mobile blockchain client
pub struct MobileClient {
    config: MobileConfig,
    sync_manager: sync::LightweightSyncManager,
    battery_manager: battery::BatteryOptimizer,
    network_manager: network::MobileNetworkManager,
    last_sync: Option<Instant>,
}

impl MobileClient {
    /// Create a new mobile client
    pub fn new(config: MobileConfig) -> Result<Self> {
        let sync_manager = sync::LightweightSyncManager::new(&config)?;
        let battery_manager = battery::BatteryOptimizer::new(config.battery_optimization);
        let network_manager = network::MobileNetworkManager::new(config.max_bandwidth)?;

        Ok(Self {
            config,
            sync_manager,
            battery_manager,
            network_manager,
            last_sync: None,
        })
    }

    /// Start the mobile client
    pub async fn start(&mut self) -> Result<()> {
        // Initialize battery optimization
        self.battery_manager.start_monitoring().await?;

        // Start network management
        self.network_manager.start().await?;

        // Begin initial sync
        if self.config.lightweight_sync {
            self.sync_manager.start_lightweight_sync().await?;
        }

        self.last_sync = Some(Instant::now());
        Ok(())
    }

    /// Perform background sync
    pub async fn background_sync(&mut self) -> Result<()> {
        let now = Instant::now();

        if let Some(last_sync) = self.last_sync {
            let elapsed = now.duration_since(last_sync);
            if elapsed < Duration::from_secs(self.config.background_sync_interval) {
                return Ok(()); // Too soon for next sync
            }
        }

        // Check battery level before syncing
        if !self.battery_manager.can_perform_sync().await? {
            return Ok(()); // Skip sync to preserve battery
        }

        // Perform lightweight sync
        self.sync_manager.sync_latest_blocks(100).await?;
        self.last_sync = Some(now);

        Ok(())
    }

    /// Get client status
    pub async fn get_status(&mut self) -> Result<MobileClientStatus> {
        let sync_status = self.sync_manager.get_sync_status().await?;
        let battery_status = self.battery_manager.get_battery_status().await?;
        let network_status = self.network_manager.get_network_status().await?;

        Ok(MobileClientStatus {
            sync_status,
            battery_status,
            network_status,
            last_sync: self.last_sync,
        })
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MobileConfig) -> Result<()> {
        self.config = config;
        self.battery_manager
            .set_optimization_level(self.config.battery_optimization);
        self.network_manager
            .set_bandwidth_limit(self.config.max_bandwidth)?;
        Ok(())
    }
}

/// Mobile client status
#[derive(Debug, Serialize)]
pub struct MobileClientStatus {
    pub sync_status: sync::SyncStatus,
    pub battery_status: battery::BatteryStatus,
    pub network_status: network::NetworkStatus,
    #[serde(skip)]
    pub last_sync: Option<Instant>,
}
