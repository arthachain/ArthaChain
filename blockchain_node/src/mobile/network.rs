//! Mobile Network Management

use anyhow::Result;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Mobile network manager
pub struct MobileNetworkManager {
    /// Maximum bandwidth in KB/s
    max_bandwidth: u32,
    /// Current bandwidth usage
    current_usage: Arc<AtomicU64>,
    /// Network statistics
    stats: Arc<RwLock<NetworkStats>>,
    /// Connection state
    connection_state: ConnectionState,
}

/// Network statistics
#[derive(Debug, Clone, Serialize)]
pub struct NetworkStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub requests_made: u64,
    pub failed_requests: u64,
    pub average_latency_ms: f32,
    pub connection_quality: ConnectionQuality,
}

/// Network status information
#[derive(Debug, Serialize)]
pub struct NetworkStatus {
    pub is_connected: bool,
    pub connection_type: ConnectionType,
    pub bandwidth_usage: f32, // Percentage of max bandwidth
    pub stats: NetworkStats,
    pub data_saver_mode: bool,
}

/// Connection quality levels
#[derive(Debug, Clone, Serialize)]
pub enum ConnectionQuality {
    Excellent, // <50ms latency
    Good,      // 50-150ms latency
    Fair,      // 150-300ms latency
    Poor,      // >300ms latency
}

/// Connection types
#[derive(Debug, Serialize)]
pub enum ConnectionType {
    WiFi,
    Cellular4G,
    Cellular5G,
    Cellular3G,
    Unknown,
}

/// Connection state
#[derive(Debug)]
enum ConnectionState {
    Connected,
    Connecting,
    Disconnected,
}

impl Default for NetworkStats {
    fn default() -> Self {
        Self {
            bytes_sent: 0,
            bytes_received: 0,
            requests_made: 0,
            failed_requests: 0,
            average_latency_ms: 0.0,
            connection_quality: ConnectionQuality::Good,
        }
    }
}

impl MobileNetworkManager {
    /// Create a new mobile network manager
    pub fn new(max_bandwidth: u32) -> Result<Self> {
        Ok(Self {
            max_bandwidth,
            current_usage: Arc::new(AtomicU64::new(0)),
            stats: Arc::new(RwLock::new(NetworkStats::default())),
            connection_state: ConnectionState::Disconnected,
        })
    }

    /// Start the network manager
    pub async fn start(&mut self) -> Result<()> {
        self.connection_state = ConnectionState::Connected;

        // Start bandwidth monitoring
        let usage_counter = Arc::clone(&self.current_usage);
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut last_reset = Instant::now();

            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;

                // Reset bandwidth counter every second
                if last_reset.elapsed() >= Duration::from_secs(1) {
                    usage_counter.store(0, Ordering::Relaxed);
                    last_reset = Instant::now();
                }

                // Update connection quality based on simulated latency
                let mut stats_guard = stats.write().await;
                stats_guard.connection_quality =
                    Self::determine_quality(stats_guard.average_latency_ms);
            }
        });

        Ok(())
    }

    /// Check if bandwidth allows for operation
    pub async fn can_use_bandwidth(&self, required_kb: u32) -> bool {
        let current_usage_kb = self.current_usage.load(Ordering::Relaxed) / 1024;
        current_usage_kb + required_kb as u64 <= self.max_bandwidth as u64
    }

    /// Record bandwidth usage
    pub async fn record_usage(&self, bytes: u64) -> Result<()> {
        self.current_usage.fetch_add(bytes, Ordering::Relaxed);

        let mut stats = self.stats.write().await;
        stats.bytes_received += bytes;
        stats.requests_made += 1;

        Ok(())
    }

    /// Record a failed request
    pub async fn record_failure(&self) -> Result<()> {
        let mut stats = self.stats.write().await;
        stats.failed_requests += 1;
        Ok(())
    }

    /// Update latency measurement
    pub async fn record_latency(&self, latency_ms: f32) -> Result<()> {
        let mut stats = self.stats.write().await;

        // Simple moving average
        if stats.average_latency_ms == 0.0 {
            stats.average_latency_ms = latency_ms;
        } else {
            stats.average_latency_ms = (stats.average_latency_ms * 0.9) + (latency_ms * 0.1);
        }

        stats.connection_quality = Self::determine_quality(stats.average_latency_ms);
        Ok(())
    }

    /// Determine connection quality from latency
    fn determine_quality(latency_ms: f32) -> ConnectionQuality {
        match latency_ms {
            x if x < 50.0 => ConnectionQuality::Excellent,
            x if x < 150.0 => ConnectionQuality::Good,
            x if x < 300.0 => ConnectionQuality::Fair,
            _ => ConnectionQuality::Poor,
        }
    }

    /// Get current network status
    pub async fn get_network_status(&self) -> Result<NetworkStatus> {
        let stats = self.stats.read().await.clone();
        let current_usage_kb = self.current_usage.load(Ordering::Relaxed) / 1024;
        let bandwidth_usage = (current_usage_kb as f32 / self.max_bandwidth as f32) * 100.0;

        Ok(NetworkStatus {
            is_connected: matches!(self.connection_state, ConnectionState::Connected),
            connection_type: self.detect_connection_type(),
            bandwidth_usage,
            stats,
            data_saver_mode: self.max_bandwidth < 50, // <50KB/s considered data saver
        })
    }

    /// Detect connection type (simplified simulation)
    fn detect_connection_type(&self) -> ConnectionType {
        // In a real implementation, this would detect actual connection type
        match self.max_bandwidth {
            x if x > 1000 => ConnectionType::WiFi,
            x if x > 500 => ConnectionType::Cellular5G,
            x if x > 100 => ConnectionType::Cellular4G,
            x if x > 20 => ConnectionType::Cellular3G,
            _ => ConnectionType::Unknown,
        }
    }

    /// Set bandwidth limit
    pub fn set_bandwidth_limit(&mut self, limit_kb: u32) -> Result<()> {
        self.max_bandwidth = limit_kb;
        Ok(())
    }

    /// Enable data saver mode
    pub fn enable_data_saver(&mut self) {
        self.max_bandwidth = self.max_bandwidth.min(25); // Limit to 25KB/s
    }

    /// Disable data saver mode
    pub fn disable_data_saver(&mut self) {
        self.max_bandwidth = 100; // Reset to default
    }

    /// Get compression recommendation based on connection
    pub async fn get_compression_level(&self) -> u8 {
        let status = self
            .get_network_status()
            .await
            .unwrap_or_else(|_| NetworkStatus {
                is_connected: false,
                connection_type: ConnectionType::Unknown,
                bandwidth_usage: 100.0,
                stats: NetworkStats::default(),
                data_saver_mode: true,
            });

        match status.connection_type {
            ConnectionType::WiFi => 3,       // Light compression
            ConnectionType::Cellular5G => 5, // Medium compression
            ConnectionType::Cellular4G => 7, // High compression
            ConnectionType::Cellular3G => 9, // Maximum compression
            ConnectionType::Unknown => 9,    // Maximum compression
        }
    }

    /// Check if should defer non-critical operations
    pub async fn should_defer_operation(&self) -> bool {
        let status = self
            .get_network_status()
            .await
            .unwrap_or_else(|_| NetworkStatus {
                is_connected: false,
                connection_type: ConnectionType::Unknown,
                bandwidth_usage: 100.0,
                stats: NetworkStats::default(),
                data_saver_mode: true,
            });

        // Defer if:
        // 1. Using >80% of bandwidth
        // 2. Poor connection quality
        // 3. High failure rate
        status.bandwidth_usage > 80.0
            || matches!(status.stats.connection_quality, ConnectionQuality::Poor)
            || (status.stats.requests_made > 0
                && (status.stats.failed_requests as f32 / status.stats.requests_made as f32) > 0.3)
    }
}
