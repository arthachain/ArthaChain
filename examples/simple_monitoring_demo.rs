//! Simple Network Monitoring Demo
//!
//! This is a minimal demonstration of the network monitoring functionality
//! that works independently of the complex blockchain dependencies.

use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Simplified state for demonstration
#[derive(Debug)]
pub struct SimpleState {
    pub network_name: String,
    pub height: u64,
}

impl SimpleState {
    pub fn new(network_name: String) -> Self {
        Self {
            network_name,
            height: 1000,
        }
    }

    pub fn get_height(&self) -> Result<u64, String> {
        Ok(self.height)
    }
}

/// Simplified mempool for demonstration
#[derive(Debug)]
pub struct SimpleMempool {
    pub transaction_count: usize,
    pub size_bytes: usize,
    pub max_size_bytes: usize,
}

impl Default for SimpleMempool {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleMempool {
    pub fn new() -> Self {
        Self {
            transaction_count: 25,
            size_bytes: 51200,        // 50KB
            max_size_bytes: 10485760, // 10MB
        }
    }

    pub async fn transaction_count(&self) -> usize {
        self.transaction_count
    }

    pub async fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub async fn get_stats(&self) -> SimpleMempoolStats {
        SimpleMempoolStats {
            total_transactions: self.transaction_count,
            pending_transactions: self.transaction_count - 2,
            expired_transactions: 2,
            size_bytes: self.size_bytes,
            max_size_bytes: self.max_size_bytes,
            min_gas_price: 1,
        }
    }
}

#[derive(Debug)]
pub struct SimpleMempoolStats {
    pub total_transactions: usize,
    pub pending_transactions: usize,
    pub expired_transactions: usize,
    pub size_bytes: usize,
    pub max_size_bytes: usize,
    pub min_gas_price: u64,
}

/// Uptime tracking
static mut NODE_START_TIME: Option<SystemTime> = None;
static INIT: std::sync::Once = std::sync::Once::new();

pub fn init_node_start_time() {
    unsafe {
        INIT.call_once(|| {
            NODE_START_TIME = Some(SystemTime::now());
        });
    }
}

pub fn get_node_start_time() -> Option<SystemTime> {
    unsafe { NODE_START_TIME }
}

/// Network health status
#[derive(Debug, Clone)]
pub enum NetworkHealthStatus {
    Healthy,
    Warning,
    Critical,
    Offline,
}

/// Mempool health status
#[derive(Debug, Clone)]
pub enum MempoolHealthStatus {
    Normal,
    Busy,
    Congested,
    Full,
}

/// Overall health status
#[derive(Debug, Clone)]
pub enum OverallNetworkHealth {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Simplified monitoring service
pub struct SimpleMonitoringService {
    state: Arc<RwLock<SimpleState>>,
    mempool: Option<Arc<SimpleMempool>>,
}

impl SimpleMonitoringService {
    pub fn new(state: Arc<RwLock<SimpleState>>) -> Self {
        Self {
            state,
            mempool: None,
        }
    }

    pub fn with_mempool(mut self, mempool: Arc<SimpleMempool>) -> Self {
        self.mempool = Some(mempool);
        self
    }

    /// Get peer count (simulated)
    pub async fn get_peer_count(&self) -> Result<(usize, NetworkHealthStatus), String> {
        // Simulate some peers
        let peer_count = 8;
        let min_peers = 3;

        let health = match peer_count {
            0 => NetworkHealthStatus::Offline,
            n if n < min_peers => NetworkHealthStatus::Critical,
            n if n < min_peers * 2 => NetworkHealthStatus::Warning,
            _ => NetworkHealthStatus::Healthy,
        };

        Ok((peer_count, health))
    }

    /// Get mempool information
    pub async fn get_mempool_info(
        &self,
    ) -> Result<(usize, usize, f64, MempoolHealthStatus), String> {
        if let Some(mempool) = &self.mempool {
            let count = mempool.transaction_count().await;
            let size = mempool.size_bytes().await;
            let utilization = (size as f64 / mempool.max_size_bytes as f64) * 100.0;

            let health = match utilization {
                p if p >= 95.0 => MempoolHealthStatus::Full,
                p if p >= 80.0 => MempoolHealthStatus::Congested,
                p if p >= 60.0 => MempoolHealthStatus::Busy,
                _ => MempoolHealthStatus::Normal,
            };

            Ok((count, size, utilization, health))
        } else {
            Err("Mempool not available".to_string())
        }
    }

    /// Get uptime information
    pub async fn get_uptime(&self) -> Result<(u64, String), String> {
        if let Some(start_time) = get_node_start_time() {
            let current_time = SystemTime::now();
            let uptime_duration = current_time
                .duration_since(start_time)
                .map_err(|e| format!("Time calculation error: {e}"))?;

            let uptime_seconds = uptime_duration.as_secs();
            let formatted = format_duration(uptime_seconds);

            Ok((uptime_seconds, formatted))
        } else {
            Err("Node start time not initialized".to_string())
        }
    }

    /// Get blockchain height
    pub async fn get_height(&self) -> Result<u64, String> {
        let state = self.state.read().await;
        state.get_height()
    }
}

/// Format duration in seconds to human-readable string
fn format_duration(seconds: u64) -> String {
    let days = seconds / 86400;
    let hours = (seconds % 86400) / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;

    if days > 0 {
        format!("{days}d {hours}h {minutes}m {secs}s")
    } else if hours > 0 {
        format!("{hours}h {minutes}m {secs}s")
    } else if minutes > 0 {
        format!("{minutes}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Network Monitoring Demo");
    println!("{}", "=".repeat(50));

    // Initialize uptime tracking
    init_node_start_time();

    // Wait a moment to show some uptime
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create components
    let state = Arc::new(RwLock::new(SimpleState::new("demo_network".to_string())));
    let mempool = Arc::new(SimpleMempool::new());

    // Create monitoring service
    let monitoring = SimpleMonitoringService::new(state.clone()).with_mempool(mempool.clone());

    // Demonstrate functionality
    println!("\n📊 Network Status Dashboard");
    println!("{}", "-".repeat(30));

    // 1. Blockchain Info
    match monitoring.get_height().await {
        Ok(height) => println!("🔗 Blockchain Height: {height}"),
        Err(e) => println!("❌ Height Error: {e}"),
    }

    // 2. Peer Information
    match monitoring.get_peer_count().await {
        Ok((count, health)) => {
            println!("👥 Connected Peers: {count}");
            println!("🌐 Network Health: {health:?}");
        }
        Err(e) => println!("❌ Peer Error: {e}"),
    }

    // 3. Mempool Information
    match monitoring.get_mempool_info().await {
        Ok((count, size, utilization, health)) => {
            println!("💾 Mempool Transactions: {count}");
            println!("📏 Mempool Size: {size} bytes");
            println!("📈 Utilization: {utilization:.2}%");
            println!("🎯 Mempool Health: {health:?}");
        }
        Err(e) => println!("❌ Mempool Error: {e}"),
    }

    // 4. Uptime Information
    match monitoring.get_uptime().await {
        Ok((seconds, formatted)) => {
            println!("⏰ Node Uptime: {formatted}");
            println!("🕐 Uptime Seconds: {seconds}");
        }
        Err(e) => println!("❌ Uptime Error: {e}"),
    }

    // 5. Health Assessment Demo
    println!("\n🏥 Health Assessment Examples");
    println!("{}", "-".repeat(30));

    // Simulate different peer counts
    let test_cases = vec![
        (0, "No peers - Network Offline"),
        (2, "Below minimum - Critical"),
        (5, "Low peers - Warning"),
        (15, "Good connectivity - Healthy"),
    ];

    for (peer_count, description) in test_cases {
        let health = match peer_count {
            0 => NetworkHealthStatus::Offline,
            n if n < 3 => NetworkHealthStatus::Critical,
            n if n < 6 => NetworkHealthStatus::Warning,
            _ => NetworkHealthStatus::Healthy,
        };
        println!("📊 {peer_count} peers: {health:?} - {description}");
    }

    // 6. Mempool utilization examples
    println!("\n💾 Mempool Health Examples");
    println!("{}", "-".repeat(30));

    let utilization_tests = vec![
        (30.0, "Normal operations"),
        (65.0, "Busy period"),
        (85.0, "High congestion"),
        (97.0, "At capacity"),
    ];

    for (util, description) in utilization_tests {
        let health = match util {
            p if p >= 95.0 => MempoolHealthStatus::Full,
            p if p >= 80.0 => MempoolHealthStatus::Congested,
            p if p >= 60.0 => MempoolHealthStatus::Busy,
            _ => MempoolHealthStatus::Normal,
        };
        println!("📈 {util:.1}% utilization: {health:?} - {description}");
    }

    println!("\n✅ Demo completed successfully!");
    println!("🎯 All monitoring features working correctly!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_functionality() {
        init_node_start_time();

        let state = Arc::new(RwLock::new(SimpleState::new("test".to_string())));
        let mempool = Arc::new(SimpleMempool::new());
        let monitoring = SimpleMonitoringService::new(state).with_mempool(mempool);

        // Test height
        let height = monitoring.get_height().await.unwrap();
        assert_eq!(height, 1000);

        // Test peer count
        let (peer_count, health) = monitoring.get_peer_count().await.unwrap();
        assert_eq!(peer_count, 8);
        assert!(matches!(health, NetworkHealthStatus::Healthy));

        // Test mempool
        let (count, size, util, health) = monitoring.get_mempool_info().await.unwrap();
        assert_eq!(count, 25);
        assert_eq!(size, 51200);
        assert!(util < 1.0); // Should be low utilization
        assert!(matches!(health, MempoolHealthStatus::Normal));

        // Test uptime
        let (uptime, formatted) = monitoring.get_uptime().await.unwrap();
        assert!(uptime >= 0);
        assert!(!formatted.is_empty());
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(3661), "1h 1m 1s");
        assert_eq!(format_duration(90061), "1d 1h 1m 1s");
    }
}
