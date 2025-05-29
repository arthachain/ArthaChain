//! Network Monitoring API Handlers
//!
//! This module provides API endpoints for monitoring network health,
//! peer connections, mempool status, and node uptime.

use anyhow::Result;
use axum::{extract::Extension, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use crate::api::ApiError;
use crate::ledger::state::State;
use crate::network::p2p::P2PNetwork;
use crate::transaction::mempool::{EnhancedMempool, MempoolStats};

/// Node startup time for uptime calculation
static mut NODE_START_TIME: Option<SystemTime> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// Initialize node start time
pub fn init_node_start_time() {
    unsafe {
        INIT.call_once(|| {
            NODE_START_TIME = Some(SystemTime::now());
        });
    }
}

/// Get node start time
pub fn get_node_start_time() -> Option<SystemTime> {
    unsafe { NODE_START_TIME }
}

/// Response for peer count endpoint
#[derive(Serialize, Deserialize)]
pub struct PeerCountResponse {
    /// Number of connected peers
    pub peer_count: usize,
    /// Maximum number of peers allowed
    pub max_peers: usize,
    /// Minimum peers needed for healthy operation
    pub min_peers: usize,
    /// Network health status
    pub network_health: NetworkHealthStatus,
}

/// Network health status
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NetworkHealthStatus {
    /// Healthy: sufficient peers connected
    Healthy,
    /// Warning: low peer count but functional
    Warning,
    /// Critical: very low peer count, may affect functionality
    Critical,
    /// Offline: no peers connected
    Offline,
}

/// Response for mempool size tracking
#[derive(Serialize, Deserialize)]
pub struct MempoolSizeResponse {
    /// Current number of transactions in mempool
    pub transaction_count: usize,
    /// Current size in bytes
    pub size_bytes: usize,
    /// Maximum size allowed in bytes
    pub max_size_bytes: usize,
    /// Utilization percentage
    pub utilization_percent: f64,
    /// Detailed mempool statistics
    pub stats: MempoolStats,
    /// Health status based on capacity
    pub health_status: MempoolHealthStatus,
}

/// Mempool health status
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MempoolHealthStatus {
    /// Normal: low to moderate usage
    Normal,
    /// Busy: high usage but manageable
    Busy,
    /// Congested: very high usage, may cause delays
    Congested,
    /// Full: at capacity, rejecting new transactions
    Full,
}

/// Response for uptime calculation
#[derive(Serialize, Deserialize)]
pub struct UptimeResponse {
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Uptime in human-readable format
    pub uptime_formatted: String,
    /// Node start timestamp (Unix timestamp)
    pub start_timestamp: u64,
    /// Current timestamp (Unix timestamp)
    pub current_timestamp: u64,
}

/// Detailed peer information
#[derive(Serialize, Deserialize, Clone)]
pub struct DetailedPeerInfo {
    /// Peer ID
    pub peer_id: String,
    /// Peer addresses
    pub addresses: Vec<String>,
    /// Connection status
    pub status: PeerConnectionStatus,
    /// Connected since (Unix timestamp)
    pub connected_since: u64,
    /// Last seen (Unix timestamp)
    pub last_seen: u64,
    /// Peer version/agent
    pub version: String,
    /// Current block height of the peer
    pub height: u64,
    /// Network latency in milliseconds
    pub latency_ms: Option<u32>,
    /// Bytes sent to this peer
    pub bytes_sent: u64,
    /// Bytes received from this peer
    pub bytes_received: u64,
    /// Connection direction
    pub direction: ConnectionDirection,
    /// Peer reputation score
    pub reputation_score: f64,
    /// Number of failed connection attempts
    pub failed_connections: u32,
    /// Last error message (if any)
    pub last_error: Option<String>,
}

/// Peer connection status
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum PeerConnectionStatus {
    /// Successfully connected and active
    Connected,
    /// Connection in progress
    Connecting,
    /// Disconnected
    Disconnected,
    /// Connection failed
    Failed,
    /// Peer is banned
    Banned,
}

/// Connection direction
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ConnectionDirection {
    /// Outbound connection (we initiated)
    Outbound,
    /// Inbound connection (peer initiated)
    Inbound,
}

/// Response for peer list endpoint
#[derive(Serialize, Deserialize)]
pub struct PeerListResponse {
    /// List of connected peers
    pub peers: Vec<DetailedPeerInfo>,
    /// Total number of peers
    pub total_peers: usize,
    /// Number of connected peers
    pub connected_peers: usize,
    /// Number of disconnected peers
    pub disconnected_peers: usize,
    /// Average latency across all peers
    pub avg_latency_ms: Option<f64>,
    /// Total bytes sent to all peers
    pub total_bytes_sent: u64,
    /// Total bytes received from all peers
    pub total_bytes_received: u64,
}

/// Comprehensive network status response
#[derive(Serialize, Deserialize)]
pub struct NetworkStatusResponse {
    /// Peer information
    pub peer_info: PeerCountResponse,
    /// Mempool information
    pub mempool_info: MempoolSizeResponse,
    /// Uptime information
    pub uptime_info: UptimeResponse,
    /// Overall network health
    pub overall_health: OverallNetworkHealth,
}

/// Overall network health assessment
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum OverallNetworkHealth {
    /// All systems healthy
    Excellent,
    /// Minor issues but functional
    Good,
    /// Some concerns, monitoring needed
    Fair,
    /// Significant issues affecting performance
    Poor,
    /// Critical issues, immediate attention needed
    Critical,
}

/// Network monitoring service
pub struct NetworkMonitoringService {
    /// P2P network reference
    p2p_network: Option<Arc<P2PNetwork>>,
    /// Mempool reference
    mempool: Option<Arc<EnhancedMempool>>,
    /// State reference
    #[allow(dead_code)]
    state: Arc<RwLock<State>>,
}

impl NetworkMonitoringService {
    pub fn new(state: Arc<RwLock<State>>) -> Self {
        Self {
            p2p_network: None,
            mempool: None,
            state,
        }
    }

    pub fn with_p2p_network(mut self, p2p_network: Arc<P2PNetwork>) -> Self {
        self.p2p_network = Some(p2p_network);
        self
    }

    pub fn with_mempool(mut self, mempool: Arc<EnhancedMempool>) -> Self {
        self.mempool = Some(mempool);
        self
    }

    /// Get peer count information
    pub async fn get_peer_count(&self) -> Result<PeerCountResponse, ApiError> {
        let peer_count = if let Some(p2p) = &self.p2p_network {
            let stats = p2p.get_stats().await;
            stats.peer_count
        } else {
            0
        };

        let max_peers = 50; // TODO: Get from config
        let min_peers = 3; // TODO: Get from config

        let network_health = self.assess_network_health(peer_count, min_peers);

        Ok(PeerCountResponse {
            peer_count,
            max_peers,
            min_peers,
            network_health,
        })
    }

    /// Get mempool size information
    pub async fn get_mempool_size(&self) -> Result<MempoolSizeResponse, ApiError> {
        if let Some(mempool) = &self.mempool {
            let stats = mempool.get_stats().await;
            let transaction_count = mempool.transaction_count().await;
            let size_bytes = mempool.size_bytes().await;

            let utilization_percent = if stats.max_size_bytes > 0 {
                (size_bytes as f64 / stats.max_size_bytes as f64) * 100.0
            } else {
                0.0
            };

            let health_status = self.assess_mempool_health(utilization_percent, transaction_count);

            Ok(MempoolSizeResponse {
                transaction_count,
                size_bytes,
                max_size_bytes: stats.max_size_bytes,
                utilization_percent,
                stats,
                health_status,
            })
        } else {
            Err(ApiError {
                status: 503,
                message: "Mempool service not available".to_string(),
            })
        }
    }

    /// Get uptime information
    pub async fn get_uptime(&self) -> Result<UptimeResponse, ApiError> {
        let current_time = SystemTime::now();
        let current_timestamp = current_time
            .duration_since(UNIX_EPOCH)
            .map_err(|e| ApiError {
                status: 500,
                message: format!("Failed to get current time: {e}"),
            })?
            .as_secs();

        if let Some(start_time) = get_node_start_time() {
            let start_timestamp = start_time
                .duration_since(UNIX_EPOCH)
                .map_err(|e| ApiError {
                    status: 500,
                    message: format!("Failed to get start time: {e}"),
                })?
                .as_secs();

            let uptime_seconds = current_timestamp - start_timestamp;
            let uptime_formatted = format_duration(uptime_seconds);

            Ok(UptimeResponse {
                uptime_seconds,
                uptime_formatted,
                start_timestamp,
                current_timestamp,
            })
        } else {
            Err(ApiError {
                status: 503,
                message: "Node start time not initialized".to_string(),
            })
        }
    }

    /// Get detailed peer list
    pub async fn get_peer_list(&self) -> Result<PeerListResponse, ApiError> {
        if let Some(p2p) = &self.p2p_network {
            let stats = p2p.get_stats().await;

            // Mock detailed peer information (in a real implementation, this would
            // come from the actual P2P network layer)
            let peers = self.get_mock_peer_details(stats.peer_count).await;

            let connected_peers = peers
                .iter()
                .filter(|p| matches!(p.status, PeerConnectionStatus::Connected))
                .count();

            let disconnected_peers = peers.len() - connected_peers;

            let avg_latency_ms = if !peers.is_empty() {
                let total_latency: u32 = peers.iter().filter_map(|p| p.latency_ms).sum();
                if total_latency > 0 {
                    Some(total_latency as f64 / peers.len() as f64)
                } else {
                    None
                }
            } else {
                None
            };

            let total_bytes_sent: u64 = peers.iter().map(|p| p.bytes_sent).sum();
            let total_bytes_received: u64 = peers.iter().map(|p| p.bytes_received).sum();

            Ok(PeerListResponse {
                peers,
                total_peers: stats.peer_count,
                connected_peers,
                disconnected_peers,
                avg_latency_ms,
                total_bytes_sent,
                total_bytes_received,
            })
        } else {
            Ok(PeerListResponse {
                peers: Vec::new(),
                total_peers: 0,
                connected_peers: 0,
                disconnected_peers: 0,
                avg_latency_ms: None,
                total_bytes_sent: 0,
                total_bytes_received: 0,
            })
        }
    }

    /// Get comprehensive network status
    pub async fn get_network_status(&self) -> Result<NetworkStatusResponse, ApiError> {
        let peer_info = self.get_peer_count().await?;
        let mempool_info = self.get_mempool_size().await?;
        let uptime_info = self.get_uptime().await?;

        let overall_health = self.assess_overall_health(&peer_info, &mempool_info);

        Ok(NetworkStatusResponse {
            peer_info,
            mempool_info,
            uptime_info,
            overall_health,
        })
    }

    /// Assess network health based on peer count
    fn assess_network_health(&self, peer_count: usize, min_peers: usize) -> NetworkHealthStatus {
        match peer_count {
            0 => NetworkHealthStatus::Offline,
            n if n < min_peers => NetworkHealthStatus::Critical,
            n if n < min_peers * 2 => NetworkHealthStatus::Warning,
            _ => NetworkHealthStatus::Healthy,
        }
    }

    /// Assess mempool health based on utilization
    fn assess_mempool_health(
        &self,
        utilization_percent: f64,
        _transaction_count: usize,
    ) -> MempoolHealthStatus {
        match utilization_percent {
            p if p >= 95.0 => MempoolHealthStatus::Full,
            p if p >= 80.0 => MempoolHealthStatus::Congested,
            p if p >= 60.0 => MempoolHealthStatus::Busy,
            _ => MempoolHealthStatus::Normal,
        }
    }

    /// Assess overall network health
    fn assess_overall_health(
        &self,
        peer_info: &PeerCountResponse,
        mempool_info: &MempoolSizeResponse,
    ) -> OverallNetworkHealth {
        let peer_score = match peer_info.network_health {
            NetworkHealthStatus::Healthy => 4,
            NetworkHealthStatus::Warning => 3,
            NetworkHealthStatus::Critical => 2,
            NetworkHealthStatus::Offline => 1,
        };

        let mempool_score = match mempool_info.health_status {
            MempoolHealthStatus::Normal => 4,
            MempoolHealthStatus::Busy => 3,
            MempoolHealthStatus::Congested => 2,
            MempoolHealthStatus::Full => 1,
        };

        let average_score = (peer_score + mempool_score) as f64 / 2.0;

        match average_score {
            s if s >= 3.5 => OverallNetworkHealth::Excellent,
            s if s >= 3.0 => OverallNetworkHealth::Good,
            s if s >= 2.5 => OverallNetworkHealth::Fair,
            s if s >= 2.0 => OverallNetworkHealth::Poor,
            _ => OverallNetworkHealth::Critical,
        }
    }

    /// Get mock peer details (replace with actual implementation)
    async fn get_mock_peer_details(&self, peer_count: usize) -> Vec<DetailedPeerInfo> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut peers = Vec::new();
        let base_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for i in 0..peer_count {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let hash = hasher.finish();

            let peer_id = format!("peer_{hash:016x}");
            let ip = format!("192.168.1.{}", (hash % 254) + 1);
            let port = 30303 + (hash % 1000);

            peers.push(DetailedPeerInfo {
                peer_id: peer_id.clone(),
                addresses: vec![format!("{ip}:{port}")],
                status: if i % 10 == 0 {
                    PeerConnectionStatus::Disconnected
                } else {
                    PeerConnectionStatus::Connected
                },
                connected_since: base_time - ((hash % 3600) * 60), // Random time within last hour
                last_seen: base_time - (hash % 60),                // Random time within last minute
                version: format!("blockchain-node/1.0.{}", hash % 100),
                height: 1000 + (hash % 500),
                latency_ms: if i % 20 == 0 {
                    None
                } else {
                    Some((hash % 200 + 10) as u32)
                },
                bytes_sent: hash * 1024,
                bytes_received: hash * 2048,
                direction: if i % 2 == 0 {
                    ConnectionDirection::Outbound
                } else {
                    ConnectionDirection::Inbound
                },
                reputation_score: 0.5 + ((hash % 100) as f64 / 200.0),
                failed_connections: (hash % 5) as u32,
                last_error: if i % 15 == 0 {
                    Some("Connection timeout".to_string())
                } else {
                    None
                },
            });
        }

        peers
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

/// Handler for peer count endpoint
pub async fn get_peer_count(
    Extension(service): Extension<Arc<NetworkMonitoringService>>,
) -> impl IntoResponse {
    match service.get_peer_count().await {
        Ok(response) => Json(response).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.message).into_response(),
    }
}

/// Handler for mempool size endpoint
pub async fn get_mempool_size(
    Extension(service): Extension<Arc<NetworkMonitoringService>>,
) -> impl IntoResponse {
    match service.get_mempool_size().await {
        Ok(response) => Json(response).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.message).into_response(),
    }
}

/// Handler for uptime endpoint
pub async fn get_uptime(
    Extension(service): Extension<Arc<NetworkMonitoringService>>,
) -> impl IntoResponse {
    match service.get_uptime().await {
        Ok(response) => Json(response).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.message).into_response(),
    }
}

/// Handler for detailed peer list endpoint
pub async fn get_peers(
    Extension(service): Extension<Arc<NetworkMonitoringService>>,
) -> impl IntoResponse {
    match service.get_peer_list().await {
        Ok(response) => Json(response).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.message).into_response(),
    }
}

/// Handler for comprehensive network status endpoint
pub async fn get_network_status(
    Extension(service): Extension<Arc<NetworkMonitoringService>>,
) -> impl IntoResponse {
    match service.get_network_status().await {
        Ok(response) => Json(response).into_response(),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.message).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::transaction::mempool::{EnhancedMempool, MempoolConfig};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::RwLock;

    /// Create test state
    fn create_test_state() -> Arc<RwLock<State>> {
        let config = Config::default();
        Arc::new(RwLock::new(State::new(&config).unwrap()))
    }

    /// Create test mempool
    fn create_test_mempool() -> Arc<EnhancedMempool> {
        let config = MempoolConfig {
            max_size_bytes: 1024 * 1024, // 1MB
            max_transactions: 1000,
            default_ttl: Duration::from_secs(3600),
            min_gas_price: 1,
            use_quantum_resistant: false,
            cleanup_interval: Duration::from_secs(60),
            max_txs_per_account: 10,
        };
        Arc::new(EnhancedMempool::new(config))
    }

    #[tokio::test]
    async fn test_uptime_tracking() {
        // Initialize node start time
        init_node_start_time();

        // Wait a small amount of time
        tokio::time::sleep(Duration::from_millis(100)).await;

        let state = create_test_state();
        let service = NetworkMonitoringService::new(state);

        let uptime_response = service.get_uptime().await.unwrap();

        // Verify uptime is greater than 0
        assert!(uptime_response.uptime_seconds >= 0);
        assert!(!uptime_response.uptime_formatted.is_empty());
        assert!(uptime_response.current_timestamp > uptime_response.start_timestamp);

        // Test format_duration function
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(3661), "1h 1m 1s");
        assert_eq!(format_duration(90061), "1d 1h 1m 1s");
    }

    #[tokio::test]
    async fn test_peer_count_tracking() {
        let state = create_test_state();
        let service = NetworkMonitoringService::new(state);

        // Test without P2P network
        let peer_response = service.get_peer_count().await.unwrap();
        assert_eq!(peer_response.peer_count, 0);
        assert!(matches!(
            peer_response.network_health,
            NetworkHealthStatus::Offline
        ));

        // Test health assessment
        assert!(matches!(
            service.assess_network_health(0, 3),
            NetworkHealthStatus::Offline
        ));
        assert!(matches!(
            service.assess_network_health(2, 3),
            NetworkHealthStatus::Critical
        ));
        assert!(matches!(
            service.assess_network_health(4, 3),
            NetworkHealthStatus::Warning
        ));
        assert!(matches!(
            service.assess_network_health(10, 3),
            NetworkHealthStatus::Healthy
        ));
    }

    #[tokio::test]
    async fn test_mempool_size_tracking() {
        let state = create_test_state();
        let mempool = create_test_mempool();

        let service = NetworkMonitoringService::new(state).with_mempool(mempool.clone());

        let mempool_response = service.get_mempool_size().await.unwrap();

        // Verify mempool tracking (empty mempool)
        assert_eq!(mempool_response.transaction_count, 0);
        assert_eq!(mempool_response.size_bytes, 0);
        assert!(mempool_response.utilization_percent >= 0.0);
        assert!(mempool_response.utilization_percent <= 100.0);
        assert!(matches!(
            mempool_response.health_status,
            MempoolHealthStatus::Normal
        ));
    }

    #[tokio::test]
    async fn test_comprehensive_network_status() {
        init_node_start_time();

        let state = create_test_state();
        let mempool = create_test_mempool();

        let service = NetworkMonitoringService::new(state).with_mempool(mempool);

        let network_status = service.get_network_status().await.unwrap();

        // Verify all components are present
        assert_eq!(network_status.peer_info.peer_count, 0);
        assert_eq!(network_status.mempool_info.transaction_count, 0);
        assert!(network_status.uptime_info.uptime_seconds >= 0);
    }

    #[tokio::test]
    async fn test_health_status_transitions() {
        let state = create_test_state();
        let service = NetworkMonitoringService::new(state);

        // Test network health transitions
        assert!(matches!(
            service.assess_network_health(0, 5),
            NetworkHealthStatus::Offline
        ));
        assert!(matches!(
            service.assess_network_health(3, 5),
            NetworkHealthStatus::Critical
        ));
        assert!(matches!(
            service.assess_network_health(7, 5),
            NetworkHealthStatus::Warning
        ));
        assert!(matches!(
            service.assess_network_health(15, 5),
            NetworkHealthStatus::Healthy
        ));

        // Test mempool health transitions
        assert!(matches!(
            service.assess_mempool_health(30.0, 100),
            MempoolHealthStatus::Normal
        ));
        assert!(matches!(
            service.assess_mempool_health(65.0, 500),
            MempoolHealthStatus::Busy
        ));
        assert!(matches!(
            service.assess_mempool_health(85.0, 1000),
            MempoolHealthStatus::Congested
        ));
        assert!(matches!(
            service.assess_mempool_health(97.0, 5000),
            MempoolHealthStatus::Full
        ));
    }
}
