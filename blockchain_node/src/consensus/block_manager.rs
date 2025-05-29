use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::types::{BlockHash, BlockHeight, NodeId, PeerId};
use crate::consensus::metrics::NetworkMetrics;

const MIN_BLOCK_SIZE: usize = 1024 * 1024; // 1MB
const MAX_BLOCK_SIZE: usize = 8 * 1024 * 1024; // 8MB
const TARGET_BLOCK_TIME: u64 = 30; // 30 seconds

pub struct AdaptiveBlockManager {
    // Track block sizes and times
    block_metrics: Arc<RwLock<BlockMetrics>>,
    // Peer management
    peer_manager: Arc<RwLock<PeerManager>>,
    // Network metrics
    metrics: Arc<NetworkMetrics>,
}

struct BlockMetrics {
    // Recent block sizes
    recent_sizes: Vec<usize>,
    // Recent block times
    recent_times: Vec<u64>,
    // Current target size
    current_target: usize,
    // Network congestion level
    congestion_level: f64,
}

struct PeerManager {
    // Active peers
    active_peers: HashMap<PeerId, PeerInfo>,
    // Peer performance metrics
    peer_metrics: HashMap<PeerId, PeerMetrics>,
    // Banned peers
    banned_peers: HashSet<PeerId>,
}

struct PeerInfo {
    node_id: NodeId,
    connection_time: u64,
    last_seen: u64,
    capabilities: Vec<String>,
}

struct PeerMetrics {
    latency: f64,
    reliability: f64,
    bandwidth: f64,
    last_updated: u64,
}

impl AdaptiveBlockManager {
    pub fn new(metrics: Arc<NetworkMetrics>) -> Self {
        Self {
            block_metrics: Arc::new(RwLock::new(BlockMetrics::new())),
            peer_manager: Arc::new(RwLock::new(PeerManager::new())),
            metrics,
        }
    }

    pub async fn update_block_metrics(&self, size: usize, time: u64) {
        let mut metrics = self.block_metrics.write().await;
        metrics.update(size, time);
        
        // Adjust block size if needed
        let new_target = metrics.calculate_optimal_size();
        if new_target != metrics.current_target {
            metrics.current_target = new_target;
            self.metrics.record_block_size_adjustment(new_target);
        }
    }

    pub async fn get_current_block_size(&self) -> usize {
        let metrics = self.block_metrics.read().await;
        metrics.current_target
    }

    pub async fn add_peer(&self, peer_id: PeerId, info: PeerInfo) {
        let mut manager = self.peer_manager.write().await;
        manager.add_peer(peer_id, info);
        self.metrics.record_peer_added(peer_id);
    }

    pub async fn update_peer_metrics(&self, peer_id: PeerId, latency: f64, reliability: f64, bandwidth: f64) {
        let mut manager = self.peer_manager.write().await;
        manager.update_metrics(peer_id, latency, reliability, bandwidth);
    }

    pub async fn get_optimal_peers(&self, count: usize) -> Vec<PeerId> {
        let manager = self.peer_manager.read().await;
        manager.get_best_peers(count)
    }

    pub async fn handle_peer_failure(&self, peer_id: PeerId, failure_type: &str) {
        let mut manager = self.peer_manager.write().await;
        manager.handle_failure(peer_id, failure_type);
        self.metrics.record_peer_failure(peer_id, failure_type);
    }
}

impl BlockMetrics {
    fn new() -> Self {
        Self {
            recent_sizes: Vec::with_capacity(100),
            recent_times: Vec::with_capacity(100),
            current_target: MIN_BLOCK_SIZE,
            congestion_level: 0.0,
        }
    }

    fn update(&mut self, size: usize, time: u64) {
        self.recent_sizes.push(size);
        self.recent_times.push(time);
        
        // Keep only recent history
        if self.recent_sizes.len() > 100 {
            self.recent_sizes.remove(0);
            self.recent_times.remove(0);
        }
        
        // Update congestion level
        self.update_congestion_level();
    }

    fn update_congestion_level(&mut self) {
        if self.recent_times.len() < 2 {
            return;
        }
        
        // Calculate average block time
        let avg_time: f64 = self.recent_times.iter().sum::<u64>() as f64 / self.recent_times.len() as f64;
        
        // Update congestion based on target time
        self.congestion_level = (avg_time - TARGET_BLOCK_TIME as f64) / TARGET_BLOCK_TIME as f64;
        self.congestion_level = self.congestion_level.clamp(-1.0, 1.0);
    }

    fn calculate_optimal_size(&self) -> usize {
        if self.recent_sizes.is_empty() {
            return self.current_target;
        }
        
        // Calculate average size
        let avg_size: f64 = self.recent_sizes.iter().sum::<usize>() as f64 / self.recent_sizes.len() as f64;
        
        // Adjust based on congestion
        let adjustment = 1.0 - self.congestion_level * 0.2; // Max 20% adjustment
        let new_size = (avg_size * adjustment) as usize;
        
        // Ensure within bounds
        new_size.clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE)
    }
}

impl PeerManager {
    fn new() -> Self {
        Self {
            active_peers: HashMap::new(),
            peer_metrics: HashMap::new(),
            banned_peers: HashSet::new(),
        }
    }

    fn add_peer(&mut self, peer_id: PeerId, info: PeerInfo) {
        if !self.banned_peers.contains(&peer_id) {
            self.active_peers.insert(peer_id, info);
            self.peer_metrics.insert(peer_id, PeerMetrics {
                latency: 0.0,
                reliability: 1.0,
                bandwidth: 0.0,
                last_updated: 0,
            });
        }
    }

    fn update_metrics(&mut self, peer_id: PeerId, latency: f64, reliability: f64, bandwidth: f64) {
        if let Some(metrics) = self.peer_metrics.get_mut(&peer_id) {
            // Use exponential moving average for smooth updates
            const ALPHA: f64 = 0.2;
            metrics.latency = (1.0 - ALPHA) * metrics.latency + ALPHA * latency;
            metrics.reliability = (1.0 - ALPHA) * metrics.reliability + ALPHA * reliability;
            metrics.bandwidth = (1.0 - ALPHA) * metrics.bandwidth + ALPHA * bandwidth;
            metrics.last_updated = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
    }

    fn get_best_peers(&self, count: usize) -> Vec<PeerId> {
        let mut peers: Vec<_> = self.peer_metrics.iter()
            .filter(|(&peer_id, _)| self.active_peers.contains_key(&peer_id))
            .collect();
        
        // Sort by composite score
        peers.sort_by(|a, b| {
            let score_a = self.calculate_peer_score(a.1);
            let score_b = self.calculate_peer_score(b.1);
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        peers.iter()
            .take(count)
            .map(|(&peer_id, _)| peer_id)
            .collect()
    }

    fn calculate_peer_score(&self, metrics: &PeerMetrics) -> f64 {
        // Weight factors for different metrics
        const LATENCY_WEIGHT: f64 = 0.3;
        const RELIABILITY_WEIGHT: f64 = 0.4;
        const BANDWIDTH_WEIGHT: f64 = 0.3;
        
        // Normalize and combine scores
        let latency_score = 1.0 / (1.0 + metrics.latency / 1000.0); // Convert to seconds
        let reliability_score = metrics.reliability;
        let bandwidth_score = metrics.bandwidth / 1_000_000.0; // Convert to MB/s
        
        LATENCY_WEIGHT * latency_score +
        RELIABILITY_WEIGHT * reliability_score +
        BANDWIDTH_WEIGHT * bandwidth_score
    }

    fn handle_failure(&mut self, peer_id: PeerId, failure_type: &str) {
        if let Some(metrics) = self.peer_metrics.get_mut(&peer_id) {
            // Update reliability based on failure type
            let penalty = match failure_type {
                "timeout" => 0.1,
                "invalid_data" => 0.3,
                "malicious" => 1.0,
                _ => 0.2,
            };
            
            metrics.reliability -= penalty;
            
            // Ban peer if reliability drops too low
            if metrics.reliability < 0.2 {
                self.ban_peer(peer_id);
            }
        }
    }

    fn ban_peer(&mut self, peer_id: PeerId) {
        self.active_peers.remove(&peer_id);
        self.peer_metrics.remove(&peer_id);
        self.banned_peers.insert(peer_id);
    }
} 