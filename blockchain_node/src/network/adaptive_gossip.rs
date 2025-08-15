use crate::network::peer::{PeerId, PeerInfo};
use crate::utils::crypto::{
    dilithium_sign, dilithium_verify, quantum_resistant_hash, PostQuantumCrypto,
};
use log::{debug, info, warn};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Configuration for adaptive gossip protocol
#[derive(Clone, Debug)]
pub struct AdaptiveGossipConfig {
    /// Minimum number of peers to maintain
    pub min_peers: usize,
    /// Maximum number of peers to connect to
    pub max_peers: usize,
    /// Optimal number of peers for the network
    pub optimal_peers: usize,
    /// Interval to check peer health and adjust gossip parameters
    pub health_check_interval: Duration,
    /// Base gossip interval
    pub base_gossip_interval: Duration,
    /// Minimum gossip interval
    pub min_gossip_interval: Duration,
    /// Maximum gossip interval
    pub max_gossip_interval: Duration,
    /// Threshold for considering a peer as high latency
    pub high_latency_threshold: Duration,
    /// Threshold for considering the network as congested
    pub congestion_threshold: f64,
    /// Whether to use quantum-resistant cryptography
    pub use_quantum_resistant: bool,
}

impl Default for AdaptiveGossipConfig {
    fn default() -> Self {
        Self {
            min_peers: 8,
            max_peers: 50,
            optimal_peers: 25,
            health_check_interval: Duration::from_secs(30),
            base_gossip_interval: Duration::from_secs(2),
            min_gossip_interval: Duration::from_millis(500),
            max_gossip_interval: Duration::from_secs(10),
            high_latency_threshold: Duration::from_millis(500),
            congestion_threshold: 0.8,
            use_quantum_resistant: true,
        }
    }
}

/// Current status of peer network
#[derive(Debug, Clone, Copy)]
pub enum NetworkStatus {
    /// Fewer peers than minimum
    Sparse,
    /// Optimal number of peers
    Healthy,
    /// More peers than optimal, but fewer than maximum
    Dense,
    /// More peers than maximum
    Congested,
}

/// Represents a message in the gossip protocol
#[derive(Debug, Clone)]
pub struct GossipMessage {
    /// Message ID (hash of content)
    pub id: Vec<u8>,
    /// Message content
    pub content: Vec<u8>,
    /// Sender ID
    pub sender: PeerId,
    /// Timestamp
    pub timestamp: u64,
    /// Time-to-live (number of hops)
    pub ttl: u8,
    /// Signature (regular or quantum-resistant)
    pub signature: Vec<u8>,
}

/// Manager for adaptive gossip protocol
pub struct AdaptiveGossipManager {
    /// Configuration
    config: AdaptiveGossipConfig,
    /// Connected peers
    peers: Arc<RwLock<HashMap<PeerId, PeerInfo>>>,
    /// Recently seen messages to avoid duplicates
    seen_messages: Arc<RwLock<HashMap<Vec<u8>, Instant>>>,
    /// Current gossip interval
    current_gossip_interval: Arc<RwLock<Duration>>,
    /// Last health check timestamp
    last_health_check: Arc<RwLock<Instant>>,
    /// Node's quantum-resistant private key
    quantum_private_key: Vec<u8>,
    /// Node's quantum-resistant public key
    quantum_public_key: Vec<u8>,
}

impl AdaptiveGossipManager {
    /// Create new adaptive gossip manager
    pub fn new(
        config: AdaptiveGossipConfig,
        quantum_private_key: Vec<u8>,
        quantum_public_key: Vec<u8>,
    ) -> Self {
        let base_interval = config.base_gossip_interval;
        Self {
            config,
            peers: Arc::new(RwLock::new(HashMap::new())),
            seen_messages: Arc::new(RwLock::new(HashMap::new())),
            current_gossip_interval: Arc::new(RwLock::new(base_interval)),
            last_health_check: Arc::new(RwLock::new(Instant::now())),
            quantum_private_key,
            quantum_public_key,
        }
    }

    /// Get current network status
    pub fn network_status(&self) -> NetworkStatus {
        let peer_count = self.peers.read().unwrap().len();

        if peer_count < self.config.min_peers {
            NetworkStatus::Sparse
        } else if peer_count <= self.config.optimal_peers {
            NetworkStatus::Healthy
        } else if peer_count <= self.config.max_peers {
            NetworkStatus::Dense
        } else {
            NetworkStatus::Congested
        }
    }

    /// Create a new gossip message with quantum-resistant signature
    pub fn create_message(
        &self,
        content: Vec<u8>,
        sender: PeerId,
        ttl: u8,
    ) -> Result<GossipMessage, anyhow::Error> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        // Generate message ID using quantum-resistant hash
        let id = if self.config.use_quantum_resistant {
            quantum_resistant_hash(&content)?
        } else {
            // Fallback to quantum-resistant hash (BLAKE3)
            blake3::hash(&content).as_bytes().to_vec()
        };

        // Generate signature
        let timestamp_bytes = timestamp.to_be_bytes().to_vec();
        let sender_bytes = sender.to_string().into_bytes();
        let signature_data = [
            &id[..],
            &content[..],
            &sender_bytes[..],
            &timestamp_bytes[..],
        ]
        .concat();
        let signature = if self.config.use_quantum_resistant {
            dilithium_sign(&self.quantum_private_key, &signature_data)?
        } else {
            // Use quantum-resistant signature for fallback too
            let pq_crypto = PostQuantumCrypto::new()?;
            pq_crypto.sign(&self.quantum_private_key, &signature_data)?
        };

        Ok(GossipMessage {
            id,
            content,
            sender,
            timestamp,
            ttl,
            signature,
        })
    }

    /// Verify a gossip message
    pub fn verify_message(
        &self,
        message: &GossipMessage,
        sender_public_key: &[u8],
    ) -> Result<bool, anyhow::Error> {
        // Verify signature
        let timestamp_bytes = message.timestamp.to_be_bytes().to_vec();
        let sender_bytes = message.sender.to_string().into_bytes();
        let signature_data = [
            &message.id[..],
            &message.content[..],
            &sender_bytes[..],
            &timestamp_bytes[..],
        ]
        .concat();

        let valid = if self.config.use_quantum_resistant {
            dilithium_verify(sender_public_key, &signature_data, &message.signature)?
        } else {
            // Use quantum-resistant verification for fallback too
            let pq_crypto = PostQuantumCrypto::new()?;
            pq_crypto.verify(sender_public_key, &signature_data, &message.signature)?
        };

        Ok(valid)
    }

    /// Process incoming gossip message
    pub fn process_message(
        &self,
        message: GossipMessage,
        sender_public_key: &[u8],
    ) -> Result<bool, anyhow::Error> {
        // Check if message already seen
        {
            let seen = self.seen_messages.read().unwrap();
            if seen.contains_key(&message.id) {
                return Ok(false); // Already processed, not an error
            }
        }

        // Verify message
        if !self.verify_message(&message, sender_public_key)? {
            return Err(anyhow::anyhow!("Invalid message signature"));
        }

        // Mark as seen
        {
            let mut seen = self.seen_messages.write().unwrap();
            seen.insert(message.id.clone(), Instant::now());
        }

        // Process message content
        // Actual message handling would be application specific

        Ok(true)
    }

    /// Perform health check and adjust gossip parameters
    pub fn check_health(&self) -> Result<(), anyhow::Error> {
        let now = Instant::now();
        let mut last_check = self.last_health_check.write().unwrap();

        // Check if it's time for a health check
        if now.duration_since(*last_check) < self.config.health_check_interval {
            return Ok(());
        }

        *last_check = now;

        // Get network statistics
        let peers = self.peers.read().unwrap();
        let peer_count = peers.len();

        // Calculate high latency peer percentage
        let high_latency_count = peers
            .values()
            .filter(|info| info.latency > self.config.high_latency_threshold)
            .count();

        let high_latency_percentage = if peer_count > 0 {
            high_latency_count as f64 / peer_count as f64
        } else {
            0.0
        };

        // Adjust gossip interval based on network status
        let mut current_interval = self.current_gossip_interval.write().unwrap();

        *current_interval = match self.network_status() {
            NetworkStatus::Sparse => {
                // More frequent gossip to find peers faster
                self.config.min_gossip_interval
            }
            NetworkStatus::Healthy => {
                // Use base interval
                self.config.base_gossip_interval
            }
            NetworkStatus::Dense => {
                // Slightly slower gossip to reduce load
                self.config.base_gossip_interval.mul_f64(1.5)
            }
            NetworkStatus::Congested => {
                // Much slower gossip to reduce load
                self.config.max_gossip_interval
            }
        };

        // Further adjust based on high latency percentage
        if high_latency_percentage > self.config.congestion_threshold {
            *current_interval =
                std::cmp::min(*current_interval * 2, self.config.max_gossip_interval);
            warn!(
                "Network experiencing high latency ({:.1}%), increasing gossip interval to {:?}",
                high_latency_percentage * 100.0,
                *current_interval
            );
        }

        // Log health status
        info!(
            "Network health check: {} peers, {:.1}% high latency, gossip interval: {:?}",
            peer_count,
            high_latency_percentage * 100.0,
            *current_interval
        );

        Ok(())
    }

    /// Clean up seen messages cache
    pub fn clean_seen_messages(&self, expiry: Duration) {
        let now = Instant::now();
        let mut seen = self.seen_messages.write().unwrap();

        seen.retain(|_, timestamp| now.duration_since(*timestamp) < expiry);

        debug!("Cleaned seen messages, {} remain", seen.len());
    }

    /// Get current gossip interval
    pub fn gossip_interval(&self) -> Duration {
        *self.current_gossip_interval.read().unwrap()
    }

    /// Add a peer to the manager
    pub fn add_peer(&self, id: PeerId, info: PeerInfo) {
        let mut peers = self.peers.write().unwrap();
        peers.insert(id, info);
    }

    /// Remove a peer from the manager
    pub fn remove_peer(&self, id: &PeerId) -> bool {
        let mut peers = self.peers.write().unwrap();
        peers.remove(id).is_some()
    }

    /// Update peer information
    pub fn update_peer(&self, id: &PeerId, info: PeerInfo) -> bool {
        let mut peers = self.peers.write().unwrap();
        if peers.contains_key(id) {
            peers.insert(id.clone(), info);
            true
        } else {
            false
        }
    }

    /// Get peer count
    pub fn peer_count(&self) -> usize {
        self.peers.read().unwrap().len()
    }

    /// Get the node's quantum-resistant public key for peer verification
    pub fn get_public_key(&self) -> &[u8] {
        &self.quantum_public_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_status() {
        let config = AdaptiveGossipConfig {
            min_peers: 5,
            optimal_peers: 15,
            max_peers: 30,
            ..Default::default()
        };

        // Generate test keys
        let (quantum_public_key, quantum_private_key) =
            match crate::utils::crypto::generate_quantum_resistant_keypair() {
                Ok(keys) => keys,
                Err(_) => (vec![0; 32], vec![0; 32]),
            };

        let manager =
            AdaptiveGossipManager::new(config.clone(), quantum_private_key, quantum_public_key);

        // Empty network should be sparse
        assert!(matches!(manager.network_status(), NetworkStatus::Sparse));

        // Add some peers
        for i in 0..3 {
            let peer_id = PeerId::from(format!("peer_{}", i));
            let peer_info = PeerInfo {
                node_id: format!("node_{}", i),
                address: format!("127.0.0.1:{}", 8000 + i),
                latency: Duration::from_millis(100),
                last_seen: Instant::now(),
                connected_since: Instant::now(),
                reputation: 0.5,
            };
            manager.add_peer(peer_id, peer_info);
        }

        // Still sparse
        assert!(matches!(manager.network_status(), NetworkStatus::Sparse));

        // Add more peers to reach healthy state
        for i in 3..10 {
            let peer_id = PeerId::from(format!("peer_{}", i));
            let peer_info = PeerInfo {
                node_id: format!("node_{}", i),
                address: format!("127.0.0.1:{}", 8000 + i),
                latency: Duration::from_millis(100),
                last_seen: Instant::now(),
                connected_since: Instant::now(),
                reputation: 0.5,
            };
            manager.add_peer(peer_id, peer_info);
        }

        // Should be healthy now
        assert!(matches!(manager.network_status(), NetworkStatus::Healthy));

        // Add more peers to reach dense state
        for i in 10..20 {
            let peer_id = PeerId::from(format!("peer_{}", i));
            let peer_info = PeerInfo {
                node_id: format!("node_{}", i),
                address: format!("127.0.0.1:{}", 8000 + i),
                latency: Duration::from_millis(100),
                last_seen: Instant::now(),
                connected_since: Instant::now(),
                reputation: 0.5,
            };
            manager.add_peer(peer_id, peer_info);
        }

        // Should be dense now
        assert!(matches!(manager.network_status(), NetworkStatus::Dense));

        // Add even more peers to reach congested state
        for i in 20..35 {
            let peer_id = PeerId::from(format!("peer_{}", i));
            let peer_info = PeerInfo {
                node_id: format!("node_{}", i),
                address: format!("127.0.0.1:{}", 8000 + i),
                latency: Duration::from_millis(100),
                last_seen: Instant::now(),
                connected_since: Instant::now(),
                reputation: 0.5,
            };
            manager.add_peer(peer_id, peer_info);
        }

        // Should be congested now
        assert!(matches!(manager.network_status(), NetworkStatus::Congested));
    }
}
