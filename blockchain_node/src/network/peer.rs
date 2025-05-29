use std::time::{Duration, Instant};

use std::fmt;
use std::hash::{Hash, Hasher};
use tokio::sync::mpsc;

use crate::network::error::NetworkError;

/// Peer identifier used in the network
#[derive(Clone, Debug, Eq)]
pub struct PeerId {
    id: String,
}

impl PeerId {
    /// Create a new peer ID from a string
    pub fn new(id: String) -> Self {
        Self { id }
    }

    /// Get the string representation of the peer ID
    pub fn as_str(&self) -> &str {
        &self.id
    }
}

impl From<String> for PeerId {
    fn from(id: String) -> Self {
        Self { id }
    }
}

impl From<&str> for PeerId {
    fn from(id: &str) -> Self {
        Self { id: id.to_string() }
    }
}

impl fmt::Display for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl PartialEq for PeerId {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Hash for PeerId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Information about a connected peer
#[derive(Clone, Debug)]
pub struct PeerInfo {
    /// Node identifier
    pub node_id: String,
    /// Network address
    pub address: String,
    /// Measured latency to peer
    pub latency: Duration,
    /// When the peer was last seen
    pub last_seen: Instant,
    /// When the peer was first connected
    pub connected_since: Instant,
    /// Peer reputation score (0.0 to 1.0)
    pub reputation: f64,
}

impl PeerInfo {
    /// Create a new peer info entry
    pub fn new(node_id: String, address: String) -> Self {
        Self {
            node_id,
            address,
            latency: Duration::from_millis(0),
            last_seen: Instant::now(),
            connected_since: Instant::now(),
            reputation: 0.5, // Default neutral reputation
        }
    }

    /// Update the last seen timestamp
    pub fn update_last_seen(&mut self) {
        self.last_seen = Instant::now();
    }

    /// Update the latency measurement
    pub fn update_latency(&mut self, latency: Duration) {
        self.latency = latency;
    }

    /// Update the reputation score
    pub fn update_reputation(&mut self, reputation: f64) {
        self.reputation = reputation.clamp(0.0, 1.0);
    }

    /// Calculate the peer's age (how long it's been connected)
    pub fn age(&self) -> Duration {
        Instant::now().duration_since(self.connected_since)
    }

    /// Check if the peer has been inactive for the given duration
    pub fn is_inactive(&self, threshold: Duration) -> bool {
        Instant::now().duration_since(self.last_seen) > threshold
    }
}

#[derive(Debug)]
pub struct PeerManager {
    peers: Vec<PeerInfo>,
    banned_peers: Vec<(String, Instant)>,
    config: PeerManagerConfig,
    event_tx: mpsc::Sender<PeerEvent>,
}

#[derive(Debug, Clone)]
pub struct PeerManagerConfig {
    pub max_peers: usize,
    pub min_reputation: f64,
    pub ban_threshold: f64,
    pub ban_duration: Duration,
    pub warning_threshold: u32,
    pub reputation_decay: f64,
    pub reputation_boost: f64,
    pub reputation_penalty: f64,
}

#[derive(Debug, Clone)]
pub enum PeerEvent {
    Connected(PeerInfo),
    Disconnected(String),
    MessageReceived { from: String, bytes: usize },
    MessageSent { to: String, bytes: usize },
    MessageFailed { to: String },
    PingUpdated { node_id: String, ping_ms: u64 },
    ReputationUpdated { node_id: String, score: f64 },
    Banned { node_id: String, duration: Duration },
    Warning { node_id: String, reason: String },
}

impl PeerManager {
    pub fn new(config: PeerManagerConfig) -> (Self, mpsc::Receiver<PeerEvent>) {
        let (tx, rx) = mpsc::channel(1000);
        (
            PeerManager {
                peers: Vec::new(),
                banned_peers: Vec::new(),
                config,
                event_tx: tx,
            },
            rx,
        )
    }

    pub async fn add_peer(&mut self, info: PeerInfo) -> Result<(), NetworkError> {
        // Check if peer is banned
        if let Some((_, until)) = self.banned_peers.iter().find(|(id, _)| id == &info.node_id) {
            if Instant::now() < *until {
                return Err(NetworkError::PeerBanned);
            }
            // Remove from banned list if ban has expired
            self.banned_peers.retain(|(id, _)| id != &info.node_id);
        }

        // Check max peers
        if self.peers.len() >= self.config.max_peers {
            return Err(NetworkError::TooManyPeers);
        }

        // Add peer
        self.peers.push(info.clone());

        // Emit event
        self.event_tx
            .send(PeerEvent::Connected(info))
            .await
            .map_err(|_| NetworkError::EventChannelClosed)?;

        Ok(())
    }

    pub async fn remove_peer(&mut self, node_id: &str) -> Result<(), NetworkError> {
        if let Some(index) = self.peers.iter().position(|p| p.node_id == node_id) {
            self.peers.remove(index);
            self.event_tx
                .send(PeerEvent::Disconnected(node_id.to_string()))
                .await
                .map_err(|_| NetworkError::EventChannelClosed)?;
        }
        Ok(())
    }

    pub async fn update_reputation(
        &mut self,
        node_id: &str,
        delta: f64,
    ) -> Result<(), NetworkError> {
        // First check if we need to ban the peer
        let should_ban = if let Some(peer) = self.peers.iter_mut().find(|p| p.node_id == node_id) {
            peer.reputation = (peer.reputation + delta).clamp(0.0, 100.0);

            let score = peer.reputation;

            // Send reputation update event
            self.event_tx
                .send(PeerEvent::ReputationUpdated {
                    node_id: node_id.to_string(),
                    score,
                })
                .await
                .map_err(|_| NetworkError::EventChannelClosed)?;

            // Check if peer should be banned (return the decision)
            score < self.config.ban_threshold
        } else {
            false
        };

        // Ban the peer if needed (now we don't hold any borrow)
        if should_ban {
            self.ban_peer(node_id, self.config.ban_duration).await?;
        }

        Ok(())
    }

    pub async fn ban_peer(
        &mut self,
        node_id: &str,
        duration: Duration,
    ) -> Result<(), NetworkError> {
        let until = Instant::now() + duration;
        self.banned_peers.push((node_id.to_string(), until));

        // Remove peer if connected
        self.remove_peer(node_id).await?;

        self.event_tx
            .send(PeerEvent::Banned {
                node_id: node_id.to_string(),
                duration,
            })
            .await
            .map_err(|_| NetworkError::EventChannelClosed)?;

        Ok(())
    }

    pub fn get_peer(&self, node_id: &str) -> Option<&PeerInfo> {
        self.peers.iter().find(|p| p.node_id == node_id)
    }

    pub fn get_peers(&self) -> &[PeerInfo] {
        &self.peers
    }

    pub fn get_banned_peers(&self) -> &[(String, Instant)] {
        &self.banned_peers
    }

    pub async fn cleanup_banned(&mut self) {
        let now = Instant::now();
        self.banned_peers.retain(|(_, until)| *until > now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_peer_id() {
        let id1 = PeerId::from("peer1");
        let id2 = PeerId::from("peer1".to_string());
        let id3 = PeerId::from("peer2");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(id1.to_string(), "peer1");
    }

    #[test]
    fn test_peer_info() {
        let mut info = PeerInfo::new("peer1".to_string(), "127.0.0.1:8000".to_string());

        // Test initial values
        assert_eq!(info.address, "127.0.0.1:8000");
        assert_eq!(info.latency, Duration::from_millis(0));
        assert_eq!(info.reputation, 0.5);

        // Test updates
        info.update_latency(Duration::from_millis(100));
        info.update_reputation(0.8);

        assert_eq!(info.latency, Duration::from_millis(100));
        assert_eq!(info.reputation, 0.8);

        // Test age calculation
        sleep(Duration::from_millis(10));
        assert!(info.age() >= Duration::from_millis(10));

        // Test inactive check
        let last_seen = info.last_seen;
        sleep(Duration::from_millis(10));
        assert!(Instant::now() > last_seen);
        assert!(info.is_inactive(Duration::from_millis(5)));
        assert!(!info.is_inactive(Duration::from_millis(50)));
    }
}
