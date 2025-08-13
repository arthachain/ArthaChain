use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableInstant {
    #[serde(with = "serde_instant")]
    pub instant: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableDuration {
    #[serde(with = "serde_duration")]
    pub duration: Duration,
}

impl SerializableInstant {
    pub fn now() -> Self {
        Self {
            instant: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.instant.elapsed()
    }
}

mod serde_instant {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::{Duration, Instant};

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = instant.duration_since(Instant::now());
        serializer.serialize_i64(duration.as_secs() as i64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = i64::deserialize(deserializer)?;
        Ok(Instant::now() + Duration::from_secs(secs as u64))
    }
}

mod serde_duration {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_i64(duration.as_secs() as i64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = i64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs as u64))
    }
}

// Implementing PartialEq for SerializableInstant
impl PartialEq for SerializableInstant {
    fn eq(&self, other: &Self) -> bool {
        // Since Instant doesn't implement PartialEq, we can compare them
        // by calculating the duration since a fixed point
        let base = Instant::now();
        self.instant.duration_since(base).as_nanos()
            == other.instant.duration_since(base).as_nanos()
    }
}

// Implementing PartialEq for SerializableDuration
impl PartialEq for SerializableDuration {
    fn eq(&self, other: &Self) -> bool {
        self.duration == other.duration
    }
}

/// Network node identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    /// Create a new node ID
    pub fn new(id: String) -> Self {
        Self(id)
    }

    /// Generate a random node ID
    pub fn random() -> Self {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();
        let id: u64 = rng.gen();
        Self(format!("node_{:016x}", id))
    }

    /// Get the string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Get the inner string
    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<String> for NodeId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for NodeId {
    fn from(id: &str) -> Self {
        Self(id.to_string())
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Total peers connected
    pub peer_count: usize,
    /// Messages sent
    pub messages_sent: usize,
    /// Messages received
    pub messages_received: usize,
    /// Known peers
    pub known_peers: HashSet<String>,
    /// Bytes sent
    pub bytes_sent: usize,
    /// Bytes received
    pub bytes_received: usize,
    /// Active connections
    pub active_connections: usize,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Success rate
    pub success_rate: f64,
    /// Blocks received
    pub blocks_received: usize,
    /// Transactions received
    pub transactions_received: usize,
    /// Last activity timestamp
    pub last_activity: chrono::DateTime<chrono::Utc>,
    /// Packets sent
    pub packets_sent: usize,
    /// Packets received
    pub packets_received: usize,
    /// Bandwidth usage in bytes per second
    pub bandwidth_usage: usize,
}

impl Default for NetworkStats {
    fn default() -> Self {
        Self {
            peer_count: 0,
            messages_sent: 0,
            messages_received: 0,
            known_peers: HashSet::new(),
            bytes_sent: 0,
            bytes_received: 0,
            active_connections: 0,
            avg_latency_ms: 0.0,
            success_rate: 1.0,
            blocks_received: 0,
            transactions_received: 0,
            last_activity: chrono::Utc::now(),
            packets_sent: 0,
            packets_received: 0,
            bandwidth_usage: 0,
        }
    }
}

impl NetworkStats {
    /// Update connection count
    pub fn update_peer_count(&mut self, count: usize) {
        self.peer_count = count;
        self.active_connections = count;
        self.last_activity = chrono::Utc::now();
    }

    /// Record message sent
    pub fn record_message_sent(&mut self, bytes: usize) {
        self.messages_sent += 1;
        self.bytes_sent += bytes;
        self.packets_sent += 1;
        self.last_activity = chrono::Utc::now();
    }

    /// Record message received
    pub fn record_message_received(&mut self, bytes: usize) {
        self.messages_received += 1;
        self.bytes_received += bytes;
        self.packets_received += 1;
        self.last_activity = chrono::Utc::now();
    }

    /// Add known peer
    pub fn add_known_peer(&mut self, peer_id: String) {
        self.known_peers.insert(peer_id);
        self.last_activity = chrono::Utc::now();
    }

    /// Update latency
    pub fn update_latency(&mut self, latency_ms: f64) {
        // Simple moving average
        self.avg_latency_ms = (self.avg_latency_ms * 0.9) + (latency_ms * 0.1);
        self.last_activity = chrono::Utc::now();
    }

    /// Calculate bandwidth usage (bytes per second)
    pub fn calculate_bandwidth(&mut self, time_window_secs: u64) {
        if time_window_secs > 0 {
            self.bandwidth_usage =
                (self.bytes_sent + self.bytes_received) / time_window_secs as usize;
        }
    }
}

/// Network message priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

impl Default for MessagePriority {
    fn default() -> Self {
        MessagePriority::Normal
    }
}

/// Network connection status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    /// Connection is being established
    Connecting,
    /// Connection is active and healthy
    Connected,
    /// Connection is established but experiencing issues
    Degraded,
    /// Connection is lost or failed
    Disconnected,
    /// Connection is being closed
    Closing,
}

impl Default for ConnectionStatus {
    fn default() -> Self {
        ConnectionStatus::Disconnected
    }
}

/// Peer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Peer node ID
    pub node_id: NodeId,
    /// Connection status
    pub status: ConnectionStatus,
    /// Network addresses
    pub addresses: Vec<String>,
    /// Last seen timestamp
    pub last_seen: chrono::DateTime<chrono::Utc>,
    /// Connection quality score (0.0 - 1.0)
    pub quality_score: f64,
    /// Number of failed connection attempts
    pub failed_attempts: u32,
    /// Peer capabilities
    pub capabilities: Vec<String>,
}

impl PeerInfo {
    /// Create new peer info
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            status: ConnectionStatus::default(),
            addresses: Vec::new(),
            last_seen: chrono::Utc::now(),
            quality_score: 1.0,
            failed_attempts: 0,
            capabilities: Vec::new(),
        }
    }

    /// Update last seen timestamp
    pub fn update_last_seen(&mut self) {
        self.last_seen = chrono::Utc::now();
    }

    /// Record failed connection attempt
    pub fn record_failed_attempt(&mut self) {
        self.failed_attempts += 1;
        self.quality_score = (self.quality_score * 0.9).max(0.1);
    }

    /// Record successful connection
    pub fn record_successful_connection(&mut self) {
        self.failed_attempts = 0;
        self.status = ConnectionStatus::Connected;
        self.quality_score = (self.quality_score * 0.9 + 0.1).min(1.0);
        self.update_last_seen();
    }
}
