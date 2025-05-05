use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{Error, ErrorKind, Result};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket as TokioUdpSocket;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task;
use log::{debug, error, info, warn};
use bincode::{deserialize, serialize};
use serde::{Deserialize, Serialize};
use blake3::Hasher;
use rand::Rng;
use parking_lot::Mutex as ParkingMutex;
use rayon::prelude::*;
use net2::UdpSocketExt;
use anyhow::{Result, Context, anyhow};

// Constants for network configuration
const MAX_UDP_PACKET_SIZE: usize = 65507; // Max practical UDP packet size
const DEFAULT_PORT: u16 = 12345;
const CONNECTION_TIMEOUT_MS: u64 = 30000; // 30 seconds
const HEARTBEAT_INTERVAL_MS: u64 = 5000; // 5 seconds
const MAX_RETRY_COUNT: u8 = 5;
const INIT_BACKOFF_MS: u64 = 100;
const CONGESTION_WINDOW_SIZE: usize = 1024;
const DEFAULT_BUFFER_SIZE: usize = 8 * 1024 * 1024; // 8MB buffer
const BATCH_SIZE: usize = 64; // Process messages in batches

/// Custom UDP protocol optimized for blockchain communication
pub struct UdpNetwork {
    /// Socket for sending/receiving
    socket: Arc<TokioUdpSocket>,
    /// Connected peers
    peers: Arc<RwLock<HashMap<SocketAddr, PeerState>>>,
    /// Message handlers by type
    handlers: Arc<RwLock<HashMap<MessageType, mpsc::Sender<Message>>>>,
    /// Outgoing message queue
    outgoing_queue: Arc<Mutex<VecDeque<(Message, SocketAddr)>>>,
    /// Known peer addresses
    known_addresses: Arc<RwLock<HashSet<SocketAddr>>>,
    /// Protocol configuration
    config: NetworkConfig,
    /// Node identifier
    node_id: String,
    /// Local address
    local_addr: SocketAddr,
    /// Statistics
    stats: Arc<RwLock<NetworkStats>>,
    /// Shutdown channel
    shutdown_tx: tokio::sync::broadcast::Sender<()>,
    /// Shutdown receiver (for handlers)
    shutdown_rx: tokio::sync::broadcast::Receiver<()>,
}

/// Peer connection state
struct PeerState {
    /// Last time we received a message from this peer
    last_seen: Instant,
    /// Connection quality metrics
    metrics: ConnectionMetrics,
    /// Pending acknowledgments
    pending_acks: HashSet<u64>,
    /// Current congestion window
    congestion_window: usize,
    /// Sequence numbers for duplicate detection
    seen_seqs: HashSet<u64>,
    /// RTT estimator
    rtt_estimator: RttEstimator,
}

/// Connection metrics for quality monitoring
#[derive(Clone, Debug, Default)]
struct ConnectionMetrics {
    /// Packets sent
    packets_sent: u64,
    /// Packets received
    packets_received: u64,
    /// Packets lost
    packets_lost: u64,
    /// Bytes sent
    bytes_sent: u64,
    /// Bytes received
    bytes_received: u64,
    /// Round-trip time in milliseconds
    rtt_ms: u64,
    /// Packet loss rate (0.0 - 1.0)
    loss_rate: f32,
}

/// RTT estimator using TCP-style algorithm
struct RttEstimator {
    /// Smoothed RTT
    srtt: Duration,
    /// RTT variation
    rttvar: Duration,
    /// Retransmission timeout
    rto: Duration,
}

impl Default for RttEstimator {
    fn default() -> Self {
        Self {
            srtt: Duration::from_millis(500), // Initial guess
            rttvar: Duration::from_millis(250),
            rto: Duration::from_secs(1),
        }
    }
}

impl RttEstimator {
    /// Update the RTT estimate with a new measurement
    fn update(&mut self, rtt: Duration) {
        // Constants from RFC 6298
        const ALPHA: f32 = 0.125;
        const BETA: f32 = 0.25;
        const K: u32 = 4;
        
        if self.srtt.as_millis() == 0 {
            // First measurement
            self.srtt = rtt;
            self.rttvar = rtt / 2;
            self.rto = self.srtt + Duration::from_micros((K as u64) * self.rttvar.as_micros() as u64);
        } else {
            // Update estimates
            let srtt_ms = self.srtt.as_millis() as f32;
            let rtt_ms = rtt.as_millis() as f32;
            let rttvar_ms = self.rttvar.as_millis() as f32;
            
            let new_rttvar_ms = (1.0 - BETA) * rttvar_ms + BETA * (srtt_ms - rtt_ms).abs();
            let new_srtt_ms = (1.0 - ALPHA) * srtt_ms + ALPHA * rtt_ms;
            
            self.srtt = Duration::from_millis(new_srtt_ms as u64);
            self.rttvar = Duration::from_millis(new_rttvar_ms as u64);
            self.rto = self.srtt + Duration::from_micros((K as u64) * self.rttvar.as_micros() as u64);
        }
        
        // Ensure RTO is within reasonable bounds
        if self.rto < Duration::from_millis(100) {
            self.rto = Duration::from_millis(100);
        } else if self.rto > Duration::from_secs(60) {
            self.rto = Duration::from_secs(60);
        }
    }
    
    /// Get the current retransmission timeout
    fn get_rto(&self) -> Duration {
        self.rto
    }
}

/// Network statistics
#[derive(Clone, Debug, Default)]
pub struct NetworkStats {
    /// Total packets sent
    packets_sent: u64,
    /// Total packets received
    packets_received: u64,
    /// Total bytes sent
    bytes_sent: u64,
    /// Total bytes received
    bytes_received: u64,
    /// Average round-trip time
    avg_rtt_ms: f32,
    /// Duplicate packets received
    duplicate_packets: u64,
    /// Invalid packets received
    invalid_packets: u64,
    /// Active connections
    active_connections: usize,
    /// Messages processed
    messages_processed: u64,
    /// Average processing time per message
    avg_processing_time_us: f32,
}

/// Message types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageType {
    /// Handshake message
    Handshake,
    /// Heartbeat message
    Heartbeat,
    /// Transaction message
    Transaction,
    /// Block message
    Block,
    /// Consensus message
    Consensus,
    /// Peer discovery message
    PeerDiscovery,
    /// Direct message to node
    Direct,
    /// Acknowledgment message
    Ack,
    /// Gossip message
    Gossip,
    /// State sync message
    StateSync,
    /// Custom message
    Custom(u16),
}

/// Flags for message control
bitflags::bitflags! {
    #[derive(Serialize, Deserialize)]
    pub struct MessageFlags: u16 {
        /// Request acknowledgment
        const REQUEST_ACK = 0x0001;
        /// This is an acknowledgment
        const IS_ACK = 0x0002;
        /// Message is encrypted
        const ENCRYPTED = 0x0004;
        /// Message is compressed
        const COMPRESSED = 0x0008;
        /// Message is a fragment
        const FRAGMENT = 0x0010;
        /// Last fragment in a series
        const LAST_FRAGMENT = 0x0020;
        /// High priority message
        const HIGH_PRIORITY = 0x0040;
        /// No retry on failure
        const NO_RETRY = 0x0080;
        /// Message should be relayed
        const RELAY = 0x0100;
        /// Message is signed
        const SIGNED = 0x0200;
    }
}

/// Network configuration
#[derive(Clone, Debug)]
pub struct NetworkConfig {
    /// Bind address
    pub bind_addr: SocketAddr,
    /// Maximum message size
    pub max_message_size: usize,
    /// Enable reliable delivery
    pub reliable_delivery: bool,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Buffer size
    pub buffer_size: usize,
    /// Enable compression
    pub compression: bool,
    /// Enable encryption
    pub encryption: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), DEFAULT_PORT),
            max_message_size: MAX_UDP_PACKET_SIZE,
            reliable_delivery: true,
            connection_timeout_ms: CONNECTION_TIMEOUT_MS,
            buffer_size: DEFAULT_BUFFER_SIZE,
            compression: true,
            encryption: true,
        }
    }
}

/// Network message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message header
    pub header: MessageHeader,
    /// Message payload
    pub payload: Vec<u8>,
}

/// Message header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHeader {
    /// Protocol version
    pub version: u8,
    /// Message type
    pub msg_type: MessageType,
    /// Message ID
    pub id: u64,
    /// Sequence number
    pub sequence: u64,
    /// Flags
    pub flags: MessageFlags,
    /// Timestamp (ms since epoch)
    pub timestamp: u64,
    /// Sender node ID
    pub sender: String,
    /// Recipient node ID (empty for broadcast)
    pub recipient: String,
    /// TTL for message propagation
    pub ttl: u8,
    /// Fragment information (if message is fragmented)
    pub fragment_info: Option<FragmentInfo>,
}

/// Fragment information for large messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentInfo {
    /// Original message ID
    pub original_id: u64,
    /// Fragment index
    pub index: u16,
    /// Total fragments
    pub total: u16,
}

impl UdpNetwork {
    /// Create a new UDP network
    pub async fn new(config: NetworkConfig, node_id: String) -> Result<Self> {
        // Create UDP socket
        let socket = if config.bind_addr.is_ipv4() {
            TokioUdpSocket::bind(config.bind_addr).await?
        } else {
            // For IPv6, enable dual-stack mode
            let socket = std::net::UdpSocket::bind(config.bind_addr)?;
            socket.set_only_v6(false)?;
            TokioUdpSocket::from_std(socket)?
        };
        
        // Set socket options
        let std_socket = socket.into_std()?;
        std_socket.set_nonblocking(true)?;
        std_socket.set_recv_buffer_size(config.buffer_size)?;
        std_socket.set_send_buffer_size(config.buffer_size)?;
        let socket = TokioUdpSocket::from_std(std_socket)?;
        
        let local_addr = socket.local_addr()?;
        
        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(1);
        
        // Create UDP network
        let network = Self {
            socket: Arc::new(socket),
            peers: Arc::new(RwLock::new(HashMap::new())),
            handlers: Arc::new(RwLock::new(HashMap::new())),
            outgoing_queue: Arc::new(Mutex::new(VecDeque::new())),
            known_addresses: Arc::new(RwLock::new(HashSet::new())),
            config,
            node_id,
            local_addr,
            stats: Arc::new(RwLock::new(NetworkStats::default())),
            shutdown_tx,
            shutdown_rx,
        };
        
        Ok(network)
    }
    
    /// Start the network
    pub async fn start(&self) -> Result<()> {
        let socket_clone = self.socket.clone();
        let peers_clone = self.peers.clone();
        let handlers_clone = self.handlers.clone();
        let stats_clone = self.stats.clone();
        let outgoing_queue_clone = self.outgoing_queue.clone();
        let mut shutdown_rx = self.shutdown_rx.resubscribe();
        
        // Spawn receiver task
        tokio::spawn(async move {
            let mut buf = vec![0u8; MAX_UDP_PACKET_SIZE];
            
            loop {
                tokio::select! {
                    recv_result = shutdown_rx.recv() => {
                        match recv_result {
                            Ok(_) => {
                                log::info!("Receiver task shutting down");
                                break;
                            },
                            Err(e) => {
                                log::error!("Error receiving shutdown signal: {}", e);
                                break;
                            }
                        }
                    }
                    result = socket_clone.recv_from(&mut buf) => {
                        match result {
                            Ok((size, addr)) => {
                                if size > 0 {
                                    let data = &buf[..size];
                                    if let Err(e) = UdpNetwork::handle_incoming_packet(
                                        data, addr, &peers_clone, &handlers_clone, &stats_clone
                                    ).await {
                                        log::error!("Error handling packet: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!("Error receiving packet: {}", e);
                                tokio::time::sleep(Duration::from_millis(10)).await;
                            }
                        }
                    }
                }
            }
        });
        
        // Spawn sender task
        let socket_clone = self.socket.clone();
        let peers_clone = self.peers.clone();
        let stats_clone = self.stats.clone();
        let mut shutdown_rx = self.shutdown_rx.resubscribe();
        
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv().await => {
                        log::info!("Sender task shutting down");
                        break;
                    }
                    _ = tokio::time::sleep(Duration::from_millis(1)) => {
                        // Check outgoing queue
                        let mut to_send = None;
                        {
                            let mut queue = outgoing_queue_clone.lock();
                            if !queue.is_empty() {
                                to_send = queue.pop_front();
                            }
                        }
                        
                        if let Some((message, addr)) = to_send {
                            if let Err(e) = Self::send_packet(&socket_clone, &message, addr, &peers_clone, &stats_clone).await {
                                log::error!("Error sending packet: {}", e);
                            }
                        } else {
                            // No messages, sleep a bit longer
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        }
                    }
                }
            }
        });
        
        // Spawn ping task
        let network = self.clone();
        let mut shutdown_rx = self.shutdown_rx.resubscribe();
        
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv().await => {
                        log::info!("Ping task shutting down");
                        break;
                    }
                    _ = tokio::time::sleep(Duration::from_millis(HEARTBEAT_INTERVAL_MS)) => {
                        network.ping_all_peers().await.ok();
                    }
                }
            }
        });
        
        // Spawn cleanup task
        let network = self.clone();
        let mut shutdown_rx = self.shutdown_rx.resubscribe();
        
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv().await => {
                        log::info!("Cleanup task shutting down");
                        break;
                    }
                    _ = tokio::time::sleep(Duration::from_millis(CONNECTION_TIMEOUT_MS)) => {
                        network.cleanup_peers().await.ok();
                    }
                }
            }
        });
        
        log::info!("UDP network started on {}", self.local_addr);
        Ok(())
    }
    
    /// Handle an incoming packet
    async fn handle_incoming_packet(
        data: &[u8],
        addr: SocketAddr,
        peers: &Arc<RwLock<HashMap<SocketAddr, PeerState>>>,
        handlers: &Arc<RwLock<HashMap<MessageType, mpsc::Sender<Message>>>>,
        stats: &Arc<RwLock<NetworkStats>>,
    ) -> Result<()> {
        // Update receive stats
        {
            let mut stats_guard = stats.write().await;
            stats_guard.packets_received += 1;
            stats_guard.bytes_received += data.len() as u64;
        }
        
        // Deserialize the message
        let message: Message = match bincode::deserialize(data) {
            Ok(msg) => msg,
            Err(e) => {
                let mut stats_guard = stats.write().await;
                stats_guard.invalid_packets += 1;
                return Err(Error::new(ErrorKind::InvalidData, format!("Error deserializing packet: {}", e)));
            }
        };
        
        // Update peer state
        let update_result = {
            let mut peers_guard = peers.write().await;
            let now = Instant::now();
            
            // Get or create peer state
            let state = peers_guard.entry(addr).or_insert_with(|| {
                debug!("New peer connected: {}", addr);
                PeerState {
                    last_seen: now,
                    metrics: ConnectionMetrics::default(),
                    pending_acks: HashSet::new(),
                    congestion_window: CONGESTION_WINDOW_SIZE,
                    seen_seqs: HashSet::new(),
                    rtt_estimator: RttEstimator::default(),
                }
            });
            
            // Update last seen
            state.last_seen = now;
            
            // Update metrics
            state.metrics.packets_received += 1;
            state.metrics.bytes_received += data.len() as u64;
            
            // Check for duplicates
            if state.seen_seqs.contains(&message.header.sequence) {
                let mut stats_guard = stats.write().await;
                stats_guard.duplicate_packets += 1;
                return Ok(());
            }
            
            // Add to seen sequences
            state.seen_seqs.insert(message.header.sequence);
            
            // If this is an ACK, update pending ACKs
            if message.header.flags.contains(MessageFlags::IS_ACK) {
                if let Some(fragment_info) = &message.header.fragment_info {
                    state.pending_acks.remove(&fragment_info.original_id);
                }
            }
            
            // If message requests ACK, send one
            if message.header.flags.contains(MessageFlags::REQUEST_ACK) {
                Some((message.header.id, message.header.sender.clone()))
            } else {
                None
            }
        };
        
        // If needed, send ACK
        if let Some((msg_id, sender)) = update_result {
            // Create ACK message
            let ack_message = Message {
                header: MessageHeader {
                    version: 1,
                    msg_type: MessageType::Ack,
                    id: rand::thread_rng().gen(),
                    sequence: rand::thread_rng().gen(),
                    flags: MessageFlags::IS_ACK | MessageFlags::NO_RETRY,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                    sender: "node_id".to_string(), // Using placeholder - in practice we'd use self.node_id
                    recipient: sender,
                    ttl: 1,
                    fragment_info: Some(FragmentInfo {
                        original_id: msg_id,
                        index: 0,
                        total: 1,
                    }),
                },
                payload: Vec::new(),
            };
            
            // Send ACK - in actual implementation we'd use self.send_message
            // For this example, we'll just log it
            debug!("Would send ACK for message {}", msg_id);
        }
        
        // Process the message based on type
        let msg_type = message.header.msg_type;
        let handlers_guard = handlers.read().await;
        
        if let Some(handler) = handlers_guard.get(&msg_type) {
            // Send to appropriate handler
            if let Err(e) = handler.send(message.clone()).await {
                error!("Error sending message to handler: {}", e);
            }
        }
        
        // Update stats
        {
            let mut stats_guard = stats.write().await;
            stats_guard.messages_processed += 1;
        }
        
        Ok(())
    }
    
    /// Stop the network
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping UDP network");
        
        // Send shutdown signal
        let _ = self.shutdown_tx.send(());
        
        // Wait for tasks to complete
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        Ok(())
    }
    
    /// Connect to a peer
    pub async fn connect(&self, addr: SocketAddr) -> Result<()> {
        // Add to known addresses
        {
            let mut known = self.known_addresses.write().await;
            known.insert(addr);
        }
        
        // Create handshake message
        let handshake = Message {
            header: MessageHeader {
                version: 1,
                msg_type: MessageType::Handshake,
                id: rand::thread_rng().gen(),
                sequence: rand::thread_rng().gen(),
                flags: MessageFlags::REQUEST_ACK,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                sender: self.node_id.clone(),
                recipient: String::new(),
                ttl: 1,
                fragment_info: None,
            },
            payload: Vec::new(),
        };
        
        // Send handshake
        self.send_message(handshake, addr).await?;
        
        Ok(())
    }
    
    /// Send a message to a specific address
    pub async fn send_message(&self, message: Message, addr: SocketAddr) -> Result<()> {
        // Queue the message
        let mut queue = self.outgoing_queue.lock().await;
        queue.push_back((message, addr));
        
        Ok(())
    }
    
    /// Broadcast a message to all peers
    pub async fn broadcast(&self, message: Message) -> Result<()> {
        // Get all peer addresses
        let peers: Vec<SocketAddr> = {
            let peers_guard = self.peers.read().await;
            peers_guard.keys().cloned().collect()
        };
        
        // Send to all peers
        let mut queue = self.outgoing_queue.lock().await;
        for addr in peers {
            // Clone the message for each peer
            let mut msg_copy = message.clone();
            
            // Update sequence for each copy
            msg_copy.header.sequence = rand::thread_rng().gen();
            
            queue.push_back((msg_copy, addr));
        }
        
        Ok(())
    }
    
    /// Register a message handler
    pub async fn register_handler(&self, msg_type: MessageType, channel: mpsc::Sender<Message>) -> Result<()> {
        let mut handlers = self.handlers.write().await;
        handlers.insert(msg_type, channel);
        Ok(())
    }
    
    /// Get current network statistics
    pub async fn get_stats(&self) -> NetworkStats {
        self.stats.read().await.clone()
    }
    
    /// Get peer metrics
    pub async fn get_peer_metrics(&self, addr: SocketAddr) -> Option<ConnectionMetrics> {
        let peers = self.peers.read().await;
        peers.get(&addr).map(|state| state.metrics.clone())
    }
    
    /// Send a large message in fragments
    pub async fn send_large_message(&self, mut payload: Vec<u8>, msg_type: MessageType, addr: SocketAddr) -> Result<()> {
        // Calculate maximum payload size per fragment
        // Allow space for headers and overhead
        let max_payload_size = self.config.max_message_size - 256;
        
        // If payload fits in a single packet, send it directly
        if payload.len() <= max_payload_size {
            let message = Message {
                header: MessageHeader {
                    version: 1,
                    msg_type,
                    id: rand::thread_rng().gen(),
                    sequence: rand::thread_rng().gen(),
                    flags: MessageFlags::REQUEST_ACK,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                    sender: self.node_id.clone(),
                    recipient: String::new(),
                    ttl: 1,
                    fragment_info: None,
                },
                payload,
            };
            
            return self.send_message(message, addr).await;
        }
        
        // Generate a common ID for all fragments
        let original_id = rand::thread_rng().gen();
        
        // Split into fragments
        let fragment_count = (payload.len() + max_payload_size - 1) / max_payload_size;
        let mut fragments = Vec::with_capacity(fragment_count);
        
        for i in 0..fragment_count {
            let start = i * max_payload_size;
            let end = std::cmp::min(start + max_payload_size, payload.len());
            
            // Last fragment?
            let is_last = i == fragment_count - 1;
            
            // Create fragment message
            let mut flags = MessageFlags::FRAGMENT | MessageFlags::REQUEST_ACK;
            if is_last {
                flags |= MessageFlags::LAST_FRAGMENT;
            }
            
            let fragment = Message {
                header: MessageHeader {
                    version: 1,
                    msg_type,
                    id: rand::thread_rng().gen(),
                    sequence: rand::thread_rng().gen(),
                    flags,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                    sender: self.node_id.clone(),
                    recipient: String::new(),
                    ttl: 1,
                    fragment_info: Some(FragmentInfo {
                        original_id,
                        index: i as u16,
                        total: fragment_count as u16,
                    }),
                },
                payload: payload[start..end].to_vec(),
            };
            
            fragments.push(fragment);
        }
        
        // Send all fragments
        for fragment in fragments {
            self.send_message(fragment, addr).await?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_udp_network_creation() {
        let config = NetworkConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..Default::default()
        };
        
        let network = UdpNetwork::new(config, "test_node".to_string()).await.unwrap();
        assert!(network.local_addr.port() > 0);
    }
    
    #[tokio::test]
    async fn test_message_serialization() {
        let message = Message {
            header: MessageHeader {
                version: 1,
                msg_type: MessageType::Transaction,
                id: 12345,
                sequence: 67890,
                flags: MessageFlags::REQUEST_ACK,
                timestamp: 1234567890,
                sender: "node1".to_string(),
                recipient: "node2".to_string(),
                ttl: 5,
                fragment_info: None,
            },
            payload: vec![1, 2, 3, 4, 5],
        };
        
        let serialized = bincode::serialize(&message).unwrap();
        let deserialized: Message = bincode::deserialize(&serialized).unwrap();
        
        assert_eq!(deserialized.header.id, message.header.id);
        assert_eq!(deserialized.header.msg_type, message.header.msg_type);
        assert_eq!(deserialized.payload, message.payload);
    }
} 