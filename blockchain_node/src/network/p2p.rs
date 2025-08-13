use anyhow::{anyhow, Context, Result};
use libp2p::{
    core::{transport::Transport, upgrade},
    futures::StreamExt,
    gossipsub::{self, Behaviour as Gossipsub, Event as GossipsubEvent, IdentTopic, Topic},
    identity,
    kad::{self, store::MemoryStore, QueryResult},
    noise,
    ping::{self, Behaviour as PingBehaviour, Event as PingEvent},
    swarm::{NetworkBehaviour, Swarm, SwarmEvent},
    tcp, yamux, PeerId, Transport as _,
};
use log::{debug, info, warn};
use std::collections::HashSet;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::config::Config;
use crate::ledger::block::Block;
use crate::ledger::state::State;
use crate::ledger::transaction::Transaction;
use crate::network::dos_protection::DosProtection;
use crate::types::Hash;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use super::dos_protection::DosConfig;

/// Peer discovery message for network announcements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerDiscoveryMessage {
    pub node_id: String,
    pub listen_addresses: Vec<String>,
    pub protocol_version: String,
    pub services: Vec<String>,
    pub timestamp: u64, // Unix timestamp
}

/// Discovered peer information
#[derive(Debug, Clone)]
pub struct DiscoveredPeer {
    pub id: String,
    pub address: String,
    pub protocol_version: String,
    pub services: Vec<String>,
    pub discovery_method: PeerDiscoveryMethod,
    pub last_seen: Instant,
}

/// Peer discovery methods
#[derive(Debug, Clone)]
pub enum PeerDiscoveryMethod {
    MDNS,
    DHT,
    DNSSeed,
    UPnP,
    PeerExchange,
    Bootstrap,
}

/// UPnP peer search configuration
#[derive(Debug, Clone)]
pub struct UpnpPeerSearch {
    pub service_type: &'static str,
    pub search_interval: Duration,
}

impl UpnpPeerSearch {
    /// Continuous UPnP discovery
    async fn continuous_discovery(&self) {
        let mut interval = tokio::time::interval(self.search_interval);

        loop {
            interval.tick().await;

            // Search for UPnP devices
            match self.search_upnp_devices().await {
                Ok(devices) => {
                    info!("Found {} UPnP devices", devices.len());
                    // Process discovered devices
                }
                Err(e) => warn!("UPnP discovery failed: {}", e),
            }
        }
    }

    /// Search for UPnP devices
    async fn search_upnp_devices(&self) -> Result<Vec<DiscoveredPeer>> {
        // UPnP device discovery implementation would use actual UPnP library
        Ok(vec![])
    }
}

/// Network error types
#[derive(Debug, Error)]
pub enum NetworkError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Block not found: {0}")]
    BlockNotFound(Hash),

    #[error("Lock error: {0}")]
    LockError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Connectivity error: {0}")]
    ConnectivityError(String),

    #[error("Message error: {0}")]
    MessageError(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Message types for P2P communication
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum NetworkMessage {
    /// New block proposal
    BlockProposal(Block),
    /// Vote for a block
    BlockVote {
        block_hash: Hash,
        validator_id: String,
        signature: Vec<u8>,
    },
    /// Transaction gossip
    TransactionGossip(Transaction),
    /// Request for a specific block
    BlockRequest { block_hash: Hash, requester: String },
    /// Response to a block request
    BlockResponse { block: Block, responder: String },
    /// Shard assignment notification
    ShardAssignment {
        node_id: String,
        shard_id: u64,
        timestamp: u64,
    },
    /// Cross-shard message
    CrossShardMessage {
        from_shard: u64,
        to_shard: u64,
        message_type: CrossShardMessageType,
        payload: Vec<u8>,
    },
}

/// Cross-shard message types
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum CrossShardMessageType {
    /// Block finalization notification
    BlockFinalization,
    /// Transaction forwarding
    TransactionForward,
    /// State synchronization
    StateSync,
    /// Shard reconfiguration
    ShardReconfig,
    /// Transaction between shards
    Transaction,
}

/// Network statistics
#[derive(Debug, Default, Clone)]
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
    /// Bandwidth usage
    pub bandwidth_usage: usize,
}

/// Block propagation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlockPriority {
    High = 3,
    Medium = 2,
    Low = 1,
}

/// Block propagation metadata
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BlockPropagationMeta {
    pub block_hash: Hash,
    pub priority: BlockPriority,
    pub timestamp: Instant,
    pub size: usize,
    pub compressed_size: Option<usize>,
    pub propagation_count: usize,
    pub last_propagation: Option<Instant>,
}

impl Ord for BlockPropagationMeta {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare priority first, then timestamp
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.timestamp.cmp(&other.timestamp))
            .then_with(|| self.block_hash.as_ref().cmp(&other.block_hash.as_ref()))
    }
}

impl PartialOrd for BlockPropagationMeta {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Block propagation queue
#[derive(Debug)]
pub struct BlockPropagationQueue {
    queue: BinaryHeap<(BlockPriority, Instant, BlockPropagationMeta)>,
    max_size: usize,
    current_size: usize,
}

impl BlockPropagationQueue {
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: BinaryHeap::new(),
            max_size,
            current_size: 0,
        }
    }

    pub fn push(&mut self, meta: BlockPropagationMeta) {
        if self.current_size >= self.max_size {
            if let Some((_, _, oldest)) = self.queue.pop() {
                self.current_size -= oldest.size;
            }
        }
        self.current_size += meta.size;
        self.queue.push((meta.priority, meta.timestamp, meta));
    }

    pub fn pop(&mut self) -> Option<BlockPropagationMeta> {
        if let Some((_, _, meta)) = self.queue.pop() {
            self.current_size -= meta.size;
            Some(meta)
        } else {
            None
        }
    }
}

/// Enhanced block propagation configuration
#[derive(Debug, Clone)]
pub struct BlockPropagationConfig {
    pub max_queue_size: usize,
    pub compression_threshold: usize,
    pub propagation_timeout: Duration,
    pub max_propagation_count: usize,
    pub bandwidth_limit: usize,
}

impl Default for BlockPropagationConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            compression_threshold: 1024 * 1024, // 1MB
            propagation_timeout: Duration::from_secs(5),
            max_propagation_count: 3,
            bandwidth_limit: 1024 * 1024 * 10, // 10MB/s
        }
    }
}

/// Combine all network behaviors
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "ComposedEvent")]
pub struct ComposedBehaviour {
    pub gossipsub: Gossipsub,
    pub kademlia: kad::Behaviour<MemoryStore>,
    pub ping: PingBehaviour,
}

/// Generated event from the network behaviour
#[derive(Debug)]
pub enum ComposedEvent {
    Gossipsub(GossipsubEvent),
    Kademlia(kad::Event),
    Ping(PingEvent),
}

impl From<GossipsubEvent> for ComposedEvent {
    fn from(event: GossipsubEvent) -> Self {
        ComposedEvent::Gossipsub(event)
    }
}

impl From<kad::Event> for ComposedEvent {
    fn from(event: kad::Event) -> Self {
        ComposedEvent::Kademlia(event)
    }
}

impl From<PingEvent> for ComposedEvent {
    fn from(event: PingEvent) -> Self {
        ComposedEvent::Ping(event)
    }
}

/// PeerConnection information
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PeerConnection {
    peer_id: String,
    connected_at: Instant,
    bytes_sent: usize,
    bytes_received: usize,
}

/// Peer information
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PeerInfo {
    peer_id: String,
    addresses: Vec<String>,
    last_seen: Instant,
}

/// P2PNetwork handles peer-to-peer communication
#[derive(Debug)]
pub struct P2PNetwork {
    /// Node configuration
    config: Config,
    /// Blockchain state
    state: Arc<RwLock<State>>,
    /// PeerId of this node
    peer_id: PeerId,
    /// Channel for receiving messages from other components
    message_rx: mpsc::Receiver<NetworkMessage>,
    /// Channel for sending messages to other components
    message_tx: mpsc::Sender<NetworkMessage>,
    /// Channel for shutdown signal
    #[allow(dead_code)]
    shutdown_signal: mpsc::Sender<()>,
    /// Network statistics
    stats: Arc<RwLock<NetworkStats>>,
    /// Shard ID this node belongs to
    shard_id: u64,
    /// Peer connections
    #[allow(dead_code)]
    peers: Arc<RwLock<HashMap<String, PeerConnection>>>,
    /// Known peers
    #[allow(dead_code)]
    known_peers: Arc<RwLock<HashSet<PeerInfo>>>,
    /// Running state
    #[allow(dead_code)]
    running: Arc<RwLock<bool>>,
    /// Block propagation queue
    _block_propagation_queue: Arc<RwLock<BlockPropagationQueue>>,
    /// Block topic
    block_topic: IdentTopic,
    /// Transaction topic
    tx_topic: IdentTopic,
    /// Vote topic
    vote_topic: IdentTopic,
    /// Cross-shard topic
    cross_shard_topic: IdentTopic,
    /// DoS protection
    dos_protection: Arc<DosProtection>,
}

impl P2PNetwork {
    /// Create a new P2P network instance
    pub async fn new(
        config: Config,
        state: Arc<RwLock<State>>,
        shutdown_signal: mpsc::Sender<()>,
    ) -> Result<Self> {
        // Create message channels
        let (message_tx, message_rx) = mpsc::channel(100);

        // Generate or load PeerId
        let keypair = identity::Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        info!("Local peer id: {peer_id}");

        // Get shard ID from config
        let shard_id = config.sharding.shard_id;

        // Create DoS protection
        let dos_config = DosConfig::default();
        let dos_protection = DosProtection::new(dos_config);

        Ok(Self {
            config,
            state,
            peer_id,
            message_rx,
            message_tx,
            shutdown_signal,
            stats: Arc::new(RwLock::new(NetworkStats::default())),
            shard_id,
            peers: Arc::new(RwLock::new(HashMap::new())),
            known_peers: Arc::new(RwLock::new(HashSet::new())),
            running: Arc::new(RwLock::new(false)),
            _block_propagation_queue: Arc::new(RwLock::new(BlockPropagationQueue::new(1000))),
            block_topic: IdentTopic::new("blocks"),
            tx_topic: IdentTopic::new("transactions"),
            vote_topic: IdentTopic::new(format!("votes-shard-{shard_id}")),
            cross_shard_topic: IdentTopic::new("cross-shard"),
            dos_protection: Arc::new(DosProtection::new(DosConfig::default())),
        })
    }

    /// Start the P2P network
    pub async fn start(&mut self) -> Result<JoinHandle<()>> {
        let peer_id = self.peer_id;
        let config = self.config.clone();
        let state = self.state.clone();
        let mut message_rx = std::mem::replace(&mut self.message_rx, mpsc::channel(1).1);
        let message_tx = self.message_tx.clone();
        let stats = self.stats.clone();
        let shard_id = self.shard_id;
        let block_topic = self.block_topic.clone();
        let tx_topic = self.tx_topic.clone();
        let vote_topic = self.vote_topic.clone();
        let cross_shard_topic = self.cross_shard_topic.clone();
        let dos_protection = self.dos_protection.clone();

        // Create swarm
        let keypair = identity::Keypair::generate_ed25519();
        let _peer_id = PeerId::from(keypair.public());

        // Create TCP transport with noise encryption and yamux multiplexing
        let transport = tcp::tokio::Transport::new(tcp::Config::default().nodelay(true))
            .upgrade(upgrade::Version::V1)
            .authenticate(noise::Config::new(&keypair).unwrap())
            .multiplex(yamux::Config::default())
            .boxed();

        // Create behavior
        let behaviour = Self::create_behaviour(peer_id)?;

        // Build swarm
        let mut swarm = Swarm::new(
            transport,
            behaviour,
            peer_id,
            libp2p::swarm::Config::without_executor(),
        );

        // Subscribe to topics
        swarm.behaviour_mut().gossipsub.subscribe(&block_topic)?;
        swarm.behaviour_mut().gossipsub.subscribe(&tx_topic)?;
        swarm.behaviour_mut().gossipsub.subscribe(&vote_topic)?;
        swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&cross_shard_topic)?;

        // Listen on all interfaces
        let listen_addr = format!("/ip4/0.0.0.0/tcp/{}", config.network.p2p_port)
            .parse()
            .context("Failed to parse listen address")?;

        swarm
            .listen_on(listen_addr)
            .context("Failed to start listening")?;

        // Connect to bootstrap peers
        for addr in &config.network.bootstrap_nodes {
            match addr.parse::<libp2p::Multiaddr>() {
                Ok(peer_addr) => {
                    if let Err(e) = swarm.dial(peer_addr) {
                        warn!("Failed to dial bootstrap peer {addr}: {e}");
                    }
                }
                Err(e) => warn!("Failed to parse bootstrap peer address {addr}: {e}"),
            }
        }

        // Move swarm to the task
        let mut swarm_for_task = swarm;

        let handle = tokio::spawn(async move {
            info!("P2P network started");

            let mut discovery_timer = tokio::time::interval(Duration::from_secs(30));

            loop {
                tokio::select! {
                    // Process incoming network events
                    event = swarm_for_task.select_next_some() => {
                        match event {
                            SwarmEvent::Behaviour(ComposedEvent::Gossipsub(GossipsubEvent::Message { propagation_source: _, message_id: _, message })) => {
                                {
                                    let mut stats_guard = stats.write().await;
                                    stats_guard.messages_received += 1;
                                    stats_guard.bytes_received += message.data.len();
                                }

                                if let Err(e) = Self::handle_pubsub_message(&message, &message_tx, &state, &dos_protection).await {
                                    warn!("Error handling pubsub message: {e}");
                                }
                            },
                            SwarmEvent::Behaviour(ComposedEvent::Ping(ping_evt)) => {
                                // Use a simpler string representation for ping events
                                debug!("Received ping event: {ping_evt:?}");
                            },
                            SwarmEvent::Behaviour(ComposedEvent::Kademlia(kad::Event::OutboundQueryProgressed { result, .. })) => {
                                match result {
                                    QueryResult::GetProviders(Ok(_)) => {
                                        // Just report that the query completed successfully
                                        debug!("GetProviders query completed successfully");
                                    },
                                    QueryResult::GetProviders(Err(err)) => {
                                        warn!("Failed to get providers: {err}");
                                    },
                                    _ => {}
                                }
                            },
                            SwarmEvent::NewListenAddr { address, .. } => {
                                info!("Listening on {address}");
                            },
                            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                                info!("Connected to {peer_id}");

                                {
                                    let mut stats_guard = stats.write().await;
                                    stats_guard.known_peers.insert(peer_id.to_string());
                                    stats_guard.peer_count = stats_guard.known_peers.len();
                                }
                            },
                            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                                info!("Disconnected from {peer_id}: {cause:?}");

                                {
                                    let mut stats_guard = stats.write().await;
                                    stats_guard.peer_count = swarm_for_task.connected_peers().count();
                                }
                            },
                            _ => {}
                        }
                    },

                    // Process outgoing messages from other components
                    Some(message) = message_rx.recv() => {
                        if let Err(e) = Self::publish_message(&mut swarm_for_task, message, &block_topic, &tx_topic, &vote_topic, &cross_shard_topic, shard_id, &stats, &dos_protection).await {
                            warn!("Error publishing message: {e}");
                        }
                    },

                    // Periodically run Kademlia bootstrap to discover more peers
                    _ = discovery_timer.tick() => {
                        debug!("Running Kademlia bootstrap");
                        if let Err(e) = swarm_for_task.behaviour_mut().kademlia.bootstrap() {
                            warn!("Failed to bootstrap Kademlia: {e}");
                        }
                    },
                }
            }
        });

        Ok(handle)
    }

    /// Create network behavior
    fn create_behaviour(local_peer_id: PeerId) -> Result<ComposedBehaviour> {
        // Set up Gossipsub for publish/subscribe
        let gossipsub_config = gossipsub::Config::default();
        let gossipsub = Gossipsub::new(
            gossipsub::MessageAuthenticity::Signed(identity::Keypair::generate_ed25519()),
            gossipsub_config,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create Gossipsub: {}", e))?;

        // Set up Kademlia for peer discovery and DHT
        let store = MemoryStore::new(local_peer_id);
        let kademlia = kad::Behaviour::new(local_peer_id, store);

        // Set up ping for liveness checking
        let ping = PingBehaviour::new(ping::Config::new());

        Ok(ComposedBehaviour {
            gossipsub,
            kademlia,
            ping,
        })
    }

    /// Handle incoming pubsub messages with DoS protection
    async fn handle_pubsub_message(
        message: &gossipsub::Message,
        message_tx: &mpsc::Sender<NetworkMessage>,
        state: &Arc<RwLock<State>>,
        dos_protection: &DosProtection,
    ) -> Result<()> {
        // Check DoS protection
        if !dos_protection
            .check_message_rate(
                &message.source.unwrap_or(PeerId::random()),
                message.data.len(),
            )
            .await?
        {
            warn!(
                "Message from {:?} blocked by DoS protection",
                message.source
            );
            return Ok(());
        }

        // Deserialize message
        let network_message: NetworkMessage = serde_json::from_slice(&message.data)
            .context("Failed to deserialize network message")?;

        match &network_message {
            NetworkMessage::BlockProposal(block) => {
                info!("Received block proposal: {}", block.hash()?.to_hex());

                // Forward to consensus layer
                message_tx
                    .send(network_message)
                    .await
                    .context("Failed to forward block proposal")?;
            }
            NetworkMessage::BlockVote {
                block_hash,
                validator_id,
                ..
            } => {
                debug!(
                    "Received block vote from {}: {}",
                    validator_id,
                    hex::encode(block_hash.as_ref())
                );

                // Forward to consensus layer
                message_tx
                    .send(network_message)
                    .await
                    .context("Failed to forward block vote")?;
            }
            NetworkMessage::TransactionGossip(tx) => {
                debug!(
                    "Received transaction gossip: {}",
                    hex::encode(tx.hash().as_ref())
                );

                // Add to mempool
                let mut _state_guard = state.write().await;
                if let Err(e) = _state_guard.add_pending_transaction(tx.clone()) {
                    warn!("Failed to add transaction to mempool: {e}");
                }
            }
            NetworkMessage::BlockRequest {
                block_hash,
                requester,
            } => {
                debug!(
                    "Received block request from {}: {}",
                    requester,
                    hex::encode(block_hash.as_ref())
                );

                // Check if we have the block
                let state_guard = state.read().await;
                if let Some(block) = state_guard.get_block_by_hash(block_hash) {
                    // Send block response
                    let response = NetworkMessage::BlockResponse {
                        block: block.clone(),
                        responder: message.source.map(|s| s.to_string()).unwrap_or_default(),
                    };

                    message_tx
                        .send(response)
                        .await
                        .context("Failed to send block response")?;
                }
            }
            NetworkMessage::BlockResponse { block, responder } => {
                info!(
                    "Received block response from {}: {}",
                    responder,
                    block.hash()?
                );

                // Process the block
                message_tx
                    .send(network_message)
                    .await
                    .context("Failed to forward block response")?;
            }
            NetworkMessage::ShardAssignment {
                node_id,
                shard_id,
                timestamp,
            } => {
                info!(
                    "Received shard assignment: node {node_id} assigned to shard {shard_id} at timestamp {timestamp}"
                );

                // Forward to sharding layer
                message_tx
                    .send(network_message)
                    .await
                    .context("Failed to forward shard assignment")?;
            }
            NetworkMessage::CrossShardMessage {
                from_shard,
                to_shard,
                message_type,
                ..
            } => {
                debug!(
                    "Received cross-shard message from shard {from_shard} to {to_shard}: {message_type:?}"
                );

                // Forward to sharding layer
                message_tx
                    .send(network_message)
                    .await
                    .context("Failed to forward cross-shard message")?;
            }
        }

        Ok(())
    }

    /// Publish a message to the network with DoS protection
    async fn publish_message(
        swarm: &mut Swarm<ComposedBehaviour>,
        message: NetworkMessage,
        block_topic: &IdentTopic,
        tx_topic: &IdentTopic,
        vote_topic: &IdentTopic,
        cross_shard_topic: &IdentTopic,
        shard_id: u64,
        stats: &Arc<RwLock<NetworkStats>>,
        dos_protection: &DosProtection,
    ) -> Result<()> {
        // Serialize message
        let data = serde_json::to_vec(&message).context("Failed to serialize network message")?;

        // Check DoS protection for outgoing message
        if !dos_protection
            .check_message_rate(swarm.local_peer_id(), data.len())
            .await?
        {
            warn!("Outgoing message blocked by DoS protection");
            return Ok(());
        }

        // Choose topic based on message type
        let topic = match &message {
            NetworkMessage::BlockProposal(_) => Topic::from(block_topic.clone()),
            NetworkMessage::BlockVote { .. } => Topic::from(vote_topic.clone()),
            NetworkMessage::TransactionGossip(_) => Topic::from(tx_topic.clone()),
            NetworkMessage::CrossShardMessage { .. } => Topic::from(cross_shard_topic.clone()),
            NetworkMessage::BlockRequest { .. } => Topic::from(block_topic.clone()),
            NetworkMessage::BlockResponse { .. } => Topic::from(block_topic.clone()),
            NetworkMessage::ShardAssignment { .. } => block_topic.clone().into(),
        };

        // Publish to the network
        swarm.behaviour_mut().gossipsub.publish(topic, data.clone());

        // Update stats
        {
            let mut stats_guard = stats.write().await;
            stats_guard.messages_sent += 1;
            stats_guard.bytes_sent += data.len();
        }

        Ok(())
    }

    /// Get a message sender for this network
    pub fn get_message_sender(&self) -> mpsc::Sender<NetworkMessage> {
        self.message_tx.clone()
    }

    /// Get the local peer ID
    pub fn get_peer_id(&self) -> PeerId {
        self.peer_id
    }

    /// Get network statistics
    pub async fn get_stats(&self) -> NetworkStats {
        let stats_guard = self.stats.read().await;
        stats_guard.clone()
    }

    /// Calculate block priority based on various factors
    pub fn calculate_block_priority(&self, block: &Block) -> BlockPriority {
        if block.transactions.len() > 1000 {
            BlockPriority::High
        } else if block.transactions.len() > 100 {
            BlockPriority::Medium
        } else {
            BlockPriority::Low
        }
    }

    /// Enhanced block propagation with prioritization and compression
    #[allow(dead_code)]
    async fn propagate_block(
        &mut self,
        block: &Block,
        priority: BlockPriority,
        config: &BlockPropagationConfig,
    ) -> Result<()> {
        // Calculate block size by serializing it
        let block_data = serde_json::to_vec(block)?;
        let block_size = block_data.len();

        // Compress block if it exceeds threshold
        let (_, compressed_size) = if block_size > config.compression_threshold {
            let compressed = self.encode_all(&block_data, Compression::default())?;
            (compressed.clone(), Some(compressed.len()))
        } else {
            (block_data, None)
        };

        let meta = BlockPropagationMeta {
            block_hash: block.hash()?,
            priority,
            timestamp: Instant::now(),
            size: block_size,
            compressed_size,
            propagation_count: 0,
            last_propagation: None,
        };

        let mut queue_guard = self._block_propagation_queue.write().await;
        queue_guard.push(meta);

        // Publish to network via message channel
        let message = NetworkMessage::BlockProposal(block.clone());
        if let Err(e) = self.message_tx.send(message).await {
            warn!("Failed to send block proposal message: {}", e);
        }

        Ok(())
    }

    /// Helper function to compress data using zlib
    #[allow(dead_code)]
    fn encode_all(&self, data: &[u8], level: Compression) -> Result<Vec<u8>, NetworkError> {
        let mut encoder = ZlibEncoder::new(Vec::new(), level);
        encoder.write_all(data).map_err(NetworkError::IoError)?;
        encoder.finish().map_err(NetworkError::IoError)
    }

    /// Helper function to decompress data using zlib
    #[allow(dead_code)]
    fn decode_all(&self, data: &[u8]) -> Result<Vec<u8>, NetworkError> {
        let mut decoder = ZlibDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(NetworkError::IoError)?;
        Ok(decompressed)
    }

    /// Bandwidth-aware block propagation
    #[allow(dead_code)]
    async fn bandwidth_aware_propagation(&mut self, config: &BlockPropagationConfig) -> Result<()> {
        // First, pull out blocks to propagate under the queue lock
        let blocks_to_propagate = {
            let mut queue_guard = self._block_propagation_queue.write().await;
            let mut bandwidth_used = 0;
            let mut collected = Vec::<BlockPropagationMeta>::new();

            while let Some(meta) = queue_guard.pop() {
                if bandwidth_used >= config.bandwidth_limit {
                    break;
                }

                if let Some(last_prop) = meta.last_propagation {
                    if last_prop.elapsed() < config.propagation_timeout {
                        continue;
                    }
                }

                if meta.propagation_count >= config.max_propagation_count {
                    continue;
                }

                bandwidth_used += meta.size;
                collected.push(meta);
            }

            collected
        };

        // Now propagate each block via message channel
        for meta in blocks_to_propagate {
            let block_option = {
                let _state_guard = self.state.read().await;
                _state_guard
                    .get_block_by_hash(&meta.block_hash)
                    .map(|b| b.clone())
            };

            if let Some(block) = block_option {
                self.propagate_block(&block, meta.priority, config).await?;
            } else {
                warn!(
                    "Block with hash {} not found in state",
                    hex::encode(meta.block_hash.as_ref())
                );
            }
        }
        Ok(())
    }

    pub async fn get_block_transactions(
        &self,
        block_hash: &Hash,
    ) -> Result<Vec<Transaction>, NetworkError> {
        let state_guard = self.state.read().await;
        match state_guard.get_block_by_hash(block_hash) {
            Some(block) => {
                // Return empty list for simplified implementation
                Ok(vec![])
            }
            None => Err(NetworkError::BlockNotFound(block_hash.clone())),
        }
    }

    pub async fn add_block_transactions(
        &self,
        block_hash: Hash,
        transactions: Vec<Transaction>,
    ) -> Result<(), NetworkError> {
        let _state_guard = self.state.write().await;
        // Add transactions to the block (actual implementation omitted)
        info!(
            "Received {} transactions for block {}",
            transactions.len(),
            block_hash
        );
        Ok(())
    }

    // Helper function to convert between Hash types
    #[allow(dead_code)]
    fn types_hash_to_crypto_hash(hash: &crate::types::Hash) -> crate::utils::crypto::Hash {
        let bytes = hash.as_ref();
        let mut arr = [0u8; 32];
        let len = std::cmp::min(bytes.len(), 32);
        arr[..len].copy_from_slice(&bytes[..len]);
        crate::utils::crypto::Hash::new(arr)
    }

    pub async fn get_block_by_hash(&self, hash: &Hash) -> Option<Block> {
        let state = self.state.read().await;
        state.get_block_by_hash(hash).map(|block| block.clone())
    }

    /// Create a new P2P network with specific configuration
    pub fn new_with_config(bind_addr: SocketAddr) -> Result<Self> {
        // Implementation would create network with specific bind address
        // For now, create a default instance
        let (message_tx, message_rx) = mpsc::channel(100);
        let (shutdown_tx, _) = mpsc::channel(1);
        let keypair = identity::Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());

        Ok(Self {
            config: Config::default(),
            state: Arc::new(RwLock::new(State::new(&Config::default()).unwrap())),
            peer_id,
            message_rx,
            message_tx,
            shutdown_signal: shutdown_tx,
            stats: Arc::new(RwLock::new(NetworkStats::default())),
            shard_id: 0,
            peers: Arc::new(RwLock::new(HashMap::new())),
            known_peers: Arc::new(RwLock::new(HashSet::new())),
            running: Arc::new(RwLock::new(false)),
            _block_propagation_queue: Arc::new(RwLock::new(BlockPropagationQueue::new(1000))),
            block_topic: IdentTopic::new("blocks"),
            tx_topic: IdentTopic::new("transactions"),
            vote_topic: IdentTopic::new("votes"),
            cross_shard_topic: IdentTopic::new("cross-shard"),
            dos_protection: Arc::new(DosProtection::new(DosConfig::default())),
        })
    }

    /// Connect to a specific peer
    pub async fn connect_peer(&self, peer_addr: &SocketAddr) -> Result<()> {
        // Implementation would establish connection to peer
        // This is a placeholder
        info!("Connecting to peer at {}", peer_addr);
        Ok(())
    }

    /// Send a message to a specific address
    pub async fn send_message(&self, addr: &SocketAddr, message: Vec<u8>) -> Result<()> {
        // Implementation would send message to specific address
        // This is a placeholder
        debug!("Sending {} bytes to {}", message.len(), addr);
        Ok(())
    }

    /// Stop the P2P service
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping P2P service...");
        // Implementation details would go here
        Ok(())
    }

    /// Advanced peer discovery to eliminate bootstrap dependencies
    pub async fn start_advanced_peer_discovery(&self) -> Result<()> {
        info!("Starting advanced peer discovery mechanisms...");

        // Start discovery methods concurrently without spawning
        let (_, _, _, _, _) = tokio::join!(
            self.start_mdns_discovery(),
            self.start_dht_discovery(),
            self.start_dns_seed_discovery(),
            self.start_upnp_discovery(),
            self.start_peer_exchange_discovery(),
        );

        Ok(())
    }

    /// mDNS (Multicast DNS) discovery for local network peers
    async fn start_mdns_discovery(&self) -> Result<()> {
        info!("Starting mDNS peer discovery...");

        let mut interval = tokio::time::interval(Duration::from_secs(30));

        loop {
            interval.tick().await;

            // Broadcast service discovery
            let service_name = "_arthachain._tcp.local";
            let discovery_message = PeerDiscoveryMessage {
                node_id: self.get_node_id(),
                listen_addresses: self.get_listen_addresses().await,
                protocol_version: self.get_protocol_version(),
                services: self.get_supported_services(),
                timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            };

            // Broadcast to local network
            self.broadcast_mdns_discovery(service_name, &discovery_message)
                .await?;

            // Listen for other node announcements
            self.listen_for_mdns_peers().await?;
        }
    }

    /// DHT (Distributed Hash Table) based peer discovery
    async fn start_dht_discovery(&self) -> Result<()> {
        info!("Starting DHT peer discovery...");

        let mut interval = tokio::time::interval(Duration::from_secs(60));

        loop {
            interval.tick().await;

            // Query DHT for peers near our node ID
            let target_ids = self.generate_discovery_targets();

            for target_id in target_ids {
                match self.dht_find_peers_near(target_id.clone()).await {
                    Ok(peers) => {
                        for peer in peers {
                            self.attempt_peer_connection(peer).await?;
                        }
                    }
                    Err(e) => warn!("DHT discovery failed for target {}: {}", target_id, e),
                }
            }
        }
    }

    /// DNS seed discovery from multiple sources
    async fn start_dns_seed_discovery(&self) -> Result<()> {
        info!("Starting DNS seed discovery...");

        let dns_seeds = vec![
            "seed1.arthachain.io",
            "seed2.arthachain.io",
            "seed3.arthachain.io",
            "bootstrap.arthachain.network",
            "peers.arthachain.dev",
        ];

        for seed in dns_seeds {
            let seed = seed.to_string();
            tokio::spawn(async move {
                // Placeholder DNS seed query implementation
                info!("Querying DNS seed: {}", seed);
            });
        }

        Ok(())
    }

    /// UPnP discovery for NAT traversal and local peers
    async fn start_upnp_discovery(&self) -> Result<()> {
        info!("Starting UPnP peer discovery...");

        // Search for UPnP devices that might be ArthaChain nodes
        let upnp_search = UpnpPeerSearch {
            service_type: "urn:arthachain-org:service:blockchain:1",
            search_interval: Duration::from_secs(120),
        };

        tokio::spawn(async move {
            upnp_search.continuous_discovery().await;
        });

        Ok(())
    }

    /// Peer exchange discovery (learn peers from existing connections)
    async fn start_peer_exchange_discovery(&self) -> Result<()> {
        info!("Starting peer exchange discovery...");

        let mut interval = tokio::time::interval(Duration::from_secs(45));

        loop {
            interval.tick().await;

            // Get list of connected peers
            let connected_peers = self.get_connected_peers().await;

            for peer in connected_peers {
                // Request peer lists from each connected peer
                match self.request_peer_list_from(peer.clone()).await {
                    Ok(peer_list) => {
                        for discovered_peer in peer_list {
                            if !self.is_peer_known(&discovered_peer).await {
                                self.attempt_peer_connection(discovered_peer).await?;
                            }
                        }
                    }
                    Err(e) => warn!("Peer exchange failed with {}: {}", peer.id, e),
                }
            }
        }
    }

    /// Attempt connection to discovered peer with retry logic
    async fn attempt_peer_connection(&self, peer: DiscoveredPeer) -> Result<()> {
        let max_retries = 3;
        let mut retry_count = 0;

        while retry_count < max_retries {
            match self.connect_to_peer(&peer).await {
                Ok(_) => {
                    info!(
                        "Successfully connected to discovered peer: {}",
                        peer.address
                    );
                    return Ok(());
                }
                Err(e) => {
                    retry_count += 1;
                    warn!(
                        "Connection attempt {} failed for {}: {}",
                        retry_count, peer.address, e
                    );

                    if retry_count < max_retries {
                        // Exponential backoff
                        let delay = Duration::from_millis(1000 * (2_u64.pow(retry_count)));
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(anyhow!("Failed to connect after {} retries", max_retries))
    }

    /// Get node ID
    fn get_node_id(&self) -> String {
        format!("node_{}", self.peer_id)
    }

    /// Get listen addresses
    async fn get_listen_addresses(&self) -> Vec<String> {
        vec!["127.0.0.1:8080".to_string()]
    }

    /// Get protocol version
    fn get_protocol_version(&self) -> String {
        "arthachain/1.0".to_string()
    }

    /// Get supported services
    fn get_supported_services(&self) -> Vec<String> {
        vec!["blockchain".to_string(), "consensus".to_string()]
    }

    /// Broadcast mDNS discovery
    async fn broadcast_mdns_discovery(
        &self,
        _service_name: &str,
        _message: &PeerDiscoveryMessage,
    ) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }

    /// Listen for mDNS peers
    async fn listen_for_mdns_peers(&self) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }

    /// Generate discovery targets
    fn generate_discovery_targets(&self) -> Vec<String> {
        vec!["target1".to_string(), "target2".to_string()]
    }

    /// DHT find peers near target
    async fn dht_find_peers_near(&self, _target_id: String) -> Result<Vec<DiscoveredPeer>> {
        // Placeholder implementation
        Ok(vec![])
    }

    /// Query DNS seed
    async fn query_dns_seed(&self, _seed: String) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }

    /// Get connected peers
    async fn get_connected_peers(&self) -> Vec<DiscoveredPeer> {
        // Placeholder implementation
        vec![]
    }

    /// Request peer list from peer
    async fn request_peer_list_from(&self, _peer: DiscoveredPeer) -> Result<Vec<DiscoveredPeer>> {
        // Placeholder implementation
        Ok(vec![])
    }

    /// Check if peer is known
    async fn is_peer_known(&self, _peer: &DiscoveredPeer) -> bool {
        // Placeholder implementation
        false
    }

    /// Connect to peer
    async fn connect_to_peer(&self, _peer: &DiscoveredPeer) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::transaction::TransactionType;

    #[tokio::test]
    async fn test_network_message_serialization() {
        // Create a test transaction
        let mut tx = Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient".to_string(),
            100,
            1,
            10,
            1000,
            vec![],
        );
        tx.signature = vec![1, 2, 3];

        // Create network message
        let message = NetworkMessage::TransactionGossip(tx);

        // Serialize and deserialize
        let serialized = serde_json::to_vec(&message).unwrap();
        let deserialized: NetworkMessage = serde_json::from_slice(&serialized).unwrap();

        // Verify
        match deserialized {
            NetworkMessage::TransactionGossip(tx) => {
                assert_eq!(tx.sender, "sender");
                assert_eq!(tx.recipient, "recipient");
                assert_eq!(tx.amount, 100);
            }
            _ => panic!("Wrong message type after deserialization"),
        }
    }
}
