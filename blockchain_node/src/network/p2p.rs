use anyhow::{Context, Result};
use libp2p::{
    core::{transport::Transport, upgrade},
    floodsub::{self, Floodsub, FloodsubEvent, Topic},
    futures::StreamExt,
    identity,
    kad::{self, store::MemoryStore, QueryResult},
    noise,
    ping::{self, Event as PingEvent},
    swarm::{NetworkBehaviour, Swarm, SwarmEvent},
    tcp, yamux, PeerId,
};
use log::{debug, info, warn};
use std::collections::HashSet;
use std::time::Duration;
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
            .then_with(|| self.block_hash.0.cmp(&other.block_hash.0))
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
    pub floodsub: Floodsub,
    pub kademlia: kad::Behaviour<MemoryStore>,
    pub ping: ping::Behaviour,
}

/// Generated event from the network behaviour
#[derive(Debug)]
pub enum ComposedEvent {
    Floodsub(FloodsubEvent),
    Kademlia(kad::Event),
    Ping(PingEvent),
}

impl From<FloodsubEvent> for ComposedEvent {
    fn from(event: FloodsubEvent) -> Self {
        ComposedEvent::Floodsub(event)
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
    block_topic: Topic,
    /// Transaction topic
    tx_topic: Topic,
    /// Vote topic
    vote_topic: Topic,
    /// Cross-shard topic
    cross_shard_topic: Topic,
    /// DoS protection
    dos_protection: Arc<DosProtection>,
    /// Network swarm
    swarm: Option<Swarm<ComposedBehaviour>>,
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

        info!("Local peer id: {}", peer_id);

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
            block_topic: Topic::new("blocks"),
            tx_topic: Topic::new("transactions"),
            vote_topic: Topic::new(format!("votes-shard-{}", shard_id)),
            cross_shard_topic: Topic::new("cross-shard"),
            dos_protection: Arc::new(dos_protection),
            swarm: None,
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
        let behaviour = Self::create_behaviour(peer_id.clone())?;

        // Build swarm
        let mut swarm = Swarm::new(
            transport,
            behaviour,
            peer_id,
            libp2p::swarm::Config::with_tokio_executor(),
        );

        // Subscribe to topics
        swarm
            .behaviour_mut()
            .floodsub
            .subscribe(block_topic.clone());
        swarm.behaviour_mut().floodsub.subscribe(tx_topic.clone());
        swarm.behaviour_mut().floodsub.subscribe(vote_topic.clone());
        swarm
            .behaviour_mut()
            .floodsub
            .subscribe(cross_shard_topic.clone());

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
                        warn!("Failed to dial bootstrap peer {}: {}", addr, e);
                    }
                }
                Err(e) => warn!("Failed to parse bootstrap peer address {}: {}", addr, e),
            }
        }

        // Store swarm in self for later use
        let mut swarm_for_task = swarm;
        self.swarm = None; // We'll set this after the task is created

        let handle = tokio::spawn(async move {
            info!("P2P network started");

            let mut discovery_timer = tokio::time::interval(Duration::from_secs(30));

            loop {
                tokio::select! {
                    // Process incoming network events
                    event = swarm_for_task.select_next_some() => {
                        match event {
                            SwarmEvent::Behaviour(ComposedEvent::Floodsub(FloodsubEvent::Message(message))) => {
                                {
                                    let mut stats_guard = stats.write().await;
                                    stats_guard.messages_received += 1;
                                    stats_guard.bytes_received += message.data.len();
                                }

                                if let Err(e) = Self::handle_pubsub_message(&message, &message_tx, &state, &dos_protection).await {
                                    warn!("Error handling pubsub message: {}", e);
                                }
                            },
                            SwarmEvent::Behaviour(ComposedEvent::Ping(ping_evt)) => {
                                // Use a simpler string representation for ping events
                                debug!("Received ping event: {:?}", ping_evt);
                            },
                            SwarmEvent::Behaviour(ComposedEvent::Kademlia(kad::Event::OutboundQueryProgressed { result, .. })) => {
                                match result {
                                    QueryResult::GetProviders(Ok(_)) => {
                                        // Just report that the query completed successfully
                                        debug!("GetProviders query completed successfully");
                                    },
                                    QueryResult::GetProviders(Err(err)) => {
                                        warn!("Failed to get providers: {}", err);
                                    },
                                    _ => {}
                                }
                            },
                            SwarmEvent::NewListenAddr { address, .. } => {
                                info!("Listening on {}", address);
                            },
                            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                                info!("Connected to {}", peer_id);

                                {
                                    let mut stats_guard = stats.write().await;
                                    stats_guard.known_peers.insert(peer_id.to_string());
                                    stats_guard.peer_count = stats_guard.known_peers.len();
                                }
                            },
                            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                                info!("Disconnected from {}: {:?}", peer_id, cause);

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
                            warn!("Error publishing message: {}", e);
                        }
                    },

                    // Periodically run Kademlia bootstrap to discover more peers
                    _ = discovery_timer.tick() => {
                        debug!("Running Kademlia bootstrap");
                        if let Err(e) = swarm_for_task.behaviour_mut().kademlia.bootstrap() {
                            warn!("Failed to bootstrap Kademlia: {}", e);
                        }
                    },
                }
            }
        });

        Ok(handle)
    }

    /// Create network behavior
    fn create_behaviour(local_peer_id: PeerId) -> Result<ComposedBehaviour> {
        // Set up Floodsub for publish/subscribe
        let floodsub = Floodsub::new(local_peer_id.clone());

        // Set up Kademlia for peer discovery and DHT
        let store = MemoryStore::new(local_peer_id.clone());
        let kademlia = kad::Behaviour::new(local_peer_id.clone(), store);

        // Set up ping for liveness checking
        let ping = ping::Behaviour::new(ping::Config::new());

        Ok(ComposedBehaviour {
            floodsub,
            kademlia,
            ping,
        })
    }

    /// Handle incoming pubsub messages with DoS protection
    async fn handle_pubsub_message(
        message: &floodsub::FloodsubMessage,
        message_tx: &mpsc::Sender<NetworkMessage>,
        state: &Arc<RwLock<State>>,
        dos_protection: &DosProtection,
    ) -> Result<()> {
        // Check DoS protection
        if !dos_protection
            .check_message_rate(&message.source, message.data.len())
            .await?
        {
            warn!("Message from {} blocked by DoS protection", message.source);
            return Ok(());
        }

        // Deserialize message
        let network_message: NetworkMessage = serde_json::from_slice(&message.data)
            .context("Failed to deserialize network message")?;

        match &network_message {
            NetworkMessage::BlockProposal(block) => {
                info!("Received block proposal: {}", block.hash());

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
                    hex::encode(block_hash.as_bytes())
                );

                // Forward to consensus layer
                message_tx
                    .send(network_message)
                    .await
                    .context("Failed to forward block vote")?;
            }
            NetworkMessage::TransactionGossip(tx) => {
                debug!("Received transaction gossip: {}", tx.hash());

                // Add to mempool
                let mut _state_guard = state.write().await;
                if let Err(e) = _state_guard.add_pending_transaction(tx.clone()) {
                    warn!("Failed to add transaction to mempool: {}", e);
                }
            }
            NetworkMessage::BlockRequest {
                block_hash,
                requester,
            } => {
                debug!(
                    "Received block request from {}: {}",
                    requester,
                    hex::encode(block_hash.as_bytes())
                );

                // Check if we have the block
                let state_guard = state.read().await;
                if let Some(block) = state_guard.get_block_by_hash(block_hash) {
                    // Send block response
                    let response = NetworkMessage::BlockResponse {
                        block: block.clone(),
                        responder: message.source.to_string(),
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
                    block.hash()
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
                    "Received shard assignment: node {} assigned to shard {} at timestamp {}",
                    node_id, shard_id, timestamp
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
                    "Received cross-shard message from shard {} to {}: {:?}",
                    from_shard, to_shard, message_type
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
        block_topic: &Topic,
        tx_topic: &Topic,
        vote_topic: &Topic,
        cross_shard_topic: &Topic,
        shard_id: u64,
        stats: &Arc<RwLock<NetworkStats>>,
        dos_protection: &DosProtection,
    ) -> Result<()> {
        // Serialize message
        let data = serde_json::to_vec(&message).context("Failed to serialize network message")?;

        // Check DoS protection for outgoing message
        if !dos_protection
            .check_message_rate(&swarm.local_peer_id(), data.len())
            .await?
        {
            warn!("Outgoing message blocked by DoS protection");
            return Ok(());
        }

        // Choose topic based on message type
        let topic = match &message {
            NetworkMessage::BlockProposal(_) => block_topic.clone(),
            NetworkMessage::BlockVote { .. } => vote_topic.clone(),
            NetworkMessage::TransactionGossip(_) => tx_topic.clone(),
            NetworkMessage::CrossShardMessage { .. } => cross_shard_topic.clone(),
            // For request/response, use the appropriate topic based on content
            NetworkMessage::BlockRequest { .. } => block_topic.clone(),
            NetworkMessage::BlockResponse { .. } => block_topic.clone(),
            NetworkMessage::ShardAssignment { .. } => Topic::new(format!("shard-{}", shard_id)),
        };

        // Publish to the network
        swarm.behaviour_mut().floodsub.publish(topic, data.clone());

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
        self.peer_id.clone()
    }

    /// Get network statistics
    pub async fn get_stats(&self) -> NetworkStats {
        let stats_guard = self.stats.read().await;
        stats_guard.clone()
    }

    /// Calculate block priority based on various factors
    pub fn calculate_block_priority(&self, block: &Block) -> BlockPriority {
        if block.body.transactions.len() > 1000 {
            BlockPriority::High
        } else if block.body.transactions.len() > 100 {
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
            block_hash: block.hash(),
            priority,
            timestamp: Instant::now(),
            size: block_size,
            compressed_size,
            propagation_count: 0,
            last_propagation: None,
        };

        let mut queue_guard = self._block_propagation_queue.write().await;
        queue_guard.push(meta);

        // Publish to network
        let message = NetworkMessage::BlockProposal(block.clone());
        if let Some(swarm) = &mut self.swarm {
            Self::publish_message(
                swarm,
                message,
                &self.block_topic,
                &self.tx_topic,
                &self.vote_topic,
                &self.cross_shard_topic,
                self.shard_id,
                &self.stats,
                &self.dos_protection,
            )
            .await?;
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

        // Now propagate each block
        if let Some(_swarm) = &mut self.swarm {
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
                        hex::encode(meta.block_hash.as_bytes())
                    );
                }
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
            Some(block) => Ok(block.body.transactions.clone()),
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
        let bytes = hash.as_bytes();
        let mut arr = [0u8; 32];
        let len = std::cmp::min(bytes.len(), 32);
        arr[..len].copy_from_slice(&bytes[..len]);
        crate::utils::crypto::Hash::new(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::transaction::TransactionType;

    #[tokio::test]
    async fn test_network_message_serialization() {
        // Create a test transaction
        let tx = Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient".to_string(),
            100,
            1,
            10,
            1000,
            vec![],
            vec![1, 2, 3],
        );

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
