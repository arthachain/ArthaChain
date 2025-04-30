use std::net::SocketAddr;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::network::error::NetworkError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    // Basic info
    pub node_id: String,
    pub addr: SocketAddr,
    pub version: String,
    
    // Connection info
    pub connected_since: Instant,
    pub last_seen: Instant,
    pub ping_ms: Option<u64>,
    
    // Geographic info
    pub region: Option<String>,
    pub country: Option<String>,
    pub city: Option<String>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    
    // Performance metrics
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub failed_messages: u64,
    
    // Reputation
    pub reputation_score: f64,
    pub banned_until: Option<Instant>,
    pub warning_count: u32,
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
            rx
        )
    }
    
    pub async fn add_peer(&mut self, info: PeerInfo) -> Result<(), NetworkError> {
        // Check if peer is banned
        if let Some((_, until)) = self.banned_peers.iter()
            .find(|(id, _)| id == &info.node_id) {
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
        self.event_tx.send(PeerEvent::Connected(info)).await
            .map_err(|_| NetworkError::EventChannelClosed)?;
            
        Ok(())
    }
    
    pub async fn remove_peer(&mut self, node_id: &str) -> Result<(), NetworkError> {
        if let Some(index) = self.peers.iter().position(|p| p.node_id == node_id) {
            self.peers.remove(index);
            self.event_tx.send(PeerEvent::Disconnected(node_id.to_string())).await
                .map_err(|_| NetworkError::EventChannelClosed)?;
        }
        Ok(())
    }
    
    pub async fn update_reputation(&mut self, node_id: &str, delta: f64) -> Result<(), NetworkError> {
        if let Some(peer) = self.peers.iter_mut().find(|p| p.node_id == node_id) {
            peer.reputation_score = (peer.reputation_score + delta)
                .max(0.0)
                .min(100.0);
                
            // Check if peer should be banned
            if peer.reputation_score < self.config.ban_threshold {
                self.ban_peer(node_id, self.config.ban_duration).await?;
            }
            
            self.event_tx.send(PeerEvent::ReputationUpdated {
                node_id: node_id.to_string(),
                score: peer.reputation_score
            }).await.map_err(|_| NetworkError::EventChannelClosed)?;
        }
        Ok(())
    }
    
    pub async fn ban_peer(&mut self, node_id: &str, duration: Duration) -> Result<(), NetworkError> {
        let until = Instant::now() + duration;
        self.banned_peers.push((node_id.to_string(), until));
        
        // Remove peer if connected
        self.remove_peer(node_id).await?;
        
        self.event_tx.send(PeerEvent::Banned {
            node_id: node_id.to_string(),
            duration
        }).await.map_err(|_| NetworkError::EventChannelClosed)?;
        
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