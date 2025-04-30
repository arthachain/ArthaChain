use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    // Basic network settings
    pub listen_addr: SocketAddr,
    pub external_addr: Option<String>,
    pub bootstrap_nodes: Vec<String>,
    pub max_peers: usize,
    pub target_peers: usize,
    
    // Connection settings
    pub connection_timeout: Duration,
    pub handshake_timeout: Duration,
    pub ping_interval: Duration,
    pub ping_timeout: Duration,
    
    // Message settings
    pub max_message_size: usize,
    pub message_timeout: Duration,
    pub broadcast_fanout: usize,
    
    // Peer discovery settings
    pub discovery_enabled: bool,
    pub discovery_interval: Duration,
    pub discovery_limit: usize,
    pub discovery_peer_expiry: Duration,
    
    // Geographic diversity settings
    pub geo_diversity_enabled: bool,
    pub min_region_peers: usize,
    pub max_region_peers: usize,
    
    // Privacy settings
    pub enable_peer_exchange: bool,
    pub enable_nat_traversal: bool,
    pub enable_upnp: bool,
    pub enable_relay: bool,
    
    // Performance settings
    pub tcp_nodelay: bool,
    pub tcp_keepalive: Option<Duration>,
    pub outbound_buffer_size: usize,
    pub inbound_buffer_size: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        NetworkConfig {
            listen_addr: "127.0.0.1:8000".parse().unwrap(),
            external_addr: None,
            bootstrap_nodes: Vec::new(),
            max_peers: 50,
            target_peers: 25,
            
            connection_timeout: Duration::from_secs(10),
            handshake_timeout: Duration::from_secs(5),
            ping_interval: Duration::from_secs(30),
            ping_timeout: Duration::from_secs(5),
            
            max_message_size: 4 * 1024 * 1024, // 4MB
            message_timeout: Duration::from_secs(30),
            broadcast_fanout: 4,
            
            discovery_enabled: true,
            discovery_interval: Duration::from_secs(60),
            discovery_limit: 1000,
            discovery_peer_expiry: Duration::from_secs(24 * 60 * 60), // 24 hours
            
            geo_diversity_enabled: true,
            min_region_peers: 2,
            max_region_peers: 10,
            
            enable_peer_exchange: true,
            enable_nat_traversal: true,
            enable_upnp: true,
            enable_relay: true,
            
            tcp_nodelay: true,
            tcp_keepalive: Some(Duration::from_secs(60)),
            outbound_buffer_size: 8 * 1024 * 1024, // 8MB
            inbound_buffer_size: 8 * 1024 * 1024, // 8MB
        }
    }
}

impl NetworkConfig {
    pub fn new() -> Self {
        NetworkConfig::default()
    }
    
    pub fn with_listen_addr(mut self, addr: SocketAddr) -> Self {
        self.listen_addr = addr;
        self
    }
    
    pub fn with_external_addr(mut self, addr: String) -> Self {
        self.external_addr = Some(addr);
        self
    }
    
    pub fn with_bootstrap_nodes(mut self, nodes: Vec<String>) -> Self {
        self.bootstrap_nodes = nodes;
        self
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.max_peers < self.target_peers {
            return Err("max_peers must be greater than or equal to target_peers".to_string());
        }
        
        if self.min_region_peers > self.max_region_peers {
            return Err("min_region_peers must be less than or equal to max_region_peers".to_string());
        }
        
        if self.max_message_size == 0 {
            return Err("max_message_size must be greater than 0".to_string());
        }
        
        Ok(())
    }
} 