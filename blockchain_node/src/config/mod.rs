use crate::ai_engine::data_chunking::ChunkingConfig;
use crate::ledger::state::ShardConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// pub mod network;
// pub mod node;
pub mod performance_monitoring;
// pub mod storage;

// Re-exports
// pub use self::network::NetworkConfig;
// pub use self::node::NodeConfig;
pub use self::performance_monitoring::PerformanceMonitoringConfig;
// pub use self::storage::StorageConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub data_dir: PathBuf,
    pub network: NetworkConfig,
    pub consensus: ConsensusConfig,
    pub storage: StorageConfig,
    pub api: ApiConfig,
    pub sharding: ShardingConfig,
    pub node_id: Option<String>,
    pub private_key_file: Option<PathBuf>,
    pub p2p_listen_addr: String,
    pub rpc_listen_addr: String,
    pub api_listen_addr: String,
    pub bootstrap_peers: Option<Vec<String>>,
    pub log_level: String,
    pub enable_metrics: bool,
    pub metrics_addr: String,
    pub svdb_url: Option<String>,
    pub enable_ai: bool,
    pub ai_model_dir: PathBuf,
    pub is_genesis: bool,
    pub enable_api: bool,
    pub enable_fuzz_testing: Option<bool>,
    pub genesis_path: PathBuf,
    /// Configuration for data chunking
    pub chunking_config: ChunkingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub p2p_port: u16,
    pub max_peers: u32,
    pub bootstrap_nodes: Vec<String>,
    pub network_name: String,
    pub bootnodes: Vec<String>,
    pub connection_timeout: u64,
    pub discovery_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    pub block_time: u64,
    pub max_block_size: u64,
    pub consensus_type: String,
    pub difficulty_adjustment_period: u64,
    pub reputation_decay_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub db_type: String,
    pub max_open_files: u32,
    pub db_path: String,
    pub svdb_url: String,
    pub size_threshold: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub enabled: bool,
    pub port: u16,
    pub host: String,
    pub address: String,
    pub cors_domains: Vec<String>,
    pub allow_origin: Vec<String>,
    pub max_request_body_size: usize,
    pub max_connections: u32,
    pub enable_websocket: bool,
    pub enable_graphql: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    pub enabled: bool,
    pub shard_count: u32,
    pub primary_shard: u32,
    pub shard_id: u64,
    pub cross_shard_timeout: u64,
    pub assignment_strategy: String,
    pub cross_shard_strategy: String,
}

impl ShardConfig for Config {
    fn get_shard_id(&self) -> u64 {
        self.sharding.shard_id
    }

    fn get_genesis_config(&self) -> Option<&Config> {
        if self.is_genesis {
            Some(self)
        } else {
            None
        }
    }

    fn is_sharding_enabled(&self) -> bool {
        self.sharding.enabled
    }

    fn get_shard_count(&self) -> u32 {
        self.sharding.shard_count
    }

    fn get_primary_shard(&self) -> u32 {
        self.sharding.primary_shard
    }
}

impl Config {
    pub fn new() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            network: NetworkConfig {
                p2p_port: 30303,
                max_peers: 50,
                bootstrap_nodes: vec![],
                network_name: "testnet".to_string(),
                bootnodes: vec![],
                connection_timeout: 30,
                discovery_enabled: true,
            },
            consensus: ConsensusConfig {
                block_time: 15,
                max_block_size: 5 * 1024 * 1024, // 5MB
                consensus_type: "svbft".to_string(),
                difficulty_adjustment_period: 2016,
                reputation_decay_rate: 0.05,
            },
            storage: StorageConfig {
                db_type: "rocksdb".to_string(),
                max_open_files: 512,
                db_path: "./data/db".to_string(),
                svdb_url: "http://localhost:3000".to_string(),
                size_threshold: 1024 * 1024, // 1MB
            },
            api: ApiConfig {
                enabled: true,
                port: 8545,
                host: "127.0.0.1".to_string(),
                address: "127.0.0.1".to_string(),
                cors_domains: vec!["*".to_string()],
                allow_origin: vec!["*".to_string()],
                max_request_body_size: 10 * 1024 * 1024, // 10MB
                max_connections: 100,
                enable_websocket: false,
                enable_graphql: false,
            },
            sharding: ShardingConfig {
                enabled: false,
                shard_count: 1,
                primary_shard: 0,
                shard_id: 0,
                cross_shard_timeout: 30,
                assignment_strategy: "static".to_string(),
                cross_shard_strategy: "atomic".to_string(),
            },
            node_id: None,
            private_key_file: None,
            p2p_listen_addr: "0.0.0.0:30303".to_string(),
            rpc_listen_addr: "0.0.0.0:8545".to_string(),
            api_listen_addr: "0.0.0.0:8080".to_string(),
            bootstrap_peers: None,
            log_level: "info".to_string(),
            enable_metrics: false,
            metrics_addr: "0.0.0.0:9100".to_string(),
            svdb_url: None,
            enable_ai: false,
            ai_model_dir: PathBuf::from("./models"),
            is_genesis: false,
            enable_api: true,
            enable_fuzz_testing: None,
            genesis_path: PathBuf::from("./genesis.json"),
            chunking_config: crate::ai_engine::data_chunking::ChunkingConfig::default(),
        }
    }

    /// Load configuration from a file
    pub fn from_file(_path: &str) -> Result<Self> {
        // Implementation omitted for brevity
        Ok(Self::new())
    }

    /// Save configuration to a file
    pub fn save_to_file(&self, _path: &str) -> Result<()> {
        // Implementation omitted for brevity
        Ok(())
    }

    /// Get or create a node identity
    pub fn get_or_create_node_identity(&self) -> (String, Vec<u8>) {
        // This is a placeholder implementation
        let node_id = self.node_id.clone().unwrap_or_else(|| "node1".to_string());
        let private_key = vec![1, 2, 3, 4]; // This would be a real key in production
        (node_id, private_key)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8545,
            host: "127.0.0.1".to_string(),
            address: "127.0.0.1".to_string(),
            cors_domains: vec!["*".to_string()],
            allow_origin: vec!["*".to_string()],
            max_request_body_size: 10 * 1024 * 1024, // 10MB
            max_connections: 100,
            enable_websocket: false,
            enable_graphql: false,
        }
    }
}
