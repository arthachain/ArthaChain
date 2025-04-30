use crate::config::{Config, NetworkConfig, ConsensusConfig, StorageConfig, ApiConfig, ShardingConfig};
use std::path::PathBuf;
use crate::ai_engine::data_chunking::ChunkingConfig;

pub fn mock_config() -> Config {
    Config {
        data_dir: PathBuf::from("/tmp/blockchain_test"),
        node_id: Some("test_node".to_string()),
        private_key_file: Some(PathBuf::from("test_key.pem")),
        p2p_listen_addr: "127.0.0.1:4001".to_string(),
        rpc_listen_addr: "127.0.0.1:4002".to_string(),
        api_listen_addr: "127.0.0.1:8080".to_string(),
        bootstrap_peers: Some(vec![]),
        log_level: "info".to_string(),
        enable_metrics: false,
        metrics_addr: "127.0.0.1:9090".to_string(),
        svdb_url: Some("http://localhost:8081".to_string()),
        enable_ai: false,
        ai_model_dir: PathBuf::from("/tmp/ai_models"),
        is_genesis: true,
        enable_api: true,
        enable_fuzz_testing: Some(false),
        genesis_path: PathBuf::from("/tmp/genesis.json"),
        network: NetworkConfig {
            p2p_port: 4001,
            max_peers: 50,
            bootstrap_nodes: vec![],
            network_name: "testnet".to_string(),
            bootnodes: vec![],
            connection_timeout: 30,
            discovery_enabled: true,
        },
        consensus: ConsensusConfig {
            block_time: 15,
            max_block_size: 1048576,
            consensus_type: "svbft".to_string(),
            difficulty_adjustment_period: 100,
            reputation_decay_rate: 0.1,
        },
        storage: StorageConfig {
            db_type: "rocksdb".to_string(),
            max_open_files: 512,
            db_path: "/tmp/blockchain_test/db".to_string(),
            svdb_url: "http://localhost:8081".to_string(),
            size_threshold: 1024,
        },
        api: ApiConfig {
            enabled: true,
            port: 8080,
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
        chunking_config: ChunkingConfig::default(),
    }
} 