use crate::ai_engine::data_chunking::ChunkingConfig;
use crate::ledger::state::ShardConfig;
use anyhow::Result;
use log::{debug, info, warn};
use serde::{Deserialize, Deserializer, Serialize};
use serde::de::{self, Visitor};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

// Custom deserializer for Duration from u64 seconds
fn deserialize_duration_from_u64<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: Deserializer<'de>,
{
    let seconds = u64::deserialize(deserializer)?;
    Ok(Duration::from_secs(seconds))
}

// pub mod network;
// pub mod node;
pub mod performance_monitoring;
// pub mod storage;

/// Node configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Node identity and basic settings
    pub node_id: String,
    pub network_id: String,
    pub data_dir: PathBuf,

    /// Network configuration
    pub listen_address: String,
    pub port: u16,
    pub bootstrap_peers: Vec<String>,

    /// Consensus configuration  
    pub validator_key: Option<String>,
    pub is_validator: bool,

    /// AI engine settings
    pub ai_enabled: bool,
    pub model_path: Option<PathBuf>,

    /// Storage settings
    pub storage_path: PathBuf,
    pub max_storage_size: u64,

    /// Performance settings
    pub max_connections: u32,
    #[serde(deserialize_with = "deserialize_duration_from_u64")]
    pub sync_timeout: Duration,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            network_id: "arthachain-mainnet".to_string(),
            data_dir: PathBuf::from("./data"),
            listen_address: "0.0.0.0".to_string(),
            port: 30303,
            bootstrap_peers: vec![],
            validator_key: None,
            is_validator: false,
            ai_enabled: true,
            model_path: None,
            storage_path: PathBuf::from("./data/storage"),
            max_storage_size: 100 * 1024 * 1024 * 1024, // 100 GB
            max_connections: 50,
            sync_timeout: Duration::from_secs(30),
        }
    }
}

impl NodeConfig {
    /// Load configuration from file
    pub async fn load_from_file(path: &str) -> Result<Self> {
        let contents = tokio::fs::read_to_string(path).await?;
        
        // Try to parse as flat NodeConfig first
        if let Ok(config) = toml::from_str::<NodeConfig>(&contents) {
            return Ok(config);
        }
        
        // If that fails, try to parse as nested Config and convert
        if let Ok(nested_config) = toml::from_str::<Config>(&contents) {
            return Ok(Self::from_nested_config(nested_config));
        }
        
        // If both fail, return the original error
        let config: NodeConfig = toml::from_str(&contents)?;
        Ok(config)
    }
    
    /// Convert from nested Config structure
    pub fn from_nested_config(config: Config) -> Self {
        Self {
            node_id: config.node_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            network_id: config.network.network_name,
            data_dir: config.data_dir,
            listen_address: "0.0.0.0".to_string(),
            port: config.network.p2p_port,
            bootstrap_peers: config.bootstrap_peers.unwrap_or_default(),
            validator_key: None,
            is_validator: false,
            ai_enabled: config.enable_ai,
            model_path: Some(config.ai_model_dir),
            storage_path: PathBuf::from(config.storage.db_path),
            max_storage_size: 100 * 1024 * 1024 * 1024, // 100 GB
            max_connections: config.network.max_peers,
            sync_timeout: Duration::from_secs(30),
        }
    }

    /// Save configuration to file  
    pub async fn save_to_file(&self, path: &str) -> Result<()> {
        let contents = toml::to_string_pretty(self)?;
        tokio::fs::write(path, contents).await?;
        Ok(())
    }

    /// Get or create node identity
    pub fn get_or_create_node_identity(&self) -> (String, Vec<u8>) {
        // Generate a simple node identity
        let node_id = self.node_id.clone();
        let private_key = vec![1u8; 32]; // Placeholder private key
        (node_id, private_key)
    }
}

// Re-exports
// pub use self::network::NetworkConfig;
// Re-export NodeConfig
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

/// Distributed Configuration Manager with consensus-based replication
pub struct DistributedConfigManager {
    /// Local configuration cache
    local_config: Arc<RwLock<Config>>,
    /// Configuration replicas across nodes
    config_replicas: Arc<RwLock<HashMap<String, ConfigReplica>>>,
    /// Configuration consensus tracker
    consensus_tracker: Arc<RwLock<ConfigConsensus>>,
    /// Configuration change history
    change_history: Arc<RwLock<Vec<ConfigChange>>>,
    /// Replication settings
    replication_config: ConfigReplicationSettings,
}

/// Configuration replica information
#[derive(Debug, Clone)]
pub struct ConfigReplica {
    pub node_id: String,
    pub last_sync: Instant,
    pub config_hash: String,
    pub is_online: bool,
    pub sync_attempts: u32,
}

/// Configuration consensus tracking
#[derive(Debug, Clone)]
pub struct ConfigConsensus {
    pub pending_changes: HashMap<String, ConfigChangeProposal>,
    pub votes: HashMap<String, HashMap<String, bool>>, // change_id -> node_id -> vote
    pub consensus_threshold: f64,                      // Percentage needed for approval
    pub active_nodes: HashSet<String>,
}

/// Configuration change tracking
#[derive(Debug, Clone)]
pub struct ConfigChange {
    pub id: String,
    pub timestamp: Instant,
    pub change_type: ConfigChangeType,
    pub old_value: Option<String>,
    pub new_value: String,
    pub proposer: String,
    pub approved_by: Vec<String>,
}

/// Types of configuration changes
#[derive(Debug, Clone)]
pub enum ConfigChangeType {
    NetworkSettings,
    ConsensusParameters,
    SecuritySettings,
    PerformanceSettings,
    NodeSettings,
    EmergencySettings,
}

/// Configuration change proposal
#[derive(Debug, Clone)]
pub struct ConfigChangeProposal {
    pub id: String,
    pub proposed_at: Instant,
    pub proposer: String,
    pub change_type: ConfigChangeType,
    pub description: String,
    pub new_config: String,
    pub urgency: ConfigUrgency,
    pub votes_for: HashSet<String>,
    pub votes_against: HashSet<String>,
}

/// Configuration change urgency levels
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigUrgency {
    Low,
    Medium,
    High,
    Emergency,
}

/// Configuration replication settings
#[derive(Debug, Clone)]
pub struct ConfigReplicationSettings {
    pub replication_factor: usize,
    pub sync_interval: Duration,
    pub consensus_timeout: Duration,
    pub max_sync_retries: u32,
    pub emergency_bypass_threshold: f64,
}

impl DistributedConfigManager {
    /// Create new distributed config manager
    pub fn new(config: Config, replication_config: ConfigReplicationSettings) -> Self {
        Self {
            local_config: Arc::new(RwLock::new(config)),
            config_replicas: Arc::new(RwLock::new(HashMap::new())),
            consensus_tracker: Arc::new(RwLock::new(ConfigConsensus {
                pending_changes: HashMap::new(),
                votes: HashMap::new(),
                consensus_threshold: 0.67, // 67% consensus needed
                active_nodes: HashSet::new(),
            })),
            change_history: Arc::new(RwLock::new(Vec::new())),
            replication_config,
        }
    }

    /// Initialize distributed configuration replication
    pub async fn initialize_distributed_config(&self) -> Result<()> {
        info!("Initializing distributed configuration management...");

        // Start configuration replication
        self.start_config_replication().await?;

        // Start consensus monitoring
        self.start_consensus_monitoring().await?;

        // Start configuration sync
        self.start_config_sync().await?;

        // Setup emergency configuration fallbacks
        self.setup_emergency_fallbacks().await?;

        info!("Distributed configuration management initialized");
        Ok(())
    }

    /// Start configuration replication across nodes
    async fn start_config_replication(&self) -> Result<()> {
        info!("Starting configuration replication...");

        // Discover peer nodes for config replication
        let peer_nodes = self.discover_config_peers().await?;

        // Initialize replicas
        let mut replicas = self.config_replicas.write().await;
        for node_id in peer_nodes {
            replicas.insert(
                node_id.clone(),
                ConfigReplica {
                    node_id: node_id.clone(),
                    last_sync: Instant::now(),
                    config_hash: String::new(),
                    is_online: false,
                    sync_attempts: 0,
                },
            );
        }

        // Start replication background task
        let replicas_clone = self.config_replicas.clone();
        let local_config = self.local_config.clone();
        let sync_interval = self.replication_config.sync_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(sync_interval);

            loop {
                interval.tick().await;

                // Sync configuration with all replicas
                Self::sync_with_replicas(&replicas_clone, &local_config).await;
            }
        });

        Ok(())
    }

    /// Discover peer nodes for configuration replication
    async fn discover_config_peers(&self) -> Result<Vec<String>> {
        // This would implement actual peer discovery logic
        // For now, return example nodes
        Ok(vec![
            "node-1".to_string(),
            "node-2".to_string(),
            "node-3".to_string(),
        ])
    }

    /// Sync configuration with replicas
    async fn sync_with_replicas(
        replicas: &Arc<RwLock<HashMap<String, ConfigReplica>>>,
        local_config: &Arc<RwLock<Config>>,
    ) {
        let config = local_config.read().await;
        let config_hash = Self::calculate_config_hash(&*config);

        let mut replicas_map = replicas.write().await;

        for replica in replicas_map.values_mut() {
            // Check if replica needs sync
            if replica.config_hash != config_hash {
                match Self::sync_replica(replica, &config_hash).await {
                    Ok(_) => {
                        replica.last_sync = Instant::now();
                        replica.is_online = true;
                        replica.sync_attempts = 0;
                        info!("Successfully synced config with {}", replica.node_id);
                    }
                    Err(e) => {
                        replica.sync_attempts += 1;
                        replica.is_online = false;
                        warn!("Failed to sync config with {}: {}", replica.node_id, e);
                    }
                }
            }
        }
    }

    /// Sync with individual replica
    async fn sync_replica(_replica: &mut ConfigReplica, _config_hash: &str) -> Result<()> {
        // This would implement actual config sync logic
        Ok(())
    }

    /// Calculate configuration hash
    fn calculate_config_hash(_config: &Config) -> String {
        // This would implement actual hash calculation
        "example_hash".to_string()
    }

    /// Start consensus monitoring for configuration changes
    async fn start_consensus_monitoring(&self) -> Result<()> {
        info!("Starting configuration consensus monitoring...");

        let consensus_tracker = self.consensus_tracker.clone();
        let change_history = self.change_history.clone();
        let consensus_timeout = self.replication_config.consensus_timeout;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                // Check for consensus on pending changes
                Self::process_consensus_votes(
                    &consensus_tracker,
                    &change_history,
                    consensus_timeout,
                )
                .await;

                // Clean up expired proposals
                Self::cleanup_expired_proposals(&consensus_tracker, consensus_timeout).await;
            }
        });

        Ok(())
    }

    /// Process consensus votes for configuration changes
    async fn process_consensus_votes(
        consensus_tracker: &Arc<RwLock<ConfigConsensus>>,
        change_history: &Arc<RwLock<Vec<ConfigChange>>>,
        consensus_timeout: Duration,
    ) {
        let mut consensus = consensus_tracker.write().await;
        let mut approved_changes = Vec::new();

        // Check each pending change for consensus
        for (change_id, proposal) in &consensus.pending_changes {
            let total_nodes = consensus.active_nodes.len();
            let votes_for = proposal.votes_for.len();
            let votes_against = proposal.votes_against.len();
            let total_votes = votes_for + votes_against;

            // Check if consensus threshold is met
            if total_votes >= total_nodes {
                let approval_rate = votes_for as f64 / total_votes as f64;

                if approval_rate >= consensus.consensus_threshold {
                    approved_changes.push(change_id.clone());
                    info!(
                        "Configuration change {} approved with {:.1}% consensus",
                        change_id,
                        approval_rate * 100.0
                    );
                } else {
                    info!(
                        "Configuration change {} rejected with {:.1}% approval",
                        change_id,
                        approval_rate * 100.0
                    );
                }
            }

            // Check for timeout
            if proposal.proposed_at.elapsed() > consensus_timeout {
                match proposal.urgency {
                    ConfigUrgency::Emergency => {
                        // Emergency changes can bypass consensus after timeout
                        approved_changes.push(change_id.clone());
                        warn!(
                            "Emergency configuration change {} auto-approved due to timeout",
                            change_id
                        );
                    }
                    _ => {
                        info!("Configuration change {} expired due to timeout", change_id);
                    }
                }
            }
        }

        // Apply approved changes
        for change_id in approved_changes {
            if let Some(proposal) = consensus.pending_changes.remove(&change_id) {
                let change = ConfigChange {
                    id: change_id,
                    timestamp: Instant::now(),
                    change_type: proposal.change_type,
                    old_value: None,
                    new_value: proposal.new_config,
                    proposer: proposal.proposer,
                    approved_by: proposal.votes_for.into_iter().collect(),
                };

                change_history.write().await.push(change);
            }
        }
    }

    /// Clean up expired configuration proposals
    async fn cleanup_expired_proposals(
        consensus_tracker: &Arc<RwLock<ConfigConsensus>>,
        consensus_timeout: Duration,
    ) {
        let mut consensus = consensus_tracker.write().await;

        consensus.pending_changes.retain(|change_id, proposal| {
            let expired = proposal.proposed_at.elapsed() > consensus_timeout;
            if expired && proposal.urgency != ConfigUrgency::Emergency {
                info!("Removing expired configuration proposal: {}", change_id);
                false
            } else {
                true
            }
        });
    }

    /// Start configuration synchronization
    async fn start_config_sync(&self) -> Result<()> {
        info!("Starting configuration synchronization...");

        // Sync with peer nodes every interval
        let local_config = self.local_config.clone();
        let config_replicas = self.config_replicas.clone();
        let sync_interval = self.replication_config.sync_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(sync_interval);

            loop {
                interval.tick().await;

                // Pull configuration updates from other nodes
                Self::pull_config_updates(&local_config, &config_replicas).await;

                // Push local changes to other nodes
                Self::push_config_updates(&local_config, &config_replicas).await;
            }
        });

        Ok(())
    }

    /// Pull configuration updates from other nodes
    async fn pull_config_updates(
        _local_config: &Arc<RwLock<Config>>,
        _config_replicas: &Arc<RwLock<HashMap<String, ConfigReplica>>>,
    ) {
        // This would implement actual config pulling logic
        debug!("Pulling configuration updates from peer nodes");
    }

    /// Push configuration updates to other nodes
    async fn push_config_updates(
        _local_config: &Arc<RwLock<Config>>,
        _config_replicas: &Arc<RwLock<HashMap<String, ConfigReplica>>>,
    ) {
        // This would implement actual config pushing logic
        debug!("Pushing configuration updates to peer nodes");
    }

    /// Setup emergency configuration fallbacks
    async fn setup_emergency_fallbacks(&self) -> Result<()> {
        info!("Setting up emergency configuration fallbacks...");

        // Create emergency configuration presets
        let emergency_configs = vec![
            (
                "network_partition",
                self.create_network_partition_config().await?,
            ),
            ("high_load", self.create_high_load_config().await?),
            (
                "security_breach",
                self.create_security_breach_config().await?,
            ),
            (
                "consensus_failure",
                self.create_consensus_failure_config().await?,
            ),
        ];

        // Store emergency configurations
        for (scenario, config) in emergency_configs {
            self.store_emergency_config(scenario, config).await?;
        }

        info!("Emergency configuration fallbacks configured");
        Ok(())
    }

    /// Create network partition emergency config
    async fn create_network_partition_config(&self) -> Result<Config> {
        let mut config = self.local_config.read().await.clone();

        // Adjust settings for network partition scenario
        config.network.max_peers = 5; // Reduce peer connections
                                      // Add other network partition specific settings

        Ok(config)
    }

    /// Create high load emergency config
    async fn create_high_load_config(&self) -> Result<Config> {
        let mut config = self.local_config.read().await.clone();

        // Adjust settings for high load scenario
        config.network.max_peers = 20; // Increase peer connections for load distribution
                                       // Add other high load specific settings

        Ok(config)
    }

    /// Create security breach emergency config
    async fn create_security_breach_config(&self) -> Result<Config> {
        let mut config = self.local_config.read().await.clone();

        // Adjust settings for security breach scenario
        config.network.max_peers = 3; // Limit connections
                                      // Add other security focused settings

        Ok(config)
    }

    /// Create consensus failure emergency config
    async fn create_consensus_failure_config(&self) -> Result<Config> {
        let mut config = self.local_config.read().await.clone();

        // Adjust settings for consensus failure scenario
        config.network.max_peers = 10;
        // Add other consensus recovery settings

        Ok(config)
    }

    /// Store emergency configuration
    async fn store_emergency_config(&self, _scenario: &str, _config: Config) -> Result<()> {
        // This would implement actual emergency config storage
        Ok(())
    }

    /// Propose configuration change
    pub async fn propose_config_change(
        &self,
        change_type: ConfigChangeType,
        new_config: String,
        description: String,
        urgency: ConfigUrgency,
        proposer: String,
    ) -> Result<String> {
        let change_id = format!("config_change_{}", Instant::now().elapsed().as_millis());

        let proposal = ConfigChangeProposal {
            id: change_id.clone(),
            proposed_at: Instant::now(),
            proposer,
            change_type,
            description,
            new_config,
            urgency,
            votes_for: HashSet::new(),
            votes_against: HashSet::new(),
        };

        // Add to pending changes
        self.consensus_tracker
            .write()
            .await
            .pending_changes
            .insert(change_id.clone(), proposal);

        info!("Configuration change proposed: {}", change_id);
        Ok(change_id)
    }

    /// Vote on configuration change
    pub async fn vote_on_change(
        &self,
        change_id: &str,
        node_id: String,
        approve: bool,
    ) -> Result<()> {
        let mut consensus = self.consensus_tracker.write().await;

        if let Some(proposal) = consensus.pending_changes.get_mut(change_id) {
            if approve {
                proposal.votes_for.insert(node_id);
            } else {
                proposal.votes_against.insert(node_id);
            }

            info!(
                "Vote recorded for change {}: {}",
                change_id,
                if approve { "FOR" } else { "AGAINST" }
            );
        }

        Ok(())
    }

    /// Get current configuration status
    pub async fn get_config_status(&self) -> ConfigStatus {
        let consensus = self.consensus_tracker.read().await;
        let replicas = self.config_replicas.read().await;

        ConfigStatus {
            pending_changes: consensus.pending_changes.len(),
            active_replicas: replicas.values().filter(|r| r.is_online).count(),
            total_replicas: replicas.len(),
            last_sync: replicas
                .values()
                .map(|r| r.last_sync)
                .max()
                .unwrap_or_else(Instant::now),
        }
    }
}

/// Configuration status information
#[derive(Debug)]
pub struct ConfigStatus {
    pub pending_changes: usize,
    pub active_replicas: usize,
    pub total_replicas: usize,
    pub last_sync: Instant,
}
