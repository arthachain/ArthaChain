use crate::ai_engine::explainability::AIExplainer;
use crate::ai_engine::security::SecurityAI;
use crate::api::metrics::MetricsService;
use crate::api::ApiServer;
use crate::config::Config;
#[cfg(not(skip_problematic_modules))]
use crate::consensus::sharding::ObjectiveSharding;
#[cfg(not(skip_problematic_modules))]
use crate::consensus::svbft::SVBFTConsensus;
use crate::consensus::svcp::SVCPMiner;
use crate::identity::IdentityManager;
use crate::ledger::state::State;
use crate::network::p2p::P2PNetwork;
use crate::network::rpc::RPCServer;
use crate::storage::RocksDbStorage;
use crate::storage::Storage;
use crate::types::Hash;
use crate::utils::fuzz::ContractFuzzer;
use crate::utils::security_audit::SecurityAuditRegistry;
use crate::utils::security_logger::SecurityLogger;
use anyhow::Context;
use log::{debug, info};
use parking_lot::RwLock as PLRwLock;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::sync::Mutex;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

#[cfg(feature = "evm")]
use crate::evm::{EvmConfig, EvmExecutor, EvmRpcService};
#[cfg(feature = "evm")]
use std::net::SocketAddr;

#[cfg(feature = "wasm")]
use crate::wasm::{WasmExecutor, WasmRpcService};

// Forward declaration for circular references
mod metrics {
    pub struct _MetricsService;
}

/// Node represents a running instance of a SocialChain blockchain node
pub struct Node {
    pub config: Arc<RwLock<Config>>,
    pub network: Arc<PLRwLock<Option<P2PNetwork>>>,
    pub api_server: Arc<PLRwLock<Option<ApiServer>>>,
    pub metrics: Arc<PLRwLock<Option<MetricsService>>>,
    pub state: Arc<State>,
    pub storage: Arc<PLRwLock<Box<dyn Storage + Send + Sync>>>,
    pub p2p_network: Option<P2PNetwork>,
    pub rpc_server: Option<RPCServer>,
    pub svcp_miner: Option<SVCPMiner>,
    #[cfg(not(skip_problematic_modules))]
    pub svbft_consensus: Option<SVBFTConsensus>,
    #[allow(dead_code)]
    #[cfg(not(skip_problematic_modules))]
    objective_sharding: Option<ObjectiveSharding>,
    pub security_ai: Option<SecurityAI>,
    pub shutdown_signal: broadcast::Sender<()>,
    pub task_handles: Vec<JoinHandle<()>>,
    /// Metrics service (if enabled)
    #[allow(dead_code)]
    metrics_service: Option<Arc<MetricsService>>,
    /// Identity manager for the node
    pub identity_manager: Option<Arc<IdentityManager>>,
    /// Node ID
    pub node_id: String,
    /// Node private key
    pub private_key: Vec<u8>,
    /// EVM executor (if enabled)
    #[cfg(feature = "evm")]
    evm_executor: Option<Arc<EvmExecutor>>,
    /// EVM RPC service (if enabled)
    #[cfg(feature = "evm")]
    evm_rpc: Option<EvmRpcService>,
    /// WASM Executor (when WASM support is enabled)
    #[cfg(feature = "wasm")]
    wasm_executor: Option<Arc<RwLock<WasmExecutor>>>,
    /// Security logger
    pub security_logger: Option<Arc<SecurityLogger>>,
    /// AI explainer for score transparency
    pub ai_explainer: Option<Arc<AIExplainer>>,
    /// Security audit registry
    pub security_audit: Option<Arc<SecurityAuditRegistry>>,
    /// Smart contract fuzzer
    pub contract_fuzzer: Option<ContractFuzzer>,
    peers: Arc<RwLock<Vec<String>>>,
    transactions: Arc<Mutex<Vec<Hash>>>,
}

impl Node {
    /// Create a new blockchain node
    pub async fn new(config: Config) -> Result<Self, anyhow::Error> {
        let state = State::new(&config)?;
        let db_path = Path::new("data/rocksdb");
        std::fs::create_dir_all(db_path)?;

        // Create RocksDbStorage
        let storage = RocksDbStorage::new();

        // Get or create node identity
        let (node_id, private_key) = config.get_or_create_node_identity();

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            network: Arc::new(PLRwLock::new(None)),
            api_server: Arc::new(PLRwLock::new(None)),
            metrics: Arc::new(PLRwLock::new(None)),
            state: Arc::new(state),
            storage: Arc::new(PLRwLock::new(Box::new(storage))),
            p2p_network: None,
            rpc_server: None,
            svcp_miner: None,
            #[cfg(not(skip_problematic_modules))]
            svbft_consensus: None,
            #[cfg(not(skip_problematic_modules))]
            objective_sharding: None,
            security_ai: None,
            shutdown_signal: broadcast::channel(1).0,
            task_handles: Vec::new(),
            metrics_service: None,
            identity_manager: None,
            node_id,
            private_key,
            #[cfg(feature = "evm")]
            evm_executor: None,
            #[cfg(feature = "evm")]
            evm_rpc: None,
            #[cfg(feature = "wasm")]
            wasm_executor: None,
            security_logger: None,
            ai_explainer: None,
            security_audit: None,
            contract_fuzzer: None,
            peers: Arc::new(RwLock::new(Vec::new())),
            transactions: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Initialize storage
    pub async fn init_storage(&self, path: &str) -> Result<(), anyhow::Error> {
        // We need to get a mutable reference to the storage inside the PL lock
        let mut storage_guard = self.storage.write();
        let _storage_ref = &mut **storage_guard;

        // TODO: Add StorageInit trait implementation
        // For now we'll just log this
        info!("Storage initialization requested for path: {}", path);

        Ok(())
    }

    /// Get the estimated transactions per second
    pub async fn get_estimated_tps(&self) -> Result<f32, anyhow::Error> {
        let tps = self.transactions.lock().await.len() as f32;
        Ok(tps)
    }

    /// Get the list of active peers
    pub async fn get_active_peers(&self) -> Result<Vec<String>, anyhow::Error> {
        let peers = self.peers.read().await.clone();
        Ok(peers)
    }

    /// Get the current memory usage in bytes
    pub async fn get_memory_usage(&self) -> Result<f64, anyhow::Error> {
        // Simple placeholder implementation
        Ok(0.0)
    }

    /// Get the current CPU usage as a percentage
    pub async fn get_cpu_usage(&self) -> Result<f64, anyhow::Error> {
        // Simple placeholder implementation
        Ok(0.0)
    }

    /// Initialize the identity manager
    #[allow(dead_code)]
    async fn init_identity_manager(&mut self) -> Result<(), anyhow::Error> {
        let (node_id, private_key) = self.config.read().await.get_or_create_node_identity();

        let identity_manager = IdentityManager::new(&node_id, private_key)
            .context("Failed to initialize Identity Manager")?;

        self.identity_manager = Some(Arc::new(identity_manager));

        info!("Identity Manager initialized successfully");
        Ok(())
    }

    /// Get the latest block hash
    pub async fn get_latest_block_hash(&self) -> crate::types::Hash {
        // Convert the string hash to a Hash type
        match self.state.get_latest_block_hash() {
            Ok(hash_str) => crate::types::Hash::from_hex(&hash_str).unwrap_or_default(),
            Err(_) => crate::types::Hash::default(),
        }
    }

    /// Get the current blockchain height
    pub async fn get_height(&self) -> u64 {
        self.state.get_height().unwrap_or(0)
    }

    pub async fn get_metrics(&self) -> Result<serde_json::Value, anyhow::Error> {
        let height = self.get_height().await;
        let state = &*self.state;
        let metrics = serde_json::json!({
            "blockchain": {
                "height": height,
                "difficulty": state.get_difficulty(),
                "total_transactions": state.get_total_transactions(),
            },
            "network": {
                "peers": self.get_active_peers().await?,
                "bandwidth": self.get_bandwidth_usage().await?,
            },
            "storage": {
                "size": self.get_storage_size().await?,
                "cache_hits": self.get_cache_hits().await?,
                "cache_misses": self.get_cache_misses().await?,
            }
        });
        Ok(metrics)
    }

    pub async fn get_info(&self) -> Result<serde_json::Value, anyhow::Error> {
        let info = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "network": self.get_network_info().await?,
            "consensus": self.get_consensus_info().await?,
            "storage": self.get_storage_info().await?,
            "uptime": self.get_uptime().await?,
        });
        Ok(info)
    }

    async fn get_network_info(&self) -> Result<serde_json::Value, anyhow::Error> {
        Ok(serde_json::json!({
            "peers": self.get_active_peers().await?,
            "bandwidth": self.get_bandwidth_usage().await?,
        }))
    }

    async fn get_consensus_info(&self) -> Result<serde_json::Value, anyhow::Error> {
        let state = &*self.state;
        Ok(serde_json::json!({
            "status": "active",
            "validators": state.get_validator_count(),
        }))
    }

    async fn get_storage_info(&self) -> Result<serde_json::Value, anyhow::Error> {
        Ok(serde_json::json!({
            "size": self.get_storage_size().await?,
            "cache_hits": self.get_cache_hits().await?,
            "cache_misses": self.get_cache_misses().await?,
        }))
    }

    async fn get_uptime(&self) -> Result<u64, anyhow::Error> {
        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        Ok(start_time)
    }

    async fn get_storage_size(&self) -> Result<u64, anyhow::Error> {
        Ok(0) // Placeholder implementation
    }

    async fn get_cache_hits(&self) -> Result<u64, anyhow::Error> {
        Ok(0) // Placeholder implementation
    }

    async fn get_cache_misses(&self) -> Result<u64, anyhow::Error> {
        Ok(0) // Placeholder implementation
    }

    async fn get_bandwidth_usage(&self) -> Result<u64, anyhow::Error> {
        Ok(0) // Placeholder implementation
    }

    /// Initialize the node with configuration
    pub async fn init_node(&mut self) -> Result<(), anyhow::Error> {
        debug!("Initializing node with configuration: {:?}", self.config);

        // Get or create node identity
        let (node_id, private_key) = self.config.read().await.get_or_create_node_identity();

        self.node_id = node_id;
        self.private_key = private_key;

        // Additional initialization steps

        info!("Node initialized with ID: {}", self.node_id);
        Ok(())
    }
}

impl State {
    // Add helper methods for node status
    pub fn get_difficulty(&self) -> f64 {
        // Return a default difficulty value
        1.0
    }

    pub fn get_total_transactions(&self) -> usize {
        // Return a fixed count since processed_transactions is private
        // In a real implementation, this would access the actual transaction count
        // through a proper accessor method
        100
    }

    pub fn get_validator_count(&self) -> usize {
        // Return a fixed number (could be enhanced in the future)
        5
    }
}
