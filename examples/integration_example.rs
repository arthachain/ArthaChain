use std::sync::Arc;
use std::time::{Duration, Instant};

use blockchain_node::ledger::transaction::{Transaction, TransactionStatus, TransactionType};
use blockchain_node::network::adaptive_gossip::{AdaptiveGossipConfig, AdaptiveGossipManager};
use blockchain_node::network::peer::{PeerId, PeerInfo};
use blockchain_node::state::quantum_cache::{
    AccountStateCache, BlockCache, CacheConfig, EvictionPolicy,
};
use blockchain_node::transaction::mempool::{EnhancedMempool, MempoolConfig};
use blockchain_node::utils::crypto::generate_quantum_resistant_keypair;

/// A simplified node that integrates all performance optimizations
struct OptimizedNode {
    /// Adaptive gossip manager
    gossip: AdaptiveGossipManager,
    /// Enhanced mempool
    mempool: Arc<EnhancedMempool>,
    /// Account state cache
    account_cache: Arc<AccountStateCache>,
    /// Block cache
    block_cache: Arc<BlockCache>,
    /// Node's quantum-resistant keypair
    pub_key: Vec<u8>,
    priv_key: Vec<u8>,
}

impl OptimizedNode {
    /// Create a new optimized node
    async fn new() -> Self {
        // Generate quantum-resistant keypair
        let (pub_key, priv_key) = generate_quantum_resistant_keypair(None).unwrap();

        // Create adaptive gossip manager
        let gossip_config = AdaptiveGossipConfig {
            min_peers: 5,
            max_peers: 30,
            optimal_peers: 15,
            health_check_interval: Duration::from_secs(5),
            base_gossip_interval: Duration::from_secs(2),
            min_gossip_interval: Duration::from_millis(500),
            max_gossip_interval: Duration::from_secs(10),
            high_latency_threshold: Duration::from_millis(200),
            congestion_threshold: 0.7,
            use_quantum_resistant: true,
        };

        let gossip = AdaptiveGossipManager::new(gossip_config, priv_key.clone(), pub_key.clone());

        // Create enhanced mempool
        let mempool_config = MempoolConfig {
            max_size_bytes: 10 * 1024 * 1024, // 10MB
            max_transactions: 1000,
            default_ttl: Duration::from_secs(60),
            min_gas_price: 1,
            use_quantum_resistant: true,
            cleanup_interval: Duration::from_secs(10),
            max_txs_per_account: 100,
        };

        let mempool = Arc::new(EnhancedMempool::new(mempool_config));

        // Create cache config
        let cache_config = CacheConfig {
            max_size_bytes: 10 * 1024 * 1024, // 10MB
            max_entries: 1000,
            default_ttl: Some(Duration::from_secs(60)),
            eviction_policy: EvictionPolicy::LRU,
            use_quantum_hash: true,
            cleanup_interval: Duration::from_secs(10),
            verify_integrity: true,
            refresh_interval: Some(Duration::from_secs(30)),
            hot_access_threshold: 5,
        };

        // Create account and block caches
        let account_cache = Arc::new(AccountStateCache::new(cache_config.clone()));
        let block_cache = Arc::new(BlockCache::new(cache_config));

        Self {
            gossip,
            mempool,
            account_cache,
            block_cache,
            pub_key,
            priv_key,
        }
    }

    /// Add a peer to the node
    fn add_peer(&self, id: &str, address: &str) {
        let peer_id = PeerId::from(id);
        let peer_info = PeerInfo {
            node_id: id.to_string(),
            address: address.to_string(),
            latency: Duration::from_millis(100),
            last_seen: Instant::now(),
            connected_since: Instant::now(),
            reputation: 0.5,
        };

        self.gossip.add_peer(peer_id, peer_info);
        println!("Added peer {id} at {address}");
    }

    /// Submit a transaction to the mempool
    async fn submit_transaction(&self, tx: Transaction) -> anyhow::Result<()> {
        let hash = self.mempool.add_transaction(tx).await?;
        println!("Transaction submitted with hash: {hash}");
        Ok(())
    }

    /// Process received transaction data (simulated)
    async fn process_received_data(&self, data: Vec<u8>, peer_id: &str) -> anyhow::Result<()> {
        // Create message from received data
        let peer = PeerId::from(peer_id);
        let message = self.gossip.create_message(data.clone(), peer, 10)?;

        // Verify message
        let is_valid = self.gossip.verify_message(&message, &self.pub_key)?;
        if !is_valid {
            println!("Received invalid message from peer {peer_id}");
            return Ok(());
        }

        // Process message content (e.g., decode transaction)
        // In a real implementation, this would deserialize and process the transaction
        println!("Processed valid message from peer {peer_id}");

        Ok(())
    }

    /// Get network status
    fn network_status(&self) -> String {
        format!("{:?}", self.gossip.network_status())
    }

    /// Get mempool statistics
    async fn mempool_stats(&self) -> String {
        let stats = self.mempool.get_stats().await;
        format!(
            "{} transactions, {} bytes",
            stats.total_transactions, stats.size_bytes
        )
    }

    /// Get cache statistics
    async fn cache_stats(&self) -> String {
        let account_stats = self.account_cache.get_stats().await;
        let block_stats = self.block_cache.get_stats().await;

        format!(
            "Account cache: {} hits, {} misses, {:.1}% hit rate; Block cache: {} hits, {} misses, {:.1}% hit rate",
            account_stats.hits,
            account_stats.misses,
            account_stats.hit_rate(),
            block_stats.hits,
            block_stats.misses,
            block_stats.hit_rate()
        )
    }
}

/// Create a test transaction
fn create_test_transaction(
    sender: &str,
    recipient: &str,
    amount: u64,
    gas_price: u64,
) -> Transaction {
    Transaction {
        tx_type: TransactionType::Transfer,
        sender: sender.to_string(),
        recipient: recipient.to_string(),
        amount,
        nonce: 1,
        gas_price,
        gas_limit: 21000,
        data: vec![],
        signature: vec![],
        bls_signature: None,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        status: TransactionStatus::Pending,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Starting optimized node integration example...");

    // Create optimized node
    let node = OptimizedNode::new().await;

    // Add some peers
    node.add_peer("peer1", "127.0.0.1:8001");
    node.add_peer("peer2", "127.0.0.1:8002");
    node.add_peer("peer3", "127.0.0.1:8003");

    // Check network status
    println!("Network status: {}", node.network_status());

    // Submit some transactions
    let tx1 = create_test_transaction("alice", "bob", 100, 10);
    let tx2 = create_test_transaction("bob", "carol", 50, 20);

    node.submit_transaction(tx1).await?;
    node.submit_transaction(tx2).await?;

    // Get mempool stats
    println!("Mempool stats: {}", node.mempool_stats().await);

    // Simulate received data from peer
    let data = b"Hello, quantum world!".to_vec();
    node.process_received_data(data, "peer1").await?;

    // Get cache stats
    println!("Cache stats: {}", node.cache_stats().await);

    println!("Integration example completed successfully!");
    Ok(())
}
