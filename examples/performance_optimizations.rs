use std::time::{Duration, Instant};

use blockchain_node::ledger::transaction::{Transaction, TransactionStatus, TransactionType};
use blockchain_node::network::adaptive_gossip::{AdaptiveGossipConfig, AdaptiveGossipManager};
use blockchain_node::state::quantum_cache::{
    AccountStateCache, BlockCache, CacheConfig, EvictionPolicy,
};
use blockchain_node::transaction::mempool::{EnhancedMempool, MempoolConfig};
use blockchain_node::types::AccountId;
use blockchain_node::utils::crypto::{generate_quantum_resistant_keypair, quantum_resistant_hash};
use blockchain_node::utils::quantum_merkle::{LightClientVerifier, MerkleProofGenerator};

/// Generate a test transaction
fn generate_test_transaction(
    sender: &str,
    recipient: &str,
    nonce: u64,
    gas_price: u64,
) -> Transaction {
    Transaction {
        tx_type: TransactionType::Transfer,
        sender: sender.to_string(),
        recipient: recipient.to_string(),
        amount: 100,
        nonce,
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

/// Generate test data for Merkle tree
fn generate_test_data(count: usize) -> Vec<Vec<u8>> {
    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        let transaction = generate_test_transaction(
            &format!("sender{i}"),
            &format!("recipient{i}"),
            i as u64,
            1 + (i as u64 % 10),
        );

        // Simple serialization instead of bincode
        let serialized = format!(
            "{}-{}-{}-{}",
            transaction.sender, transaction.recipient, transaction.nonce, transaction.amount
        )
        .into_bytes();
        data.push(serialized);
    }
    data
}

/// Test Merkle proof generation and verification with quantum resistance
async fn test_merkle_proofs() {
    println!("Testing quantum-resistant Merkle proofs...");

    // Generate test data
    let data = generate_test_data(1000);
    println!("Generated 1000 test transactions for Merkle tree");

    // Create Merkle proof generator
    let start = Instant::now();
    let generator = MerkleProofGenerator::new(&data).unwrap();
    let build_time = start.elapsed();
    println!("Built Merkle tree in {build_time:?}");

    // Get root hash
    let root_hash = generator.root_hash().unwrap();

    // Create light client verifier
    let verifier = LightClientVerifier::new(vec![root_hash.clone()]);

    // Test proof generation and verification for a sample of data
    let mut total_proof_time = Duration::from_secs(0);
    let mut total_verify_time = Duration::from_secs(0);
    let sample_count = 100;

    for i in 0..sample_count {
        let index = i * 10; // Sample every 10th transaction
        let target_data = &data[index];

        // Generate proof
        let start = Instant::now();
        let proof = generator.generate_proof(target_data).unwrap();
        total_proof_time += start.elapsed();

        // Verify proof
        let start = Instant::now();
        let is_valid = verifier.verify_proof(&proof).unwrap();
        total_verify_time += start.elapsed();

        assert!(
            is_valid,
            "Proof verification failed for data at index {index}"
        );
    }

    let avg_proof_time = total_proof_time / sample_count as u32;
    let avg_verify_time = total_verify_time / sample_count as u32;

    println!("Merkle Proof Results:");
    println!("  Avg Proof Generation Time: {avg_proof_time:?}");
    println!("  Avg Proof Verification Time: {avg_verify_time:?}");
}

/// Test mempool with TTL and prioritization
async fn test_mempool() {
    println!("Testing enhanced mempool with TTL and prioritization...");

    // Create mempool configuration
    let config = MempoolConfig {
        max_size_bytes: 10 * 1024 * 1024, // 10MB
        max_transactions: 10000,
        default_ttl: Duration::from_secs(60),
        min_gas_price: 1,
        use_quantum_resistant: true,
        cleanup_interval: Duration::from_secs(10),
        max_txs_per_account: 100,
    };

    // Create mempool
    let mempool = EnhancedMempool::new(config);

    // Add transactions with different gas prices
    println!("Adding 1000 transactions to mempool...");
    let mut txs_by_priority = Vec::new();

    let start = Instant::now();
    for i in 0..1000 {
        // Vary gas price to test prioritization
        let gas_price = 1 + (i % 20) as u64;
        let tx = generate_test_transaction(
            &format!("sender{}", i / 10),
            "recipient",
            i as u64,
            gas_price,
        );
        txs_by_priority.push((tx.clone(), gas_price));
        mempool.add_transaction(tx).await.unwrap();
    }
    let add_time = start.elapsed();
    println!("Added 1000 transactions in {add_time:?}");

    // Get mempool stats
    let stats = mempool.get_stats().await;
    println!(
        "Mempool stats: {} transactions, {} bytes",
        stats.total_transactions, stats.size_bytes
    );

    // Get best transactions
    let start = Instant::now();
    let best_txs = mempool.get_best_transactions(10).await;
    let query_time = start.elapsed();

    println!("Retrieved top 10 transactions in {query_time:?}");
    println!("Top transaction gas prices:");
    for (i, tx) in best_txs.iter().enumerate() {
        println!("  #{}: Gas price {}", i + 1, tx.gas_price);
    }

    // Test account-based retrieval
    let account_id = AccountId::from("sender0");
    let sender_txs = mempool.get_account_transactions(&account_id).await;
    println!("Account sender0 has {} transactions", sender_txs.len());

    // Clean up expired transactions
    let expired = mempool.cleanup_expired().await.unwrap();
    println!("Cleaned up {expired} expired transactions");
}

/// Test account state caching
async fn test_state_caching() {
    println!("Testing quantum-resistant state caching...");

    // Create cache configuration
    let config = CacheConfig {
        max_size_bytes: 10 * 1024 * 1024, // 10MB
        max_entries: 10000,
        default_ttl: Some(Duration::from_secs(60)),
        eviction_policy: EvictionPolicy::LRU,
        use_quantum_hash: true,
        cleanup_interval: Duration::from_secs(10),
        verify_integrity: true,
        refresh_interval: Some(Duration::from_secs(30)),
        hot_access_threshold: 5,
    };

    // Create account state cache
    let account_cache = AccountStateCache::new(config.clone());
    let block_cache = BlockCache::new(config);

    // Test account state caching
    println!("Adding 1000 account states to cache...");
    let start = Instant::now();
    for i in 0..1000 {
        let account_id = AccountId::from(format!("account{i}"));
        let state = blockchain_node::state::quantum_cache::AccountState {
            balance: 1000 * i as u64,
            nonce: i as u64,
            storage: std::collections::HashMap::new(),
            code: None,
            last_updated: i as u64,
        };
        account_cache
            .update_account_state(account_id, state)
            .await
            .unwrap();
    }
    let add_time = start.elapsed();
    println!("Added 1000 account states in {add_time:?}");

    // Test block caching
    println!("Adding 1000 blocks to cache...");
    let start = Instant::now();
    for i in 0..1000 {
        let block_data = blockchain_node::state::quantum_cache::BlockData {
            height: i as u64,
            hash: quantum_resistant_hash(i.to_string().as_bytes()).unwrap(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            finalized: i < 950,
            transaction_count: 100,
            size_bytes: 1024 * 10,
        };
        block_cache.cache_block(i as u64, block_data).await.unwrap();
    }
    let add_time = start.elapsed();
    println!("Added 1000 blocks in {add_time:?}");

    // Test cache performance
    println!("Testing cache read performance...");

    // Access some accounts multiple times to make them "hot"
    for _ in 0..10 {
        for i in 0..100 {
            let account_id = AccountId::from(format!("account{i}"));
            let _ = account_cache.get_account_state(&account_id).await;
        }
    }

    // Access random blocks
    let mut hit_count = 0;
    let start = Instant::now();
    for i in 0..1000 {
        let block_height = (i % 1200) as u64; // Use deterministic pattern instead of random
        if let Some(_) = block_cache.get_block(&block_height).await {
            hit_count += 1;
        }
    }
    let query_time = start.elapsed();

    println!("Performed 1000 block queries in {query_time:?}");
    println!(
        "Cache hit rate: {}/{} = {:.1}%",
        hit_count,
        1000,
        (hit_count as f32 / 1000.0) * 100.0
    );

    // Get cache stats
    let account_stats = account_cache.get_stats().await;
    let block_stats = block_cache.get_stats().await;

    println!(
        "Account cache stats: hits={}, misses={}, hit_rate={:.1}%",
        account_stats.hits,
        account_stats.misses,
        account_stats.hit_rate()
    );
    println!(
        "Block cache stats: hits={}, misses={}, hit_rate={:.1}%",
        block_stats.hits,
        block_stats.misses,
        block_stats.hit_rate()
    );
}

/// Test adaptive gossip
async fn test_adaptive_gossip() {
    println!("Testing adaptive gossip with peer count monitoring...");

    // Generate quantum-resistant keys
    let (public_key, private_key) = generate_quantum_resistant_keypair(None).unwrap();

    // Create gossip config
    let config = AdaptiveGossipConfig {
        min_peers: 5,
        max_peers: 50,
        optimal_peers: 25,
        health_check_interval: Duration::from_secs(5),
        base_gossip_interval: Duration::from_secs(2),
        min_gossip_interval: Duration::from_millis(500),
        max_gossip_interval: Duration::from_secs(10),
        high_latency_threshold: Duration::from_millis(200),
        congestion_threshold: 0.7,
        use_quantum_resistant: true,
    };

    // Create gossip manager
    let gossip = AdaptiveGossipManager::new(config, private_key, public_key.clone());

    // Simulate varying network conditions
    println!("Simulating sparse network...");
    // Add 3 peers (below min_peers)
    for i in 0..3 {
        let peer_id = blockchain_node::network::peer::PeerId::from(format!("peer_{i}"));
        let peer_info = blockchain_node::network::peer::PeerInfo {
            node_id: format!("node_{i}"),
            address: format!("127.0.0.1:{}", 8000 + i),
            latency: Duration::from_millis(100),
            last_seen: Instant::now(),
            connected_since: Instant::now(),
            reputation: 0.5,
        };
        gossip.add_peer(peer_id, peer_info);
    }

    // Check gossip interval
    gossip.check_health().unwrap();
    println!(
        "Sparse network (3 peers) - Gossip interval: {:?}",
        gossip.gossip_interval()
    );

    // Simulate healthy network
    println!("Simulating healthy network...");
    // Add more peers to reach optimal
    for i in 3..25 {
        let peer_id = blockchain_node::network::peer::PeerId::from(format!("peer_{i}"));
        let peer_info = blockchain_node::network::peer::PeerInfo {
            node_id: format!("node_{i}"),
            address: format!("127.0.0.1:{}", 8000 + i),
            latency: Duration::from_millis(100),
            last_seen: Instant::now(),
            connected_since: Instant::now(),
            reputation: 0.5,
        };
        gossip.add_peer(peer_id, peer_info);
    }

    // Check gossip interval
    gossip.check_health().unwrap();
    println!(
        "Healthy network (25 peers) - Gossip interval: {:?}",
        gossip.gossip_interval()
    );

    // Simulate congested network
    println!("Simulating congested network...");
    // Add more peers to exceed max
    for i in 25..60 {
        let peer_id = blockchain_node::network::peer::PeerId::from(format!("peer_{i}"));
        let peer_info = blockchain_node::network::peer::PeerInfo {
            node_id: format!("node_{i}"),
            address: format!("127.0.0.1:{}", 8000 + i),
            latency: Duration::from_millis(if i > 40 { 300 } else { 100 }),
            last_seen: Instant::now(),
            connected_since: Instant::now(),
            reputation: 0.5,
        };
        gossip.add_peer(peer_id, peer_info);
    }

    // Check gossip interval
    gossip.check_health().unwrap();
    println!(
        "Congested network (60 peers) - Gossip interval: {:?}",
        gossip.gossip_interval()
    );

    // Create and verify a message
    let peer_id = blockchain_node::network::peer::PeerId::from("local_node");
    let message_content = b"Hello, quantum world!".to_vec();

    let message = gossip
        .create_message(message_content.clone(), peer_id.clone(), 10)
        .unwrap();
    let is_valid = gossip.verify_message(&message, &public_key).unwrap();

    println!("Quantum-resistant message verification: {is_valid}");
}

#[tokio::main]
async fn main() {
    println!("Starting performance optimizations example...");

    // Test adaptive gossip
    test_adaptive_gossip().await;

    // Test mempool
    test_mempool().await;

    // Test Merkle proofs
    test_merkle_proofs().await;

    // Test state caching
    test_state_caching().await;

    println!("All tests completed successfully!");
}
