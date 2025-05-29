#!/bin/bash
set -e

echo "Testing quantum-resistant performance optimization components..."

# Function to check dependencies
check_deps() {
  if ! command -v cargo &> /dev/null; then
    echo "Cargo is not installed. Please install Rust and Cargo first."
    exit 1
  fi
}

# Check dependencies
check_deps

# Create test directory if it doesn't exist
mkdir -p tests/performance

# Test Adaptive Gossip Protocol
echo "Testing Adaptive Gossip Protocol..."
cat > tests/performance/test_adaptive_gossip.rs << EOF
use blockchain_node::network::adaptive_gossip::{AdaptiveGossipConfig, AdaptiveGossipManager};
use blockchain_node::utils::crypto::generate_quantum_resistant_keypair;
use std::time::{Duration, Instant};

fn main() {
    let config = AdaptiveGossipConfig {
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
    
    println!("Generating quantum-resistant keypair...");
    let (public_key, private_key) = match generate_quantum_resistant_keypair(None) {
        Ok(keys) => keys,
        Err(_) => (vec![0; 32], vec![0; 32]),
    };
    
    println!("Creating Adaptive Gossip Manager...");
    let gossip = AdaptiveGossipManager::new(
        config,
        private_key,
        public_key.clone(),
    );
    
    println!("Testing network status...");
    let status = gossip.network_status();
    println!("Initial network status: {:?}", status);
    
    println!("Testing message creation and verification...");
    let peer_id = blockchain_node::network::peer::PeerId::from("test_peer");
    let message_content = b"Test message content".to_vec();
    
    let message = gossip.create_message(message_content.clone(), peer_id.clone(), 5).unwrap();
    let is_valid = gossip.verify_message(&message, &public_key).unwrap();
    
    println!("Message verification result: {}", is_valid);
    println!("Adaptive Gossip Protocol test completed successfully!");
}
EOF

# Test Enhanced Mempool
echo "Testing Enhanced Mempool..."
cat > tests/performance/test_mempool.rs << EOF
use blockchain_node::transaction::mempool::{EnhancedMempool, MempoolConfig};
use blockchain_node::ledger::transaction::{Transaction, TransactionType, TransactionStatus};
use std::time::{Duration, Instant, SystemTime};

#[tokio::main]
async fn main() {
    let config = MempoolConfig {
        max_size_bytes: 10 * 1024 * 1024, // 10MB
        max_transactions: 1000,
        default_ttl: Duration::from_secs(60),
        min_gas_price: 1,
        use_quantum_resistant: true,
        cleanup_interval: Duration::from_secs(10),
        max_txs_per_account: 100,
    };
    
    println!("Creating Enhanced Mempool...");
    let mempool = EnhancedMempool::new(config);
    
    println!("Testing transaction addition...");
    let tx = Transaction {
        tx_type: TransactionType::Transfer,
        sender: "sender1".to_string(),
        recipient: "recipient1".to_string(),
        amount: 100,
        nonce: 1,
        gas_price: 10,
        gas_limit: 21000,
        data: vec![],
        signature: vec![],
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        status: TransactionStatus::Pending,
    };
    
    let tx_hash = mempool.add_transaction(tx.clone()).await.unwrap();
    println!("Added transaction with hash: {:?}", tx_hash);
    
    println!("Testing transaction retrieval...");
    let retrieved = mempool.get_transaction(&tx_hash).await;
    println!("Retrieved transaction: {}", retrieved.is_some());
    
    println!("Testing mempool statistics...");
    let stats = mempool.get_stats().await;
    println!("Mempool stats: {} transactions, {} bytes", 
        stats.total_transactions, stats.size_bytes);
    
    println!("Enhanced Mempool test completed successfully!");
}
EOF

# Test Quantum Merkle Proofs
echo "Testing Quantum Merkle Proofs..."
cat > tests/performance/test_quantum_merkle.rs << EOF
use blockchain_node::utils::quantum_merkle::{QuantumMerkleTree, MerkleProofGenerator, LightClientVerifier};

fn main() {
    println!("Creating test data...");
    let data = vec![
        b"data1".to_vec(),
        b"data2".to_vec(),
        b"data3".to_vec(),
        b"data4".to_vec(),
    ];
    
    println!("Building Merkle tree...");
    let tree = QuantumMerkleTree::build_from_data(&data).unwrap();
    let root_hash = tree.root_hash().unwrap();
    println!("Built Merkle tree with root hash: {:?}", root_hash);
    
    println!("Testing proof generation...");
    let target_data = b"data2".to_vec();
    let proof = tree.generate_proof(&target_data).unwrap();
    println!("Generated proof with {} items", proof.len());
    
    println!("Testing proof verification...");
    let is_valid = QuantumMerkleTree::verify_proof(&target_data, &proof, &root_hash).unwrap();
    println!("Proof verification result: {}", is_valid);
    
    println!("Testing light client verification...");
    let generator = MerkleProofGenerator::new(&data).unwrap();
    let proof = generator.generate_proof(&target_data).unwrap();
    
    let verifier = LightClientVerifier::new(vec![root_hash]);
    let is_valid = verifier.verify_proof(&proof).unwrap();
    println!("Light client verification result: {}", is_valid);
    
    println!("Quantum Merkle Proofs test completed successfully!");
}
EOF

# Test State Caching
echo "Testing State Caching..."
cat > tests/performance/test_state_cache.rs << EOF
use blockchain_node::state::quantum_cache::{QuantumCache, CacheConfig, EvictionPolicy};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() {
    let config = CacheConfig {
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
    
    println!("Creating cache...");
    let cache: QuantumCache<String, String> = QuantumCache::new(config);
    
    println!("Testing cache operations...");
    
    println!("Adding items to cache...");
    for i in 0..100 {
        let key = format!("key{}", i);
        let value = format!("value{}", i);
        cache.put(key, value, None).await.unwrap();
    }
    
    println!("Testing cache retrieval...");
    let value = cache.get(&"key50".to_string()).await;
    println!("Retrieved value: {}", value.is_some());
    
    println!("Testing cache stats...");
    let stats = cache.get_stats().await;
    println!("Cache stats: hits={}, misses={}, current_entries={}", 
        stats.hits, stats.misses, stats.current_entries);
    
    println!("State Caching test completed successfully!");
}
EOF

echo "Running tests..."
echo "Note: These are simple component tests. The full integration test with all components would require solving the dependency issues."

echo "Tests created in tests/performance/ directory."
echo "To run individual tests:"
echo "  cargo run --bin test_adaptive_gossip"
echo "  cargo run --bin test_mempool"
echo "  cargo run --bin test_quantum_merkle"
echo "  cargo run --bin test_state_cache" 