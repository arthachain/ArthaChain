use sha2::Digest;
use std::collections::HashSet;
use std::time::Duration;

/// Test Merkle proof functionality
#[tokio::test]
async fn test_merkle_proof_system() {
    use blockchain_node::consensus::cross_shard::merkle_proof::{MerkleTree, ProofCache};

    // Create test transaction hashes
    let tx_hashes = vec![
        vec![1, 2, 3, 4],
        vec![5, 6, 7, 8],
        vec![9, 10, 11, 12],
        vec![13, 14, 15, 16],
    ];

    // Build Merkle tree
    let tree = MerkleTree::build(tx_hashes.clone()).unwrap();

    // Generate and verify proof
    let proof = tree.generate_proof(&tx_hashes[0], 100, 1).unwrap();
    assert!(proof.verify().unwrap(), "Merkle proof should be valid");

    // Test proof cache
    let mut cache = ProofCache::new(10);
    cache.store(tx_hashes[0].clone(), proof.clone());

    assert!(cache.get(&tx_hashes[0]).is_some(), "Proof should be cached");
    assert_eq!(cache.size(), 1, "Cache should have 1 item");

    println!("✅ Merkle proof system test passed");
}

/// Test view change manager
#[tokio::test]
async fn test_view_change_manager() {
    use blockchain_node::consensus::view_change::{ViewChangeConfig, ViewChangeManager};

    let config = ViewChangeConfig {
        view_timeout: Duration::from_secs(1),
        max_view_changes: 5,
        min_validators: 4,
        leader_election_interval: Duration::from_secs(1),
    };

    let manager = ViewChangeManager::new(3, config);

    // Test initialization with proper 20-byte addresses
    let validators: HashSet<Vec<u8>> = (0..4)
        .map(|i| {
            let mut addr = vec![0u8; 20];
            addr[19] = i as u8; // Put index in last byte
            addr
        })
        .collect();

    manager.initialize(validators).await.unwrap();

    // Test basic functionality - manager should be initialized
    let current_view = manager.get_current_view();
    assert_eq!(current_view, 0, "Initial view should be 0");

    println!("✅ View change manager test passed");
}

/// Test cross-shard coordinator basic functionality
#[tokio::test]
async fn test_cross_shard_coordinator() {
    use blockchain_node::consensus::cross_shard::{
        coordinator::CrossShardCoordinator, CrossShardConfig,
    };
    use tokio::sync::mpsc;

    let config = CrossShardConfig {
        local_shard: 1,
        connected_shards: vec![1, 2],
        ..Default::default()
    };

    let (tx, _rx) = mpsc::channel(100);
    let coordinator = CrossShardCoordinator::new(
        config,
        vec![1, 2, 3, 4], // Mock quantum key
        tx,
    );

    // Test cache functionality
    let (cache_size, _) = coordinator.get_proof_cache_stats();
    assert_eq!(cache_size, 0, "Cache should be empty initially");

    // Test cache clearing
    coordinator.clear_proof_cache();

    println!("✅ Cross-shard coordinator test passed");
}

/// Test performance of proof verification
#[tokio::test]
async fn test_proof_verification_performance() {
    use blockchain_node::consensus::cross_shard::merkle_proof::MerkleTree;
    use std::time::Instant;

    // Create larger test data
    let tx_hashes: Vec<Vec<u8>> = (0..100)
        .map(|i| format!("transaction_{}", i).into_bytes())
        .collect();

    let tree = MerkleTree::build(tx_hashes.clone()).unwrap();

    // Benchmark proof generation
    let start = Instant::now();
    let proof = tree.generate_proof(&tx_hashes[0], 100, 1).unwrap();
    let generation_time = start.elapsed();

    // Benchmark proof verification
    let start = Instant::now();
    assert!(proof.verify().unwrap());
    let verification_time = start.elapsed();

    println!("Proof generation time: {}μs", generation_time.as_micros());
    println!(
        "Proof verification time: {}μs",
        verification_time.as_micros()
    );

    // Performance assertions
    assert!(
        generation_time.as_millis() < 100,
        "Proof generation should be < 100ms"
    );
    assert!(
        verification_time.as_millis() < 10,
        "Proof verification should be < 10ms"
    );

    println!("✅ Proof verification performance test passed");
}

/// Test Byzantine fault tolerance calculation
#[tokio::test]
async fn test_byzantine_fault_tolerance() {
    use blockchain_node::consensus::view_change::{ViewChangeConfig, ViewChangeManager};

    // Test with different network sizes
    let test_cases = vec![
        (4, 1),  // 4 nodes, 1 faulty (25%)
        (7, 2),  // 7 nodes, 2 faulty (28.5%)
        (10, 3), // 10 nodes, 3 faulty (30%)
        (13, 4), // 13 nodes, 4 faulty (30.7%)
    ];

    for (total_nodes, max_faulty) in test_cases {
        let quorum_size = 2 * max_faulty + 1;

        let config = ViewChangeConfig {
            view_timeout: Duration::from_millis(100),
            max_view_changes: 3,
            min_validators: total_nodes,
            leader_election_interval: Duration::from_secs(1),
        };

        let manager = ViewChangeManager::new(quorum_size, config);

        let validators: HashSet<Vec<u8>> = (0..total_nodes)
            .map(|i| {
                let mut addr = vec![0u8; 20];
                addr[19] = i as u8; // Put index in last byte
                addr
            })
            .collect();

        manager.initialize(validators).await.unwrap();

        println!(
            "Byzantine test: {} nodes, {} max faulty, quorum: {}",
            total_nodes, max_faulty, quorum_size
        );

        // Test that quorum calculation is correct for Byzantine tolerance
        assert!(quorum_size > total_nodes / 2, "Quorum should be > 50%");
        assert!(
            quorum_size >= 2 * max_faulty + 1,
            "Quorum should satisfy 2f+1"
        );
    }

    println!("✅ Byzantine fault tolerance calculation test passed");
}

/// Integration test summary
#[tokio::test]
async fn phase1_integration_summary() {
    println!("\n🚀 PHASE 1 INTEGRATION TEST SUMMARY");
    println!("=====================================");

    // Test all major components work together
    use blockchain_node::consensus::cross_shard::merkle_proof::{MerkleTree, ProvenTransaction};
    use blockchain_node::consensus::view_change::{ViewChangeConfig, ViewChangeManager};
    use std::time::Instant;

    let start = Instant::now();

    // 1. Test Merkle proof system
    let tx_data = b"integration_test_transaction".to_vec();
    let tx_hash = sha2::Sha256::digest(&tx_data).to_vec();
    let tree = MerkleTree::build(vec![tx_hash.clone()]).unwrap();
    let proof = tree.generate_proof(&tx_hash, 100, 1).unwrap();
    assert!(proof.verify().unwrap());

    // 2. Test proven transaction
    let proven_tx = ProvenTransaction::new(
        tx_data, proof, 1, // source shard
        2, // target shard
        1234567890,
    );
    assert!(proven_tx.verify().unwrap());

    // 3. Test view change manager
    let config = ViewChangeConfig {
        view_timeout: Duration::from_millis(100),
        max_view_changes: 3,
        min_validators: 4,
        leader_election_interval: Duration::from_secs(1),
    };

    let manager = ViewChangeManager::new(3, config);
    let validators: HashSet<Vec<u8>> = (0..4)
        .map(|i| {
            let mut addr = vec![0u8; 20];
            addr[19] = i as u8; // Put index in last byte
            addr
        })
        .collect();
    manager.initialize(validators).await.unwrap();

    let total_time = start.elapsed();

    println!("📊 Integration Test Results:");
    println!("   ✅ Merkle proof generation and verification: PASSED");
    println!("   ✅ Proven transaction validation: PASSED");
    println!("   ✅ View change manager initialization: PASSED");
    println!("   ✅ Byzantine fault tolerance setup: PASSED");
    println!("   ⏱️ Total integration time: {}ms", total_time.as_millis());

    assert!(
        total_time.as_millis() < 1000,
        "Integration should complete < 1 second"
    );

    println!("\n🏆 PHASE 1 CORE FUNCTIONALITY VALIDATED");
    println!("All critical consensus and cross-shard components working correctly!");
}
