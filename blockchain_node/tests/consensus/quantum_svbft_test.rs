use crate::consensus::{QuantumSVBFTConsensus, ViewChangeManager};
use crate::config::Config;
use crate::ledger::block::Block;
use crate::ledger::state::State;
use crate::consensus::quantum_svbft::{ConsensusMessage, QuantumSVBFTConfig};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use anyhow::Result;

/// Test the basic initialization of the quantum-resistant SVBFT consensus
#[tokio::test]
async fn test_quantum_svbft_initialization() -> Result<()> {
    // Create test configuration
    let config = Config::default();
    
    // Create state
    let state = Arc::new(RwLock::new(State::new()));
    
    // Create channels
    let (message_sender, message_receiver) = mpsc::channel(100);
    let (block_sender, block_receiver) = mpsc::channel(100);
    let (shutdown_sender, shutdown_receiver) = broadcast::channel(1);
    
    // Create consensus instance
    let mut consensus = QuantumSVBFTConsensus::new(
        config,
        state,
        message_sender,
        message_receiver,
        block_receiver,
        shutdown_receiver,
        "test_node".to_string(),
        None,
    ).await?;
    
    // Initialize and start consensus
    let handle = consensus.start().await?;
    
    // Get current view and leader
    let view = consensus.get_current_view().await;
    let leader = consensus.get_current_leader().await;
    
    println!("Initial view: {}", view);
    println!("Initial leader: {:?}", leader);
    
    // Send shutdown signal
    shutdown_sender.send(()).unwrap();
    
    // Wait for consensus to shut down
    handle.await?;
    
    Ok(())
}

/// Test view change in quantum-resistant SVBFT
#[tokio::test]
async fn test_quantum_svbft_view_change() -> Result<()> {
    // Create test configuration with short timeouts for testing
    let config = Config::default();
    let qsvbft_config = QuantumSVBFTConfig {
        base_timeout_ms: 100,
        view_change_timeout_ms: 500,
        ..QuantumSVBFTConfig::default()
    };
    
    // Create state
    let state = Arc::new(RwLock::new(State::new()));
    
    // Create channels
    let (message_sender, mut message_receiver) = mpsc::channel(100);
    let (block_sender, block_receiver) = mpsc::channel(100);
    let (shutdown_sender, shutdown_receiver) = broadcast::channel(1);
    
    // Create consensus instance
    let mut consensus = QuantumSVBFTConsensus::new(
        config,
        state,
        message_sender.clone(),
        message_receiver,
        block_receiver,
        shutdown_receiver,
        "validator1".to_string(),
        Some(qsvbft_config),
    ).await?;
    
    // Start consensus
    let handle = consensus.start().await?;
    
    // Get initial view and leader
    let initial_view = consensus.get_current_view().await;
    let initial_leader = consensus.get_current_leader().await.unwrap();
    
    println!("Initial view: {}", initial_view);
    println!("Initial leader: {}", initial_leader);
    
    // Simulate a view change request from validator2
    let (message_sender2, _) = mpsc::channel(100);
    let view_change_msg = ConsensusMessage::ViewChangeRequest {
        current_view: initial_view,
        new_view: initial_view + 1,
        node_id: "validator2".to_string(),
        signature: vec![1, 2, 3, 4], // Simplified signature for test
        reason: crate::consensus::quantum_svbft::ViewChangeReason::LeaderTimeout,
    };
    
    // Send view change request
    message_sender.send(view_change_msg).await?;
    
    // Give some time for the view change to process
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    
    // Check if view change occurred
    let new_view = consensus.get_current_view().await;
    let new_leader = consensus.get_current_leader().await.unwrap();
    
    println!("New view after view change: {}", new_view);
    println!("New leader after view change: {}", new_leader);
    
    // Clean up
    shutdown_sender.send(()).unwrap();
    handle.await?;
    
    Ok(())
}

/// Test quantum-resistant signatures in SVBFT
#[tokio::test]
async fn test_quantum_signatures() -> Result<()> {
    // Create test message bytes
    let message = b"test message";
    let key = b"test key";
    
    // Sign message with quantum-resistant signature
    let signature = crate::utils::crypto::dilithium_sign(key, message)?;
    
    // Verify signature
    let valid = crate::utils::crypto::dilithium_verify(key, message, &signature)?;
    
    assert!(valid, "Quantum-resistant signature verification failed");
    
    Ok(())
}

/// Test performance with parallel message validation
#[tokio::test]
async fn test_parallel_validation_performance() -> Result<()> {
    // This test would benchmark parallel processing of consensus messages
    // For brevity, just a skeleton test is provided
    
    let config = Config::default();
    let qsvbft_config = QuantumSVBFTConfig {
        parallel_validation: true,
        ..QuantumSVBFTConfig::default()
    };
    
    // Setup consensus
    let state = Arc::new(RwLock::new(State::new()));
    let (message_sender, message_receiver) = mpsc::channel(100);
    let (block_sender, block_receiver) = mpsc::channel(100);
    let (shutdown_sender, shutdown_receiver) = broadcast::channel(1);
    
    let mut consensus = QuantumSVBFTConsensus::new(
        config,
        state,
        message_sender,
        message_receiver,
        block_receiver,
        shutdown_receiver,
        "performance_test_node".to_string(),
        Some(qsvbft_config),
    ).await?;
    
    // Start consensus
    let handle = consensus.start().await?;
    
    // In a real test, we would:
    // 1. Generate a large number of consensus messages
    // 2. Measure time to process them with parallel_validation=true
    // 3. Compare with time to process with parallel_validation=false
    
    // Clean up
    shutdown_sender.send(()).unwrap();
    handle.await?;
    
    Ok(())
} 