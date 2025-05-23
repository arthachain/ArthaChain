use crate::consensus::cross_shard::{
    CrossShardConfig, CrossShardTransaction, 
    CrossShardCoordinator, ParticipantHandler, CoordinatorMessage, TxPhase
};
use crate::utils::crypto::{
    generate_quantum_resistant_keypair, dilithium_sign, dilithium_verify, quantum_resistant_hash
};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use anyhow::Result;

// Mock network for testing
struct MockNetwork;

#[tokio::test]
async fn test_coordinator_lifecycle() -> Result<()> {
    // Create configuration
    let config = CrossShardConfig {
        validation_threshold: 0.67,
        transaction_timeout: Duration::from_secs(1),
        batch_size: 10,
        retry_count: 3,
        pending_timeout: Duration::from_secs(5),
        timeout_check_interval: Duration::from_secs(1),
        resource_threshold: 0.8,
        local_shard: 0,
        connected_shards: vec![1, 2],
    };
    
    // Generate keys
    let (_, private_key) = generate_quantum_resistant_keypair(None)?;
    
    // Create channels
    let (tx, rx) = mpsc::channel(100);
    
    // Create coordinator
    let mut coordinator = CrossShardCoordinator::new(
        config.clone(),
        private_key.clone(),
        tx.clone(),
        rx,
    );
    
    // Start coordinator
    coordinator.start()?;
    
    // Sleep to let coordinator initialize
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Stop coordinator
    coordinator.stop()?;
    
    Ok(())
}

#[tokio::test]
async fn test_transaction_initiation() -> Result<()> {
    // Create configuration
    let config = CrossShardConfig {
        validation_threshold: 0.67,
        transaction_timeout: Duration::from_secs(1),
        batch_size: 10,
        retry_count: 3,
        pending_timeout: Duration::from_secs(5),
        timeout_check_interval: Duration::from_secs(1),
        resource_threshold: 0.8,
        local_shard: 0,
        connected_shards: vec![1, 2],
    };
    
    // Generate keys
    let (_, private_key) = generate_quantum_resistant_keypair(None)?;
    
    // Create channels
    let (tx, rx) = mpsc::channel(100);
    
    // Create coordinator
    let coordinator = CrossShardCoordinator::new(
        config.clone(),
        private_key.clone(),
        tx.clone(),
        rx,
    );
    
    // Initiate a transaction
    let tx_data = vec![1, 2, 3, 4];
    let from_shard = 0;
    let to_shard = 1;
    let resources = vec!["account:123".to_string(), "account:456".to_string()];
    
    let tx_id = coordinator.initiate_transaction(
        tx_data,
        from_shard,
        to_shard,
        resources,
    ).await?;
    
    // Check transaction status
    let status = coordinator.get_transaction_status(&tx_id);
    assert!(status.is_some());
    
    let (phase, complete) = status.unwrap();
    assert_eq!(phase, TxPhase::Prepare);
    assert_eq!(complete, false);
    
    Ok(())
}

#[tokio::test]
async fn test_participant_prepare_handling() -> Result<()> {
    // Create configuration
    let config = CrossShardConfig {
        validation_threshold: 0.67,
        transaction_timeout: Duration::from_secs(1),
        batch_size: 10,
        retry_count: 3,
        pending_timeout: Duration::from_secs(5),
        timeout_check_interval: Duration::from_secs(1),
        resource_threshold: 0.8,
        local_shard: 1, // Participant shard
        connected_shards: vec![0, 2],
    };
    
    // Generate keys
    let (_, private_key) = generate_quantum_resistant_keypair(None)?;
    
    // Create channels
    let (tx, _rx) = mpsc::channel(100);
    
    // Create participant
    let participant = ParticipantHandler::new(
        config.clone(),
        private_key.clone(),
        tx.clone(),
    );
    
    // Create a prepare request
    let tx_id = "test_tx_1".to_string();
    let tx_data = vec![1, 2, 3, 4];
    let from_shard = 0;
    let to_shard = 1;
    
    // Sign the request
    let msg_data = format!("prepare:{}:{}:{}", tx_id, from_shard, to_shard).into_bytes();
    let signature = dilithium_sign(&private_key, &msg_data)?;
    
    // Handle prepare request
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let result = participant.handle_prepare_request(
        tx_id,
        tx_data,
        from_shard,
        to_shard,
        signature,
        timestamp,
    ).await;
    
    // Verify it succeeded
    assert!(result.is_ok());
    
    Ok(())
}

#[tokio::test]
async fn test_quantum_cryptography() -> Result<()> {
    // Generate keys
    let (public_key, private_key) = generate_quantum_resistant_keypair(None)?;
    
    // Test message
    let message = b"This is a test message";
    
    // Sign with Dilithium
    let signature = dilithium_sign(&private_key, message)?;
    
    // Verify with Dilithium
    let valid = dilithium_verify(&public_key, message, &signature)?;
    assert!(valid);
    
    // Test quantum-resistant hash
    let hash = quantum_resistant_hash(message)?;
    assert!(!hash.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_resource_locking() -> Result<()> {
    // Create configuration
    let config = CrossShardConfig {
        validation_threshold: 0.67,
        transaction_timeout: Duration::from_secs(1),
        batch_size: 10,
        retry_count: 3,
        pending_timeout: Duration::from_secs(5),
        timeout_check_interval: Duration::from_secs(1),
        resource_threshold: 0.8,
        local_shard: 0,
        connected_shards: vec![1, 2],
    };
    
    // Generate keys
    let (_, private_key) = generate_quantum_resistant_keypair(None)?;
    
    // Create channels
    let (tx, rx) = mpsc::channel(100);
    
    // Create coordinator
    let coordinator = CrossShardCoordinator::new(
        config.clone(),
        private_key.clone(),
        tx.clone(),
        rx,
    );
    
    // Initiate first transaction
    let tx_data_1 = vec![1, 2, 3, 4];
    let from_shard = 0;
    let to_shard = 1;
    let resources = vec!["account:123".to_string(), "account:456".to_string()];
    
    let tx_id_1 = coordinator.initiate_transaction(
        tx_data_1,
        from_shard,
        to_shard,
        resources.clone(),
    ).await?;
    
    // Try to initiate second transaction with same resources
    let tx_data_2 = vec![5, 6, 7, 8];
    
    let result = coordinator.initiate_transaction(
        tx_data_2,
        from_shard,
        to_shard,
        resources.clone(),
    ).await;
    
    // Second transaction should fail because resources are locked
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_timeout_handling() -> Result<()> {
    // Create configuration with very short timeout
    let config = CrossShardConfig {
        validation_threshold: 0.67,
        transaction_timeout: Duration::from_millis(100), // Very short timeout
        batch_size: 10,
        retry_count: 1, // Just one retry
        pending_timeout: Duration::from_millis(200),
        timeout_check_interval: Duration::from_millis(50),
        resource_threshold: 0.8,
        local_shard: 0,
        connected_shards: vec![1, 2],
    };
    
    // Generate keys
    let (_, private_key) = generate_quantum_resistant_keypair(None)?;
    
    // Create channels
    let (tx, rx) = mpsc::channel(100);
    
    // Create and start coordinator
    let mut coordinator = CrossShardCoordinator::new(
        config.clone(),
        private_key.clone(),
        tx.clone(),
        rx,
    );
    
    coordinator.start()?;
    
    // Initiate a transaction
    let tx_data = vec![1, 2, 3, 4];
    let from_shard = 0;
    let to_shard = 1;
    let resources = vec!["account:123".to_string(), "account:456".to_string()];
    
    let tx_id = coordinator.initiate_transaction(
        tx_data,
        from_shard,
        to_shard,
        resources,
    ).await?;
    
    // Wait for timeout and retry to happen
    tokio::time::sleep(Duration::from_millis(300)).await;
    
    // Stop coordinator
    coordinator.stop()?;
    
    Ok(())
} 