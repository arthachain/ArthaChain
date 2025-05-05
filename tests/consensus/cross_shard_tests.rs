use std::sync::Arc;
use std::time::{SystemTime, Duration};
use std::collections::HashMap;
use blockchain_node::network::cross_shard::{
    CrossShardManager,
    CrossShardConfig,
    CrossShardMessage,
    CrossShardMessageType,
    MessageStatus,
};
use blockchain_node::consensus::cross_shard::{
    CrossShardTransaction,
};
use blockchain_node::sharding::CrossShardStatus;
use blockchain_node::consensus::reputation::ReputationManager;
use anyhow::Result;
use tokio::sync::mpsc;
use uuid::Uuid;

async fn setup_test_environment() -> (CrossShardManager, mpsc::Sender<CrossShardMessage>) {
    let (tx, _rx) = mpsc::channel::<CrossShardMessage>(100);
    let _reputation_manager = Arc::new(ReputationManager::new(
        10.0,  // max_score
        100,   // history_size
        0.5,   // slashing_threshold
        1000,  // min_stake
    ));
    
    // Create config for CrossShardManager
    let config = CrossShardConfig {
        max_retries: 3,
        retry_interval: Duration::from_secs(5),
        message_timeout: Duration::from_secs(30),
        batch_size: 100,
        max_queue_size: 1000,
        sync_interval: Duration::from_secs(60),
    };
    
    let manager = CrossShardManager::new(config);

    (manager, tx)
}

// We're keeping this function even though it has the dead_code warning
// because it might be useful in future tests
fn create_test_transaction(source: u64, target: u64) -> CrossShardTransaction {
    CrossShardTransaction {
        id: format!("tx_{}_{}", source, target),
        source_shard: source,
        target_shard: target,
        data: vec![5, 6, 7, 8],
        timestamp: SystemTime::now(),
    }
}

#[tokio::test]
async fn test_send_message() -> Result<()> {
    let (manager, _) = setup_test_environment().await;
    let block_hash = vec![1, 2, 3, 4];
    
    // Create a message
    let message = CrossShardMessage {
        id: Uuid::new_v4(),
        message_type: CrossShardMessageType::BlockFinalization {
            block_hash: block_hash.clone(),
            shard_id: 1,
            height: 1,
        },
        source_shard: 0,
        target_shard: 1,
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        sequence_number: 0,
        signature: vec![],
        retry_count: 0,
        status: MessageStatus::Pending,
    };

    // Send the message
    let result = manager.send_message(message).await;
    assert!(result.is_ok(), "Failed to send message");

    // Process the queue
    let process_result = manager.process_queue().await;
    assert!(process_result.is_ok(), "Failed to process message queue");

    // Wait for processing
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    Ok(())
}

#[tokio::test]
async fn test_sync_state() -> Result<()> {
    let (manager, _) = setup_test_environment().await;
    let state_root = vec![1, 2, 3, 4];

    // Test state sync
    let result = manager.sync_state(1, state_root.clone(), 100).await;
    assert!(result.is_ok(), "Failed to sync state");

    // Get state sync info
    let sync_info = manager.get_state_sync_info(1).await;
    assert!(sync_info.is_some(), "No sync info found");
    
    if let Some(info) = sync_info {
        assert_eq!(info.shard_id, 1);
        assert_eq!(info.state_root, state_root);
    }

    Ok(())
}

#[tokio::test]
async fn test_message_acknowledgment() -> Result<()> {
    let (manager, _) = setup_test_environment().await;
    
    // Create and send a message
    let message_id = Uuid::new_v4();
    let message = CrossShardMessage {
        id: message_id,
        message_type: CrossShardMessageType::StateSync {
            shard_id: 1,
            state_root: vec![1, 2, 3, 4],
            height: 100,
        },
        source_shard: 0,
        target_shard: 1,
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        sequence_number: 0,
        signature: vec![],
        retry_count: 0,
        status: MessageStatus::Pending,
    };

    // Send the message
    manager.send_message(message).await?;
    
    // Process the message queue
    manager.process_queue().await?;
    
    // Acknowledge the message
    let result = manager.handle_acknowledgment(message_id, 1).await;
    assert!(result.is_ok(), "Failed to handle acknowledgment");
    
    // Check message status
    let status = manager.get_message_status(message_id).await;
    assert!(status.is_some(), "No message status found");
    
    if let Some(status) = status {
        assert_eq!(status, MessageStatus::Acknowledged);
    }

    Ok(())
}

#[test]
fn test_cross_shard_transaction_creation() {
    let tx = CrossShardTransaction {
        id: "test_tx_1234".to_string(),
        source_shard: 0,
        target_shard: 1,
        data: vec![5, 6, 7, 8],
        timestamp: SystemTime::now(),
    };
    
    assert_eq!(tx.source_shard, 0);
    assert_eq!(tx.target_shard, 1);
    assert_eq!(tx.data, vec![5, 6, 7, 8]);
}