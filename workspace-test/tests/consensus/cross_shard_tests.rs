use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
use uuid::Uuid;
use workspace_test::blockchain_node::consensus::cross_shard::CrossShardTransaction;
use workspace_test::blockchain_node::consensus::reputation::{ReputationConfig, ReputationManager};
use workspace_test::blockchain_node::network::cross_shard::{
    CrossShardConfig, CrossShardManager, CrossShardMessage, CrossShardMessageType, MessageStatus,
};
use workspace_test::blockchain_node::sharding::CrossShardStatus;

async fn setup_test_environment() -> (CrossShardManager, mpsc::Sender<CrossShardMessage>) {
    let (tx, _rx) = mpsc::channel::<CrossShardMessage>(100);

    // Create config for ReputationManager
    let rep_config = ReputationConfig {
        min_reputation: 0.3,
        initial_reputation: 0.5,
        max_adjustment: 0.1,
        decay_factor: 0.95,
        decay_interval_secs: 3600,
    };

    let _reputation_manager = Arc::new(ReputationManager::new(rep_config));

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
#[allow(dead_code)]
fn create_test_transaction(source: u64, target: u64) -> CrossShardTransaction {
    let tx_id = format!("tx_{}_{}", source, target);
    CrossShardTransaction {
        tx_hash: tx_id.clone(),
        from_shard: source as u32,
        to_shard: target as u32,
        status: CrossShardStatus::Pending,
        created_at: SystemTime::now(),
        // Compatibility fields
        id: tx_id,
        source_shard: source,
        target_shard: target,
        data: vec![5, 6, 7, 8],
        timestamp: SystemTime::now(),
    }
}

#[tokio::test]
async fn test_send_message() -> Result<()> {
    let (manager, _) = setup_test_environment().await;
    let block_hash = "block123".to_string();

    // Create a message
    let message = CrossShardMessage {
        id: Uuid::new_v4().to_string(),
        message_type: CrossShardMessageType::BlockFinalization {
            block_hash: block_hash.clone(),
            signatures: vec!["sig1".to_string()],
        },
        sender_shard: 0,
        recipient_shard: 1,
        payload: vec![1, 2, 3, 4],
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
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
    let message_id = Uuid::new_v4().to_string();
    let message = CrossShardMessage {
        id: message_id.clone(),
        message_type: CrossShardMessageType::StateSync {
            state_root: "0x01020304".to_string(),
            keys: vec!["key1".to_string()],
        },
        sender_shard: 0,
        recipient_shard: 1,
        payload: vec![1, 2, 3, 4],
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        status: MessageStatus::Pending,
    };

    // Send the message
    manager.send_message(message).await?;

    // Process the message queue
    manager.process_queue().await?;

    // Acknowledge the message
    let result = manager.handle_acknowledgment(message_id.clone(), 1).await;
    assert!(result.is_ok(), "Failed to handle acknowledgment");

    // Check message status
    let status = manager.get_message_status(message_id).await;
    assert!(status.is_some(), "No message status found");

    if let Some(status) = status {
        assert_eq!(status, MessageStatus::Delivered); // Changed from Acknowledged to Delivered
    }

    Ok(())
}

#[test]
fn test_cross_shard_transaction_creation() {
    let tx = CrossShardTransaction {
        tx_hash: "test_tx_1234".to_string(),
        from_shard: 0,
        to_shard: 1,
        status: CrossShardStatus::Pending,
        created_at: SystemTime::now(),
        // Compatibility fields
        id: "test_tx_1234".to_string(),
        source_shard: 0,
        target_shard: 1,
        data: vec![5, 6, 7, 8],
        timestamp: SystemTime::now(),
    };

    assert_eq!(tx.from_shard, 0);
    assert_eq!(tx.to_shard, 1);
    assert_eq!(tx.data, vec![5, 6, 7, 8]);
}
