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

async fn setup_consensus_test_environment() -> (CrossShardManager, mpsc::Sender<CrossShardMessage>)
{
    let (tx, _rx) = mpsc::channel(100);

    // Create config for ReputationManager
    let rep_config = ReputationConfig {
        min_reputation: 0.3,
        initial_reputation: 0.5,
        max_adjustment: 0.1,
        decay_factor: 0.95,
        decay_interval_secs: 3600,
    };

    let _reputation_manager = Arc::new(ReputationManager::new(rep_config));

    let config = CrossShardConfig {
        max_retries: 3,
        retry_interval: Duration::from_secs(5),
        message_timeout: Duration::from_secs(30),
        batch_size: 100,
        max_queue_size: 1000,
        sync_interval: Duration::from_secs(10),
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

fn create_test_message(source: u64, target: u64) -> CrossShardMessage {
    CrossShardMessage {
        id: Uuid::new_v4().to_string(),
        message_type: CrossShardMessageType::BlockFinalization {
            block_hash: format!("block_{}_{}", source, target),
            signatures: vec!["sig1".to_string()],
        },
        sender_shard: source as u32,
        recipient_shard: target as u32,
        payload: vec![1, 2, 3, 4],
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        status: MessageStatus::Pending,
    }
}

#[tokio::test]
async fn test_message_sending() -> Result<()> {
    let (manager, _tx) = setup_consensus_test_environment().await;

    // Create a test message
    let message = create_test_message(0, 1);
    let message_id = message.id.clone(); // Clone the ID before moving the message

    // Send message
    manager.send_message(message).await?;

    // Process the queue
    manager.process_queue().await?;

    // Wait for processing
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Get message status - might be pending or sent depending on processing
    let status = manager.get_message_status(message_id).await;
    assert!(status.is_some(), "Message status should exist");

    Ok(())
}

#[tokio::test]
async fn test_acknowledgment() -> Result<()> {
    let (manager, _tx) = setup_consensus_test_environment().await;

    // Create and send a test message
    let message = create_test_message(0, 1);
    let message_id = message.id.clone(); // Clone the ID before moving the message

    manager.send_message(message).await?;

    // Process the queue
    manager.process_queue().await?;

    // Acknowledge the message
    manager.handle_acknowledgment(message_id.clone(), 1).await?;

    // Get message status
    let status = manager.get_message_status(message_id).await;
    assert!(status.is_some(), "Message status should exist");

    Ok(())
}

#[tokio::test]
async fn test_state_sync() -> Result<()> {
    let (manager, _tx) = setup_consensus_test_environment().await;

    // Sync state
    let shard_id = 1;
    let state_root = vec![10, 20, 30, 40];
    let height = 100;

    manager
        .sync_state(shard_id, state_root.clone(), height)
        .await?;

    // Get state sync info
    let info = manager.get_state_sync_info(shard_id).await;
    assert!(info.is_some(), "State sync info should exist");

    if let Some(info) = info {
        assert_eq!(info.shard_id, shard_id);
        assert_eq!(info.state_root, state_root);
    }

    Ok(())
}

#[test]
fn test_cross_shard_transaction_basic() {
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
