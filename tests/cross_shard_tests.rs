use std::sync::Arc;
use tokio::sync::mpsc;
use blockchain_node::network::cross_shard::{
    CrossShardManager,
    CrossShardMessage,
    CrossShardConfig,
    CrossShardMessageType,
    MessageStatus,
};
use blockchain_node::consensus::cross_shard::{
    CrossShardTransaction,
};
use blockchain_node::sharding::CrossShardStatus;
use blockchain_node::consensus::reputation::ReputationManager;
use anyhow::Result;
use std::time::{SystemTime, Duration};
use std::collections::HashMap;
use uuid::Uuid;

async fn setup_consensus_test_environment() -> (CrossShardManager, mpsc::Sender<CrossShardMessage>) {
    let (tx, _rx) = mpsc::channel(100);
    let _reputation_manager = Arc::new(ReputationManager::new(
        10.0,  // max_score
        100,   // history_size
        0.5,   // slashing_threshold
        1000,  // min_stake
    ));
    
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
fn create_test_transaction(source: u64, target: u64) -> CrossShardTransaction {
    CrossShardTransaction {
        id: format!("tx_{}_{}", source, target),
        source_shard: source,
        target_shard: target,
        data: vec![5, 6, 7, 8],
        timestamp: SystemTime::now(),
    }
}

fn create_test_message(source: u64, target: u64) -> CrossShardMessage {
    CrossShardMessage {
        id: Uuid::new_v4(),
        message_type: CrossShardMessageType::TransactionForward {
            transaction: vec![1, 2, 3, 4],
            source_shard: source,
            target_shard: target,
        },
        source_shard: source,
        target_shard: target,
        timestamp: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
        sequence_number: 1,
        signature: vec![],
        retry_count: 0,
        status: MessageStatus::Pending,
    }
}

#[tokio::test]
async fn test_message_sending() -> Result<()> {
    let (manager, _tx) = setup_consensus_test_environment().await;
    
    // Create a test message
    let message = create_test_message(0, 1);
    
    // Send message
    manager.send_message(message.clone()).await?;
    
    // Process the queue
    manager.process_queue().await?;
    
    // Wait for processing
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Get message status - might be pending or sent depending on processing
    let status = manager.get_message_status(message.id).await;
    assert!(status.is_some(), "Message status should exist");
    
    Ok(())
}

#[tokio::test]
async fn test_acknowledgment() -> Result<()> {
    let (manager, _tx) = setup_consensus_test_environment().await;
    
    // Create and send a test message
    let message = create_test_message(0, 1);
    manager.send_message(message.clone()).await?;
    
    // Process the queue
    manager.process_queue().await?;
    
    // Acknowledge the message
    manager.handle_acknowledgment(message.id, 1).await?;
    
    // Get message status
    let status = manager.get_message_status(message.id).await;
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
    
    manager.sync_state(shard_id, state_root.clone(), height).await?;
    
    // Get state sync info
    let info = manager.get_state_sync_info(shard_id).await;
    assert!(info.is_some(), "State sync info should exist");
    
    if let Some(info) = info {
        assert_eq!(info.shard_id, shard_id);
        assert_eq!(info.state_root, state_root);
        assert_eq!(info.status.current_height, height);
        assert!(info.status.is_syncing);
    }
    
    Ok(())
}

#[test]
fn test_cross_shard_transaction_basic() {
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