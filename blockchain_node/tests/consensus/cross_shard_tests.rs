use std::sync::Arc;
use tokio::sync::mpsc;
use blockchain_node::consensus::cross_shard::{
    CrossShardManager,
    CrossShardMessage,
    CrossShardTransaction,
    CrossShardTxStatus,
    CrossShardStatus,
};
use blockchain_node::consensus::reputation::ReputationManager;
use std::time::Duration;

async fn setup_test_environment() -> (CrossShardManager, mpsc::Sender<CrossShardMessage>) {
    let (tx, rx) = mpsc::channel(100);
    let reputation_manager = Arc::new(ReputationManager::new(10));
    
    let manager = CrossShardManager::new(
        0, // shard_id
        3, // total_shards
        rx,
        tx.clone(),
        2, // required_signatures
        5, // finalization_timeout
        reputation_manager,
        10, // recovery_timeout
        3,  // max_recovery_attempts
    );

    (manager, tx)
}

fn create_test_transaction(source: u64, target: u64) -> CrossShardTransaction {
    CrossShardTransaction {
        tx_hash: vec![1, 2, 3, 4],
        source_shard: source,
        target_shard: target,
        data: vec![5, 6, 7, 8],
        status: CrossShardTxStatus::Pending,
    }
}

#[tokio::test]
async fn test_finalization_request() {
    let (mut manager, tx) = setup_test_environment().await;
    let block_hash = vec![1, 2, 3, 4];
    let cross_shard_txs = vec![create_test_transaction(0, 1)];

    // Send finalization request
    tx.send(CrossShardMessage::FinalizationRequest {
        shard_id: 1,
        block_hash: block_hash.clone(),
        height: 1,
        timestamp: 1000,
        cross_shard_txs: cross_shard_txs.clone(),
    }).await.unwrap();

    // Process the message
    tokio::spawn(async move {
        manager.start().await.unwrap();
    });

    // Wait for processing
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Verify consensus state
    let consensus = manager.get_consensus_state(&block_hash).await.unwrap();
    assert_eq!(consensus.status, CrossShardStatus::Pending);
    assert_eq!(consensus.height, 1);
}

#[tokio::test]
async fn test_finalization_response() {
    let (mut manager, tx) = setup_test_environment().await;
    let block_hash = vec![1, 2, 3, 4];
    let cross_shard_txs = vec![create_test_transaction(0, 1)];

    // Initialize consensus state with a finalization request
    tx.send(CrossShardMessage::FinalizationRequest {
        shard_id: 1,
        block_hash: block_hash.clone(),
        height: 1,
        timestamp: 1000,
        cross_shard_txs,
    }).await.unwrap();

    // Send finalization responses
    tx.send(CrossShardMessage::FinalizationResponse {
        shard_id: 1,
        block_hash: block_hash.clone(),
        signature: vec![1, 2, 3, 4],
        timestamp: 1000,
    }).await.unwrap();

    tx.send(CrossShardMessage::FinalizationResponse {
        shard_id: 2,
        block_hash: block_hash.clone(),
        signature: vec![5, 6, 7, 8],
        timestamp: 1000,
    }).await.unwrap();

    // Process messages
    tokio::spawn(async move {
        manager.start().await.unwrap();
    });

    // Wait for processing
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Verify finalization
    assert!(manager.is_finalized(&block_hash).await);
}

#[tokio::test]
async fn test_recovery_mechanism() {
    let (mut manager, tx) = setup_test_environment().await;
    let block_hash = vec![1, 2, 3, 4];
    let cross_shard_txs = vec![create_test_transaction(0, 1)];

    // Initialize consensus state
    tx.send(CrossShardMessage::FinalizationRequest {
        shard_id: 1,
        block_hash: block_hash.clone(),
        height: 1,
        timestamp: 1000,
        cross_shard_txs,
    }).await.unwrap();

    // Trigger timeout
    tx.send(CrossShardMessage::FinalizationTimeout {
        shard_id: 1,
        block_hash: block_hash.clone(),
        timestamp: 1006,
    }).await.unwrap();

    // Send recovery responses
    tx.send(CrossShardMessage::RecoveryResponse {
        shard_id: 1,
        block_hash: block_hash.clone(),
        signature: vec![1, 2, 3, 4],
        timestamp: 1007,
    }).await.unwrap();

    tx.send(CrossShardMessage::RecoveryResponse {
        shard_id: 2,
        block_hash: block_hash.clone(),
        signature: vec![5, 6, 7, 8],
        timestamp: 1007,
    }).await.unwrap();

    // Process messages
    tokio::spawn(async move {
        manager.start().await.unwrap();
    });

    // Wait for processing
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Verify recovery and finalization
    let consensus = manager.get_consensus_state(&block_hash).await.unwrap();
    assert_eq!(consensus.status, CrossShardStatus::Finalized);
    assert_eq!(consensus.recovery_attempts, 1);
}

#[tokio::test]
async fn test_invalid_transaction_verification() {
    let (manager, _) = setup_test_environment().await;

    // Test with invalid source shard
    let invalid_tx = create_test_transaction(10, 1); // source_shard > total_shards
    assert!(!manager.verify_transaction_data(&invalid_tx).await.unwrap());

    // Test with invalid target shard
    let invalid_tx = create_test_transaction(0, 10); // target_shard > total_shards
    assert!(!manager.verify_transaction_data(&invalid_tx).await.unwrap());

    // Test with empty data
    let mut empty_data_tx = create_test_transaction(0, 1);
    empty_data_tx.data = vec![];
    assert!(!manager.verify_transaction_data(&empty_data_tx).await.unwrap());
}

#[tokio::test]
async fn test_signature_verification() {
    let (manager, _) = setup_test_environment().await;
    let block_hash = vec![1, 2, 3, 4];
    let signature = vec![5, 6, 7, 8];

    // Test signature verification (currently returns true for testing)
    assert!(manager.verify_signature(1, &block_hash, &signature).await.unwrap());
}

#[tokio::test]
async fn test_concurrent_finalization_requests() {
    let (mut manager, tx) = setup_test_environment().await;
    let block_hash_1 = vec![1, 2, 3, 4];
    let block_hash_2 = vec![5, 6, 7, 8];
    
    let tx_clone = tx.clone();
    
    // Spawn task for first finalization request
    tokio::spawn(async move {
        tx_clone.send(CrossShardMessage::FinalizationRequest {
            shard_id: 1,
            block_hash: block_hash_1.clone(),
            height: 1,
            timestamp: 1000,
            cross_shard_txs: vec![create_test_transaction(0, 1)],
        }).await.unwrap();
    });

    // Send second finalization request
    tx.send(CrossShardMessage::FinalizationRequest {
        shard_id: 2,
        block_hash: block_hash_2.clone(),
        height: 1,
        timestamp: 1000,
        cross_shard_txs: vec![create_test_transaction(0, 2)],
    }).await.unwrap();

    // Process messages
    tokio::spawn(async move {
        manager.start().await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Both consensus states should be pending
    let consensus_1 = manager.get_consensus_state(&block_hash_1).await.unwrap();
    let consensus_2 = manager.get_consensus_state(&block_hash_2).await.unwrap();
    assert_eq!(consensus_1.status, CrossShardStatus::Pending);
    assert_eq!(consensus_2.status, CrossShardStatus::Pending);
}

#[tokio::test]
async fn test_reputation_impact() {
    let (mut manager, tx) = setup_test_environment().await;
    let block_hash = vec![1, 2, 3, 4];
    
    // Initialize consensus state
    tx.send(CrossShardMessage::FinalizationRequest {
        shard_id: 1,
        block_hash: block_hash.clone(),
        height: 1,
        timestamp: 1000,
        cross_shard_txs: vec![create_test_transaction(0, 1)],
    }).await.unwrap();

    // Simulate multiple timeouts to check reputation decrease
    for _ in 0..3 {
        tx.send(CrossShardMessage::FinalizationTimeout {
            shard_id: 1,
            block_hash: block_hash.clone(),
            timestamp: 1006,
        }).await.unwrap();
    }

    tokio::spawn(async move {
        manager.start().await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify reputation impact
    let reputation = manager.get_shard_reputation(1).await;
    assert!(reputation < 10); // Initial reputation was 10
}

#[tokio::test]
async fn test_max_recovery_attempts() {
    let (mut manager, tx) = setup_test_environment().await;
    let block_hash = vec![1, 2, 3, 4];
    
    // Initialize consensus state
    tx.send(CrossShardMessage::FinalizationRequest {
        shard_id: 1,
        block_hash: block_hash.clone(),
        height: 1,
        timestamp: 1000,
        cross_shard_txs: vec![create_test_transaction(0, 1)],
    }).await.unwrap();

    // Trigger multiple recovery attempts
    for i in 0..4 {
        tx.send(CrossShardMessage::FinalizationTimeout {
            shard_id: 1,
            block_hash: block_hash.clone(),
            timestamp: 1006 + i,
        }).await.unwrap();
    }

    tokio::spawn(async move {
        manager.start().await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify consensus failed after max attempts
    let consensus = manager.get_consensus_state(&block_hash).await.unwrap();
    assert_eq!(consensus.status, CrossShardStatus::Failed);
    assert_eq!(consensus.recovery_attempts, 3); // Max attempts was 3
}

#[tokio::test]
async fn test_invalid_message_sequence() {
    let (mut manager, tx) = setup_test_environment().await;
    let block_hash = vec![1, 2, 3, 4];
    
    // Send finalization response without a request first
    tx.send(CrossShardMessage::FinalizationResponse {
        shard_id: 1,
        block_hash: block_hash.clone(),
        signature: vec![1, 2, 3, 4],
        timestamp: 1000,
    }).await.unwrap();

    tokio::spawn(async move {
        manager.start().await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify no consensus state was created
    let consensus = manager.get_consensus_state(&block_hash).await;
    assert!(consensus.is_none());
}

#[tokio::test]
async fn test_duplicate_finalization_responses() {
    let (mut manager, tx) = setup_test_environment().await;
    let block_hash = vec![1, 2, 3, 4];
    
    // Initialize consensus state
    tx.send(CrossShardMessage::FinalizationRequest {
        shard_id: 1,
        block_hash: block_hash.clone(),
        height: 1,
        timestamp: 1000,
        cross_shard_txs: vec![create_test_transaction(0, 1)],
    }).await.unwrap();

    // Send duplicate responses from same shard
    for _ in 0..2 {
        tx.send(CrossShardMessage::FinalizationResponse {
            shard_id: 1,
            block_hash: block_hash.clone(),
            signature: vec![1, 2, 3, 4],
            timestamp: 1000,
        }).await.unwrap();
    }

    tokio::spawn(async move {
        manager.start().await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify only one signature was counted
    let consensus = manager.get_consensus_state(&block_hash).await.unwrap();
    assert_eq!(consensus.signatures.len(), 1);
} 