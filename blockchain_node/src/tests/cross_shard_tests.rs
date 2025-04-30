use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
use anyhow::Result;

use crate::consensus::cross_shard::{
    CrossShardManager, CrossShardMessage, CrossShardTransaction,
    CrossShardTxType, CrossShardTxStatus, CrossShardError,
    BeaconBlockInfo, CheckpointData
};
use crate::consensus::reputation::ReputationManager;

async fn setup_test_manager() -> (CrossShardManager, mpsc::Sender<CrossShardMessage>) {
    let (tx, rx) = mpsc::channel(100);
    let reputation_manager = std::sync::Arc::new(ReputationManager::new(
        0.5_f64,   // threshold
        10_usize,  // window_size
        1.0_f64,   // reward
        10_u64,    // penalty
    ));
    
    let manager = CrossShardManager::new(
        0,      // shard_id
        3,      // total_shards
        rx,
        tx.clone(),
        2,      // required_signatures
        5,      // finalization_timeout
        reputation_manager,
        10,     // recovery_timeout
        3,      // max_recovery_attempts
    );

    (manager, tx)
}

fn create_test_transaction() -> CrossShardTransaction {
    CrossShardTransaction {
        tx_hash: vec![1, 2, 3, 4],
        tx_type: CrossShardTxType::DirectTransfer {
            from: vec![5, 6, 7, 8],
            to: vec![9, 10, 11, 12],
            amount: 100,
        },
        source_shard: 0,
        target_shards: vec![1],
        data: vec![13, 14, 15, 16],
        status: CrossShardTxStatus::Pending,
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        size: 100,
        priority: 1,
        locality_hint: Some(1),
        merkle_proof: None,
        witness_data: None,
        last_update: Some(SystemTime::now()),
    }
}

#[tokio::test]
async fn test_finalization_request() -> Result<()> {
    let (manager, _tx) = setup_test_manager().await;
    let block_hash = vec![1, 2, 3, 4];
    let cross_shard_txs = vec![create_test_transaction()];

    // Test finalization request handling
    let result = manager.handle_finalization_request(
        1,
        block_hash.clone(),
        1,
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        cross_shard_txs,
        Vec::new(),
    ).await;

    assert!(result.is_ok());

    // Verify consensus state
    let state = manager.get_consensus_state(&block_hash).await.unwrap();
    assert_eq!(state.status, CrossShardStatus::Pending);
    assert_eq!(state.height, 1);
    assert_eq!(state.signatures.len(), 0);

    Ok(())
}

#[tokio::test]
async fn test_finalization_response() -> Result<()> {
    let (manager, _tx) = setup_test_manager().await;
    let block_hash = vec![1, 2, 3, 4];
    let signature = vec![5, 6, 7, 8];

    // First create a finalization request
    manager.handle_finalization_request(
        1,
        block_hash.clone(),
        1,
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        vec![create_test_transaction()],
        Vec::new(),
    ).await?;

    // Test finalization response handling
    let result = manager.handle_finalization_response(
        1,
        block_hash.clone(),
        CrossShardStatus::Finalized,
        signature.clone(),
        None,
    ).await;

    assert!(result.is_ok());

    // Verify consensus state
    let state = manager.get_consensus_state(&block_hash).await.unwrap();
    assert!(state.signatures.contains_key(&1));
    assert_eq!(state.signatures.get(&1).unwrap(), &signature);

    Ok(())
}

#[tokio::test]
async fn test_transaction_verification() -> Result<()> {
    let (manager, _tx) = setup_test_manager().await;
    let tx = create_test_transaction();

    // Test valid transaction
    let result = manager.verify_transaction(&tx).await;
    assert!(result.is_ok());

    // Test invalid shard ID
    let mut invalid_tx = tx.clone();
    invalid_tx.source_shard = 10; // Greater than total_shards
    let result = manager.verify_transaction(&invalid_tx).await;
    assert!(matches!(result, Err(CrossShardError::InvalidShardId(_))));

    // Test empty transaction hash
    let mut invalid_tx = tx.clone();
    invalid_tx.tx_hash = vec![];
    let result = manager.verify_transaction(&invalid_tx).await;
    assert!(matches!(result, Err(CrossShardError::InvalidTransaction(_))));

    // Test transaction timeout
    let mut timed_out_tx = tx.clone();
    timed_out_tx.last_update = Some(
        SystemTime::now() - Duration::from_secs(10) // 10 seconds ago
    );
    let result = manager.verify_transaction(&timed_out_tx).await;
    assert!(matches!(result, Err(CrossShardError::ConsensusTimeout(_))));

    Ok(())
}

#[tokio::test]
async fn test_error_conversion() -> Result<()> {
    let anyhow_err = anyhow::anyhow!("test error");
    let cross_shard_err: CrossShardError = anyhow_err.into();
    
    match cross_shard_err {
        CrossShardError::Internal(msg) => assert!(msg.contains("test error")),
        _ => panic!("Expected Internal error variant"),
    }

    Ok(())
}

#[tokio::test]
async fn test_beacon_update() -> Result<()> {
    let (manager, _tx) = setup_test_manager().await;
    
    let beacon_block = BeaconBlockInfo {
        height: 1,
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        merkle_root: vec![1, 2, 3, 4],
        finality_cert: vec![5, 6, 7, 8],
    };

    let checkpoint = CheckpointData {
        checkpoint_hash: vec![9, 10, 11, 12],
        state_root: vec![13, 14, 15, 16],
        shard_states: std::collections::HashMap::new(),
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    let result = manager.handle_beacon_update(beacon_block, Some(checkpoint)).await;
    assert!(result.is_ok());

    Ok(())
}

#[tokio::test]
async fn test_performance_metrics() -> Result<()> {
    let (manager, _tx) = setup_test_manager().await;
    
    // Initial metrics should have default values
    let metrics = manager.metrics.read().await;
    assert_eq!(metrics.tps, 0.0);
    assert_eq!(metrics.success_rate, 0.0);
    assert_eq!(metrics.avg_block_size, 0);
    
    // Test metrics adjustment
    let config = manager.config.read().await;
    assert!(config.min_signatures > 0);
    assert!(config.max_block_size > 0);
    assert!(config.target_tps > 0.0);

    Ok(())
} 