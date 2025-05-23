use anyhow::Result;
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;

use crate::consensus::cross_shard::{
    BeaconBlockInfo, CheckpointData, CrossShardError, CrossShardManager, CrossShardMessage,
    CrossShardTransaction, CrossShardTxStatus, CrossShardTxType,
};
use crate::consensus::reputation::ReputationManager;

async fn setup_test_manager() -> (CrossShardManager, mpsc::Sender<CrossShardMessage>) {
    let (tx, rx) = mpsc::channel(100);
    let reputation_manager = std::sync::Arc::new(ReputationManager::new(
        0.5_f64,  // threshold
        10_usize, // window_size
        1.0_f64,  // reward
        10_u64,   // penalty
    ));

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
    let result = manager
        .handle_finalization_request(
            1,
            block_hash.clone(),
            1,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cross_shard_txs,
            Vec::new(),
        )
        .await;

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
    manager
        .handle_finalization_request(
            1,
            block_hash.clone(),
            1,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            vec![create_test_transaction()],
            Vec::new(),
        )
        .await?;

    // Test finalization response handling
    let result = manager
        .handle_finalization_response(
            1,
            block_hash.clone(),
            CrossShardStatus::Finalized,
            signature.clone(),
            None,
        )
        .await;

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
    assert!(matches!(
        result,
        Err(CrossShardError::InvalidTransaction(_))
    ));

    // Test transaction timeout
    let mut timed_out_tx = tx.clone();
    timed_out_tx.last_update = Some(
        SystemTime::now() - Duration::from_secs(10), // 10 seconds ago
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

    let result = manager
        .handle_beacon_update(beacon_block, Some(checkpoint))
        .await;
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

#[cfg(test)]
mod tests {
    use rand::{seq::SliceRandom, thread_rng, Rng};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::runtime::Runtime;
    use tokio::sync::{mpsc, RwLock};

    use crate::sharding::{CrossShardMessageType, ShardConfig, ShardManager};

    #[test]
    fn test_cross_shard_performance() {
        // This test will measure the performance of cross-shard transactions
        // using various ratios of cross-shard vs. intra-shard transactions

        let rt = Runtime::new().unwrap();

        // Configure sharding
        rt.block_on(async {
            let num_shards = 4;
            let num_nodes = 16; // 4 nodes per shard
            let cross_shard_ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0];

            for &cross_shard_ratio in &cross_shard_ratios {
                println!("\nTesting with cross-shard ratio: {:.2}", cross_shard_ratio);

                // Create shard managers
                let mut shard_managers = Vec::with_capacity(num_shards);
                let mut message_queues = Vec::with_capacity(num_shards);

                for shard_id in 0..num_shards {
                    let (tx, rx) = mpsc::channel(1000);

                    let config = ShardConfig {
                        shard_id,
                        total_shards: num_shards,
                        nodes_per_shard: num_nodes / num_shards,
                        block_size: 1000,
                        block_interval: Duration::from_millis(500),
                        batch_size: 100,
                    };

                    let manager = Arc::new(RwLock::new(ShardManager::new(config, rx)));

                    shard_managers.push(manager);
                    message_queues.push(tx);
                }

                // Generate transactions (80% local, 20% cross-shard)
                let num_transactions = 10_000;
                let mut rng = thread_rng();

                // Track metrics for reporting
                let start_time = Instant::now();
                let mut local_tx_count = 0;
                let mut cross_tx_count = 0;

                for _ in 0..num_transactions {
                    // Determine if this is a cross-shard transaction
                    let is_cross_shard = rng.gen::<f64>() < cross_shard_ratio;

                    if is_cross_shard {
                        // Select source and destination shards
                        let source_shard = rng.gen_range(0..num_shards);
                        let mut dest_shard = rng.gen_range(0..num_shards);

                        // Make sure destination is different from source
                        while dest_shard == source_shard {
                            dest_shard = rng.gen_range(0..num_shards);
                        }

                        // Create a cross-shard message
                        let message = CrossShardMessageType::Transaction {
                            tx_id: format!("tx-{}", rng.gen::<u64>()),
                            source_shard,
                            destination_shard: dest_shard,
                            amount: rng.gen_range(1..1000),
                        };

                        // Send to source shard
                        if let Err(e) = message_queues[source_shard].send(message).await {
                            println!("Failed to send cross-shard tx: {}", e);
                        } else {
                            cross_tx_count += 1;
                        }
                    } else {
                        // Local transaction
                        let shard_id = rng.gen_range(0..num_shards);

                        // Create a local transaction message
                        let message = CrossShardMessageType::LocalTransaction {
                            tx_id: format!("tx-{}", rng.gen::<u64>()),
                            sender: format!("account-{}", rng.gen::<u64>()),
                            receiver: format!("account-{}", rng.gen::<u64>()),
                            amount: rng.gen_range(1..1000),
                        };

                        // Send to destination shard
                        if let Err(e) = message_queues[shard_id].send(message).await {
                            println!("Failed to send local tx: {}", e);
                        } else {
                            local_tx_count += 1;
                        }
                    }
                }

                // Process transactions
                let mut handles = Vec::new();

                for (i, manager) in shard_managers.iter().enumerate() {
                    let manager_clone = manager.clone();

                    let handle = tokio::spawn(async move {
                        let mut processed = 0;
                        let mut pending = 0;

                        // Process for 5 seconds
                        let end_time = Instant::now() + Duration::from_secs(5);

                        while Instant::now() < end_time {
                            // Process batch of transactions
                            let result = {
                                let mut mgr = manager_clone.write().await;
                                mgr.process_batch(100).await
                            };

                            if let Ok((proc, pend)) = result {
                                processed += proc;
                                pending = pend;
                            }

                            // Short sleep to allow other shards to process
                            tokio::time::sleep(Duration::from_millis(50)).await;
                        }

                        (i, processed, pending)
                    });

                    handles.push(handle);
                }

                // Wait for all shards to finish processing
                let mut total_processed = 0;
                let mut total_pending = 0;

                for handle in handles {
                    let (shard_id, processed, pending) = handle.await.unwrap();
                    println!(
                        "Shard {}: processed {} txs, {} pending",
                        shard_id, processed, pending
                    );

                    total_processed += processed;
                    total_pending += pending;
                }

                // Calculate metrics
                let elapsed = start_time.elapsed();
                let tps = total_processed as f64 / elapsed.as_secs_f64();

                println!("Total processed: {}", total_processed);
                println!("Total pending: {}", total_pending);
                println!("Local transactions: {}", local_tx_count);
                println!("Cross-shard transactions: {}", cross_tx_count);
                println!("Time elapsed: {:.2?}", elapsed);
                println!("Throughput: {:.2} tps", tps);

                // For automated testing, assert minimum performance
                let min_tps = match cross_shard_ratio {
                    r if r < 0.2 => 1000.0, // Higher requirements for mostly local txs
                    r if r < 0.5 => 500.0,
                    _ => 250.0, // Lower requirements for mostly cross-shard txs
                };

                assert!(
                    tps > min_tps,
                    "TPS too low: {:.2} (min: {:.2}) for cross-shard ratio {:.2}",
                    tps,
                    min_tps,
                    cross_shard_ratio
                );

                // Check that all transactions were processed or are pending
                assert_eq!(
                    pending, 0,
                    "Not all transactions were processed for cross-shard ratio {:.2}",
                    cross_shard_ratio
                );
            }
        });
    }
}
