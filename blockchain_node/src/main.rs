use arthachain_node::{
    config::Config,
    ledger::state::State,
    transaction::mempool::Mempool,
    consensus::validator_set::{ValidatorSetManager, ValidatorSetConfig},
    api::testnet_router::create_testnet_router,
    ledger::block::{Block, BlockHeader},
    types::{Hash, Transaction},
    performance::{ParallelProcessor, parallel_processor::ProcessingTask},
    sharding::{ShardManager, ShardingConfig, ShardInfo, ShardType},
};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

const TARGET_TPS: u64 = 100_000;
const WORKER_COUNT: usize = 16;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env_logger::init();
    
    println!("Starting ArthaChain Blockchain Node");
    println!("Target: {} TPS | Confirmation: <0.1s", TARGET_TPS);
    
    let config = Config::default();
    println!("Configuration loaded");
    
    let state = Arc::new(RwLock::new(State::new(&config)?));
    println!("Blockchain state initialized");
    
    let mempool = Arc::new(RwLock::new(Mempool::new(1000000)));
    println!("High-capacity mempool initialized (1M transactions)");
    
    let validator_manager = Arc::new(ValidatorSetManager::new(ValidatorSetConfig {
        min_validators: 1,
        max_validators: 100,
        rotation_interval: 1000,
    }));
    println!("Validator manager initialized");
    
    // Initialize sharding system
    let sharding_config = ShardingConfig::default();
    let shard_manager = Arc::new(ShardManager::new(sharding_config.clone()));
    println!("Sharding system initialized with {} shards", sharding_config.total_shards);
    
    // Initialize parallel processor
    let parallel_processor = Arc::new(ParallelProcessor::new(WORKER_COUNT, shard_manager.clone()));
    println!("Parallel processor initialized with {} workers", WORKER_COUNT);
    
    // Initialize node as validator
    println!("Initializing node as validator...");
    
    let bls_public_key = vec![
        0x8f, 0x4e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e,
        0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e,
        0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e,
        0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e, 0x8c, 0x8e
    ];
    
    let private_key = vec![
        0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x7a, 0x8b, 0x9c, 0x0d, 0x1e, 0x2f,
        0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f, 0x9a, 0x0b, 0x1c, 0x2d, 0x3e, 0x4f,
        0x5a, 0x6b, 0x7c, 0x8d, 0x9e, 0x0f, 0x1a, 0x2b
    ];
    
    match validator_manager
        .register_validator(bls_public_key.clone(), private_key.clone())
        .await
    {
        Ok(_) => println!("Node successfully registered as validator"),
        Err(e) => {
            println!("Validator registration failed: {}", e);
            println!("Attempting alternative registration method...");
            
            let node_address = vec![0x74, 0x2d, 0x35, 0x43, 0x63, 0x66, 0x34, 0x43, 0x30, 0x35, 0x33, 0x32, 0x39, 0x32, 0x35, 0x61, 0x33, 0x62, 0x38, 0x44];
            
            if let Err(e2) = validator_manager
                .register_validator(node_address, private_key)
                .await
            {
                println!("Alternative registration also failed: {}", e2);
                return Err(format!("Failed to register as validator: {} | {}", e, e2).into());
            } else {
                println!("Alternative registration successful");
            }
        }
    }
    
    // Initialize shards with different types for optimization
    initialize_shards(&shard_manager).await;
    
    // Start high-performance mining system
    println!("Starting high-performance mining system...");
    
    // Start mining workers for each shard
    for shard_id in 0..sharding_config.total_shards {
        let processor_clone = parallel_processor.clone();
        let state_clone = state.clone();
        let mempool_clone = mempool.clone();
        let shard_manager_clone = shard_manager.clone();
        
        tokio::spawn(async move {
            mining_worker(
                shard_id,
                processor_clone,
                state_clone,
                mempool_clone,
                shard_manager_clone
            ).await;
        });
    }
    
    // Start transaction processing worker
    let processor_tx = parallel_processor.clone();
    let state_tx = state.clone();
    let mempool_tx = mempool.clone();
    let shard_manager_tx = shard_manager.clone();
    tokio::spawn(async move {
        transaction_processor(processor_tx, state_tx, mempool_tx, shard_manager_tx).await;
    });
    
    // Start performance monitoring
    let processor_monitor = parallel_processor.clone();
    let shard_monitor = shard_manager.clone();
    tokio::spawn(async move {
        performance_monitor(processor_monitor, shard_monitor).await;
    });
    
    let app = create_testnet_router(state.clone(), validator_manager, mempool).await;
    println!("API router created");
    
    println!("ArthaChain Blockchain launched successfully");
    println!("🌐 GLOBAL ENDPOINTS:");
    println!("API Dashboard: https://api.arthachain.in/api/stats");
    println!("Block Explorer: https://explorer.arthachain.in/api/explorer/blocks/recent");
    println!("Faucet: https://faucet.arthachain.in/");
    println!("Validators: https://validators.arthachain.in/");
    println!("High-Performance Mining System: ACTIVE");
    println!("Target: {} TPS | <0.1s confirmation", TARGET_TPS);
    
    println!("Starting API server on http://0.0.0.0:8080 (globally accessible)");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    println!("API server listening on http://0.0.0.0:8080 (globally accessible)");
    
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn initialize_shards(shard_manager: &Arc<ShardManager>) {
    // Initialize different types of shards for optimization
    let shard_types = vec![
        (0, ShardType::HighPerformance),
        (1, ShardType::StorageOptimized),
        (2, ShardType::ComputeIntensive),
        (3, ShardType::GeneralPurpose),
    ];
    
    for (shard_id, shard_type) in shard_types {
        if let Err(e) = shard_manager.register_shard(shard_id, shard_type.clone(), 16).await {
            println!("Failed to register shard {}: {}", shard_id, e);
        } else {
            println!("Shard {} registered with type {:?}", shard_id, shard_type);
        }
    }
}

async fn mining_worker(
    shard_id: u64,
    processor: Arc<ParallelProcessor>,
    state: Arc<RwLock<State>>,
    mempool: Arc<RwLock<Mempool>>,
    shard_manager: Arc<ShardManager>,
) {
    println!("Mining Worker {} started for shard {}", shard_id, shard_id);
    
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
    
    loop {
        interval.tick().await;
        
        let mempool_read = mempool.read().await;
        let pending_count = mempool_read.get_stats().await.pending_count;
        drop(mempool_read);
        
        if pending_count > 0 {
            let batch_size = 10000;
            
            let mut mempool_write = mempool.write().await;
            let transactions = mempool_write.get_transactions_for_block(batch_size).await;
            let tx_count = transactions.len();
            
            if tx_count > 0 {
                let current_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                let mut state_write = state.write().await;
                let current_height = state_write.get_height().unwrap_or(0);
                
                let prev_hash = state_write.latest_block()
                    .map(|b| b.hash().unwrap_or_default())
                    .unwrap_or_else(|| Hash::default());
                
                // Convert mempool transactions to ledger transactions for block creation
                let ledger_transactions: Vec<arthachain_node::ledger::transaction::Transaction> = transactions.iter().map(|tx| {
                    arthachain_node::ledger::transaction::Transaction {
                        tx_type: arthachain_node::ledger::transaction::TransactionType::Transfer,
                        sender: tx.from.to_hex(),
                        recipient: tx.to.to_hex(),
                        amount: tx.value,
                        nonce: tx.nonce,
                        gas_price: tx.gas_price,
                        gas_limit: tx.gas_limit,
                        data: tx.data.clone(),
                        signature: tx.signature.clone(),
                        timestamp: current_time,
                        #[cfg(feature = "bls")]
                        bls_signature: None,
                        status: arthachain_node::ledger::transaction::TransactionStatus::Pending,
                    }
                }).collect();
                
                let new_block = create_block(
                    prev_hash,
                    transactions.clone(),
                    current_height + 1,
                    current_time,
                    shard_id as u32
                );
                
                match state_write.add_block(new_block.clone()) {
                    Ok(_) => {
                        println!("Worker {}: Block {} mined with {} transactions", 
                                shard_id, current_height + 1, tx_count);
                        
                        // Update shard performance metrics
                        if let Some(shard_info) = shard_manager.get_shard_info(shard_id).await {
                            // Update shard metrics through the performance monitor
                            let _ = shard_manager.get_performance_monitor().update_shard_metrics(
                                shard_id,
                                tx_count as u64,
                                tx_count as u64,
                            ).await;
                        }
                        
                        for tx in transactions {
                                                let tx_hash = arthachain_node::utils::crypto::Hash::from_slice(tx.hash().as_bytes());
                    mempool_write.mark_executed(&tx_hash).await;
                        }
                    }
                    Err(e) => {
                        println!("Worker {}: Error adding block: {}", shard_id, e);
                    }
                }
            }
            drop(mempool_write);
        }
    }
}

async fn transaction_processor(
    processor: Arc<ParallelProcessor>,
    state: Arc<RwLock<State>>,
    mempool: Arc<RwLock<Mempool>>,
    shard_manager: Arc<ShardManager>,
) {
    println!("Transaction Processor started - Target: {} TPS", TARGET_TPS);
    
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(1));
    
    loop {
        interval.tick().await;
        
        let state_read = state.read().await;
        let pending_txs = state_read.get_pending_transactions(10000);
        drop(state_read);
        
        if !pending_txs.is_empty() {
            let mut mempool_write = mempool.write().await;
            
            for tx in pending_txs {
                // Convert ledger Transaction to types Transaction
                let mempool_tx = Transaction {
                    from: arthachain_node::types::Address::from_bytes(tx.sender.as_bytes()).unwrap_or_default(),
                    to: arthachain_node::types::Address::from_bytes(tx.recipient.as_bytes()).unwrap_or_default(),
                    value: tx.amount,
                    gas_price: tx.gas_price,
                    gas_limit: tx.gas_limit,
                    nonce: tx.nonce,
                    data: tx.data.clone(),
                    signature: tx.signature.clone(),
                    hash: arthachain_node::utils::crypto::Hash::from_slice(tx.signature.as_slice()),
                };
                
                if let Ok(_) = mempool_write.add_transaction(mempool_tx.clone()).await {
                    // Get optimal shard for transaction
                    let transaction_data = arthachain_node::sharding::TransactionData {
                        from: mempool_tx.from.clone(),
                        to: mempool_tx.to.clone(),
                        value: mempool_tx.value,
                        nonce: mempool_tx.nonce,
                        data: mempool_tx.data.clone(),
                    };
                    
                    if let Some(optimal_shard) = shard_manager.get_optimal_shard(&transaction_data).await {
                        // Send to parallel processor with shard information
                        let _ = processor.submit_task(
                            ProcessingTask::TransactionProcessing {
                                transaction: mempool_tx.clone(),
                                shard_id: optimal_shard,
                            }
                        ).await;
                    }
                }
            }
            drop(mempool_write);
        }
    }
}

async fn performance_monitor(
    processor: Arc<ParallelProcessor>,
    shard_manager: Arc<ShardManager>,
) {
    println!("Performance Monitor started");
    
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
    
    loop {
        interval.tick().await;
        
        let processor_metrics = processor.get_performance_metrics().await;
        let shard_metrics = shard_manager.get_performance_monitor().get_global_metrics().await;
        
        println!("Performance Report:");
        println!("  Total Transactions: {}", processor_metrics.total_transactions_processed);
        println!("  Total Blocks: {}", processor_metrics.total_blocks_created);
        println!("  Current TPS: {}", processor_metrics.current_tps);
        println!("  Avg Confirmation: {}ms", processor_metrics.average_processing_time.as_millis());
        println!("  Peak TPS: {}", processor_metrics.peak_tps);
        println!("  Shard Metrics:");
        println!("    Total Shard TPS: {}", shard_metrics.total_transactions);
        println!("    Total Blocks: {}", shard_metrics.total_blocks);
        println!("    Average TPS: {:.2}", shard_metrics.average_tps);
        println!("");
    }
}

fn create_block(
    prev_hash: Hash,
    transactions: Vec<Transaction>,
    height: u64,
    timestamp: u64,
    shard_id: u32,
) -> Block {
    let merkle_root = calculate_merkle_root(&transactions);
    
    let header = BlockHeader {
        previous_hash: prev_hash,
        merkle_root,
        timestamp,
        height,
        producer: arthachain_node::ledger::block::BlsPublicKey::default(),
        nonce: shard_id as u64,
        difficulty: 1000000,
    };
    
    Block {
        header,
        transactions: transactions.into_iter().map(|tx| {
            arthachain_node::ledger::block::Transaction {
                id: Hash::new(tx.hash().as_bytes().to_vec()),
                from: tx.from.as_bytes().to_vec(),
                to: tx.to.as_bytes().to_vec(),
                amount: tx.value,
                fee: tx.gas_price,
                data: tx.data,
                nonce: tx.nonce,
                signature: Some(arthachain_node::Signature::new(tx.signature)),
            }
        }).collect(),
        signature: None,
    }
}

fn calculate_merkle_root(transactions: &[Transaction]) -> Hash {
    if transactions.is_empty() {
        return Hash::default();
    }
    
    use blake3::Hasher;
    let mut hasher = Hasher::new();
    
    for tx in transactions {
        hasher.update(&tx.from.as_bytes());
        hasher.update(&tx.to.as_bytes());
        hasher.update(&tx.value.to_le_bytes());
        hasher.update(&tx.nonce.to_le_bytes());
    }
    
    let hash = hasher.finalize();
    Hash::new(hash.as_bytes().to_vec())
}
