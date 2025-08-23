use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use std::time::{Duration, Instant};
use num_traits::ToPrimitive;

use crate::types::{Transaction, Address, Hash};
use crate::ledger::block::Block;
use crate::sharding::{ShardManager, ShardInfo, ShardType};

/// Performance metrics for the parallel processor
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_transactions_processed: u64,
    pub total_blocks_created: u64,
    pub total_blocks_validated: u64,
    pub average_processing_time: Duration,
    pub peak_tps: u64,
    pub current_tps: u64,
}

/// Performance metrics for individual shards
#[derive(Debug)]
pub struct ShardMetrics {
    pub transactions_processed: AtomicU64,
    pub blocks_created: AtomicU64,
    pub average_processing_time: AtomicU64, // in nanoseconds
    pub current_load: AtomicU64,
}

/// Types of processing tasks
#[derive(Debug, Clone)]
pub enum ProcessingTask {
    TransactionProcessing {
        transaction: Transaction,
        shard_id: u64,
    },
    BlockCreation {
        shard_id: u64,
        transactions: Vec<Transaction>,
        previous_hash: Hash,
        height: u64,
    },
    BlockValidation {
        block: Block,
        shard_id: u64,
    },
    UpdateMetrics {
        shard_id: u64,
        // TODO: Fix when ShardMetrics can be cloned
        // metrics: ShardMetrics,
    },
}

/// High-performance parallel processor for ArthaChain
pub struct ParallelProcessor {
    worker_pools: Vec<JoinHandle<()>>,
    task_sender: mpsc::Sender<ProcessingTask>,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    shard_distribution: Arc<RwLock<HashMap<u64, ShardMetrics>>>,
    shard_manager: Arc<ShardManager>,
}

impl ParallelProcessor {
    /// Create a new parallel processor
    pub fn new(worker_count: usize, shard_manager: Arc<ShardManager>) -> Self {
        let (task_sender, task_receiver) = mpsc::channel(10000);
        let performance_metrics = Arc::new(RwLock::new(PerformanceMetrics {
            total_transactions_processed: 0,
            total_blocks_created: 0,
            total_blocks_validated: 0,
            average_processing_time: Duration::from_millis(0),
            peak_tps: 0,
            current_tps: 0,
        }));
        let shard_distribution = Arc::new(RwLock::new(HashMap::new()));

        let mut worker_pools = Vec::new();
        let task_receiver = Arc::new(RwLock::new(Some(task_receiver)));

        for worker_id in 0..worker_count {
            let task_receiver_clone = task_receiver.clone();
            let performance_metrics_clone = performance_metrics.clone();
            let shard_distribution_clone = shard_distribution.clone();
            let shard_manager_clone = shard_manager.clone();

            let handle = tokio::spawn(async move {
                Self::worker_execution_loop(
                    worker_id,
                    task_receiver_clone,
                    performance_metrics_clone,
                    shard_distribution_clone,
                    shard_manager_clone,
                ).await;
            });

            worker_pools.push(handle);
        }

        Self {
            worker_pools,
            task_sender,
            performance_metrics,
            shard_distribution,
            shard_manager,
        }
    }

    /// Worker execution loop
    async fn worker_execution_loop(
        worker_id: usize,
        task_receiver: Arc<RwLock<Option<mpsc::Receiver<ProcessingTask>>>>,
        performance_metrics: Arc<RwLock<PerformanceMetrics>>,
        shard_distribution: Arc<RwLock<HashMap<u64, ShardMetrics>>>,
        shard_manager: Arc<ShardManager>,
    ) {
        println!("Worker {} started", worker_id);

        loop {
            // Check if we have a receiver available
            let has_receiver = {
                let receiver_guard = task_receiver.read().await;
                receiver_guard.is_some()
            };
            
            if !has_receiver {
                break;
            }
            
            // Try to receive a task without holding the lock
            let task = {
                let mut receiver_guard = task_receiver.write().await;
                if let Some(receiver) = receiver_guard.as_mut() {
                    match receiver.try_recv() {
                        Ok(task) => Some(task),
                        Err(mpsc::error::TryRecvError::Empty) => {
                            // No task available, sleep briefly and continue
                            drop(receiver_guard);
                            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                            continue;
                        }
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            // Receiver is disconnected, remove it
                            *receiver_guard = None;
                            break;
                        }
                    }
                } else {
                    break;
                }
            };
            
            if let Some(task) = task {
                let start_time = Instant::now();
                
                match task {
                    ProcessingTask::TransactionProcessing { transaction, shard_id } => {
                        Self::execute_transaction_processing(
                            transaction,
                            shard_id,
                            &performance_metrics,
                            &shard_distribution,
                            &shard_manager,
                        ).await;
                    }
                    ProcessingTask::BlockCreation { shard_id, transactions, previous_hash, height } => {
                        Self::execute_block_creation(
                            shard_id,
                            transactions,
                            previous_hash,
                            height,
                            &performance_metrics,
                            &shard_distribution,
                            &shard_manager,
                        ).await;
                    }
                    ProcessingTask::BlockValidation { block, shard_id } => {
                        Self::execute_block_validation(
                            block,
                            shard_id,
                            &performance_metrics,
                            &shard_distribution,
                            &shard_manager,
                        ).await;
                    }
                    ProcessingTask::UpdateMetrics { shard_id, .. } => {
                        // TODO: Update shard metrics when ShardMetrics can be cloned
                        println!("Updating metrics for shard {}", shard_id);
                    }
                }

                let processing_time = start_time.elapsed();
                
                // Update performance metrics
                let mut metrics = performance_metrics.write().await;
                metrics.total_transactions_processed += 1;
                metrics.average_processing_time = Duration::from_nanos(
                    ((metrics.average_processing_time.as_nanos() + processing_time.as_nanos()) / 2) as u64
                );
            }
        }
    }

    /// Execute transaction processing task
    async fn execute_transaction_processing(
        transaction: Transaction,
        shard_id: u64,
        performance_metrics: &Arc<RwLock<PerformanceMetrics>>,
        shard_distribution: &Arc<RwLock<HashMap<u64, ShardMetrics>>>,
        shard_manager: &Arc<ShardManager>,
    ) {
        // Process transaction in the assigned shard
        let result = Self::process_transaction_in_shard(&transaction, shard_id, shard_manager).await;
        
        if let Ok(_) = result {
            // Update shard metrics
            let mut shards = shard_distribution.write().await;
            if let Some(shard_metrics) = shards.get_mut(&shard_id) {
                shard_metrics.transactions_processed.fetch_add(1, Ordering::Relaxed);
            }

            // Update performance metrics
            let mut metrics = performance_metrics.write().await;
            metrics.total_transactions_processed += 1;
            metrics.current_tps = metrics.current_tps.saturating_add(1);
            if metrics.current_tps > metrics.peak_tps {
                metrics.peak_tps = metrics.current_tps;
            }
        }
    }

    /// Execute block creation task
    async fn execute_block_creation(
        shard_id: u64,
        transactions: Vec<Transaction>,
        previous_hash: Hash,
        height: u64,
        performance_metrics: &Arc<RwLock<PerformanceMetrics>>,
        shard_distribution: &Arc<RwLock<HashMap<u64, ShardMetrics>>>,
        shard_manager: &Arc<ShardManager>,
    ) {
        // Create block in the assigned shard
        let result = Self::create_shard_block(
            shard_id,
            transactions,
            previous_hash,
            height,
            shard_manager,
        ).await;

        if let Ok(_) = result {
            // Update shard metrics
            let mut shards = shard_distribution.write().await;
            if let Some(shard_metrics) = shards.get_mut(&shard_id) {
                shard_metrics.blocks_created.fetch_add(1, Ordering::Relaxed);
            }

            // Update performance metrics
            let mut metrics = performance_metrics.write().await;
            metrics.total_blocks_created += 1;
        }
    }

    /// Execute block validation task
    async fn execute_block_validation(
        _block: Block,
        shard_id: u64,
        performance_metrics: &Arc<RwLock<PerformanceMetrics>>,
        shard_distribution: &Arc<RwLock<HashMap<u64, ShardMetrics>>>,
        shard_manager: &Arc<ShardManager>,
    ) {
        // Validate block consensus
        let result = Self::validate_block_consensus().await;
        
        if let Ok(_) = result {
            // Update shard metrics
            let mut shards = shard_distribution.write().await;
            if let Some(shard_metrics) = shards.get_mut(&shard_id) {
                shard_metrics.blocks_created.fetch_add(1, Ordering::Relaxed);
            }

            // Update performance metrics
            let mut metrics = performance_metrics.write().await;
            metrics.total_blocks_validated += 1;
        }
    }

    /// Update shard metrics
    async fn update_shard_metrics(
        shard_id: u64,
        metrics: ShardMetrics,
        shard_distribution: &Arc<RwLock<HashMap<u64, ShardMetrics>>>,
    ) {
        let mut shards = shard_distribution.write().await;
        shards.insert(shard_id, metrics);
    }

    /// Calculate shard assignment for a transaction
    pub fn calculate_shard_assignment(transaction: &Transaction, total_shards: u64) -> u64 {
        let address_hash = Self::hash_address(&transaction.from);
        address_hash % total_shards
    }

    /// Hash an address for deterministic shard assignment
    fn hash_address(address: &Address) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        address.hash(&mut hasher);
        hasher.finish()
    }

    /// Submit a task to the worker pool
    pub async fn submit_task(&self, task: ProcessingTask) -> Result<(), mpsc::error::SendError<ProcessingTask>> {
        self.task_sender.send(task).await
    }

    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().await.clone()
    }

    /// Get shard distribution metrics
    pub async fn get_shard_distribution(&self) -> HashMap<u64, ShardMetrics> {
        // Since ShardMetrics can't be cloned, we'll return an empty HashMap for now
        // TODO: Implement proper cloning or return a different structure
        HashMap::new()
    }

    /// Get worker count
    pub fn get_worker_count(&self) -> usize {
        self.worker_pools.len()
    }
}

// Simulated functions for transaction processing and block operations
impl ParallelProcessor {
    /// Process a transaction in a specific shard
    async fn process_transaction_in_shard(
        _tx: &Transaction,
        shard_id: u64,
        _shard_manager: &Arc<ShardManager>,
    ) -> Result<(), String> {
        // Simulate transaction processing
        tokio::time::sleep(Duration::from_micros(100)).await;
        
        if shard_id % 100 == 0 {
            println!("Processing transaction in shard {}", shard_id);
        }
        
        Ok(())
    }

    /// Create a block in a specific shard
    async fn create_shard_block(
        shard_id: u64,
        transactions: Vec<Transaction>,
        _previous_hash: Hash,
        height: u64,
        _shard_manager: &Arc<ShardManager>,
    ) -> Result<Block, String> {
        // Simulate block creation
        tokio::time::sleep(Duration::from_micros(500)).await;
        
        if shard_id % 100 == 0 {
            println!("Creating block {} in shard {} with {} transactions", height, shard_id, transactions.len());
        }

        // Return a mock block (in real implementation, this would create an actual block)
        use crate::ledger::block::{BlockHeader, Block};
        use crate::ledger::block::BlsPublicKey;
        
        let header = BlockHeader {
            previous_hash: Hash::default(),
            merkle_root: Hash::default(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            height,
            producer: BlsPublicKey::default(),
            nonce: 0,
            difficulty: 1000000,
        };

        Ok(Block {
            header,
            transactions: vec![],
            signature: None,
        })
    }

    /// Validate block consensus
    async fn validate_block_consensus() -> Result<(), String> {
        // Simulate consensus validation
        tokio::time::sleep(Duration::from_micros(200)).await;
        
        // Simulate 99.9% success rate
        if rand::random::<f64>() < 0.999 {
            Ok(())
        } else {
            Err("Consensus validation failed".to_string())
        }
    }
}
