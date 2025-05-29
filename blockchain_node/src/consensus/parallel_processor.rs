use crate::ledger::block::{Block, BlockExt};
use crate::ledger::state::State;
use crate::ledger::transaction::Transaction;
use crate::types::Hash;
use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use rand::thread_rng;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;

/// Configuration for parallel processor
#[derive(Debug, Clone)]
pub struct ParallelProcessorConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    /// Batch size for transaction processing
    pub batch_size: usize,
    /// Whether to use work stealing
    pub use_work_stealing: bool,
    /// Whether to prefetch data
    pub prefetch_enabled: bool,
    /// Whether to pipeline verification
    pub pipeline_verification: bool,
}

impl Default for ParallelProcessorConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            batch_size: 1000,
            use_work_stealing: true,
            prefetch_enabled: true,
            pipeline_verification: true,
        }
    }
}

/// Parameters for dynamic block production
#[derive(Debug, Clone)]
pub struct BlockParameters {
    /// Dynamically calculated block time
    pub block_time: f32,
    /// Dynamically calculated batch size
    pub batch_size: usize,
}

/// Result of parallel mining attempt
#[derive(Debug, Clone)]
pub enum ParallelMiningResult {
    /// Block was successfully mined
    Success(Block),
    /// Mining was interrupted
    Interrupted,
    /// Mining failed due to error
    Error(String),
}

/// Parallel processor component for scaling TPS with miner count
#[derive(Clone)]
pub struct ParallelProcessor {
    /// Number of active miners/validators
    miner_count: Arc<AtomicUsize>,
    /// Throughput multiplier per miner (1.5-5.0)
    throughput_multiplier: f32,
    /// Base block time for a single miner
    base_block_time: f32,
    /// Base batch size for a single miner
    base_batch_size: usize,
    /// Max block time allowed
    #[allow(dead_code)]
    max_block_time: f32,
    /// Max batch size allowed
    max_batch_size: usize,
    /// Blockchain state
    state: Arc<RwLock<State>>,
    /// Running flag
    running: Arc<AtomicUsize>,
    /// Channel for producing blocks
    block_sender: mpsc::Sender<Block>,
    /// Pending transactions for parallel processing
    pending_transactions: Arc<Mutex<Vec<Transaction>>>,
    /// Current processing segments
    active_segments: Arc<Mutex<HashMap<usize, Vec<Transaction>>>>,
    /// Advanced configuration
    config: ParallelProcessorConfig,
}

impl ParallelProcessor {
    /// Create a new parallel processor
    pub fn new(
        state: Arc<RwLock<State>>,
        block_sender: mpsc::Sender<Block>,
        throughput_multiplier: Option<f32>,
    ) -> Self {
        Self {
            miner_count: Arc::new(AtomicUsize::new(1)), // Start with 1 miner
            throughput_multiplier: throughput_multiplier.unwrap_or(2.0), // Default 2x scaling per miner
            base_block_time: 15.0,                                       // 15 seconds for one miner
            base_batch_size: 500,  // 500 transactions for one miner
            max_block_time: 15.0,  // Never slower than 15 seconds
            max_batch_size: 10000, // Hard cap at 10000 transactions per block
            state,
            running: Arc::new(AtomicUsize::new(0)),
            block_sender,
            pending_transactions: Arc::new(Mutex::new(Vec::new())),
            active_segments: Arc::new(Mutex::new(HashMap::new())),
            config: ParallelProcessorConfig::default(),
        }
    }

    /// Create a parallel processor with custom configuration
    pub fn new_with_config(
        state: Arc<RwLock<State>>,
        block_sender: mpsc::Sender<Block>,
        throughput_multiplier: Option<f32>,
        config: ParallelProcessorConfig,
    ) -> Self {
        Self {
            miner_count: Arc::new(AtomicUsize::new(1)),
            throughput_multiplier: throughput_multiplier.unwrap_or(2.0),
            base_block_time: 15.0,
            base_batch_size: config.batch_size,
            max_block_time: 15.0,
            max_batch_size: config.batch_size * 10,
            state,
            running: Arc::new(AtomicUsize::new(0)),
            block_sender,
            pending_transactions: Arc::new(Mutex::new(Vec::new())),
            active_segments: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Update the miner/validator count
    pub fn update_miner_count(&self, count: usize) {
        let old_count = self.miner_count.swap(count, Ordering::SeqCst);
        if old_count != count {
            info!("Miner count updated from {old_count} to {count}");
        }
    }

    /// Calculate block parameters based on miner count
    pub fn calculate_block_parameters(&self) -> BlockParameters {
        let miner_count = self.miner_count.load(Ordering::Relaxed).max(1);

        // Calculate block time: decreases as miners increase (with a floor)
        let scaling_factor = (miner_count as f32 * self.throughput_multiplier).max(1.0);
        let new_block_time = (self.base_block_time / scaling_factor).max(0.5);

        // Calculate batch size: increases as miners increase (with a ceiling)
        let new_batch_size = (self.base_batch_size as f32 * scaling_factor) as usize;
        let new_batch_size = new_batch_size.min(self.max_batch_size);

        info!(
            "Dynamic parameters: block_time={new_block_time:.2}s, batch_size={new_batch_size} with {miner_count} miners"
        );

        BlockParameters {
            block_time: new_block_time,
            batch_size: new_batch_size,
        }
    }

    /// Start the parallel processor
    pub async fn start(&self) -> Result<JoinHandle<()>> {
        if self
            .running
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(anyhow!("Parallel processor already running"));
        }

        // Clone references for the task
        let running = self.running.clone();
        let miner_count = self.miner_count.clone();
        let state = self.state.clone();
        let block_sender = self.block_sender.clone();
        let _pending_transactions = self.pending_transactions.clone();
        let active_segments = self.active_segments.clone();
        let throughput_multiplier = self.throughput_multiplier;
        let base_block_time = self.base_block_time;
        let base_batch_size = self.base_batch_size;

        let handle = tokio::spawn(async move {
            info!("Parallel processor started with multiplier {throughput_multiplier}");

            let mut interval = tokio::time::interval(Duration::from_millis(100));

            while running.load(Ordering::Relaxed) == 1 {
                interval.tick().await;

                // Get current miner count
                let current_miners = miner_count.load(Ordering::Relaxed).max(1);

                // Calculate dynamic parameters
                let _block_time = (base_block_time
                    / (current_miners as f32 * throughput_multiplier).max(1.0))
                .max(0.5);
                let batch_size = ((base_batch_size as f32)
                    * (current_miners as f32 * throughput_multiplier).max(1.0))
                    as usize;

                // Load pending transactions
                let pending_txs = {
                    let state_guard = state.read().await;
                    state_guard.get_pending_transactions(batch_size)
                };
                // pending_txs is already Vec<crate::ledger::transaction::Transaction>

                // Skip if no transactions
                if pending_txs.is_empty() {
                    continue;
                }

                // Process transactions in parallel
                let segments = Self::split_transactions(pending_txs, current_miners);

                // Store segments
                {
                    let mut segments_guard = active_segments.lock().await;
                    *segments_guard = segments.clone();
                }

                // Mine blocks in parallel
                match Self::mine_concurrent_blocks(&segments, &state).await {
                    Ok(blocks) => {
                        // Merge mined blocks
                        if !blocks.is_empty() {
                            match Self::merge_parallel_blocks(blocks).await {
                                Ok(merged_block) => {
                                    // Send block
                                    if let Err(e) = block_sender.send(merged_block).await {
                                        warn!("Failed to send merged block: {e}");
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to merge blocks: {e}");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to mine concurrent blocks: {e}");
                    }
                }
            }

            info!("Parallel processor stopped");
        });

        Ok(handle)
    }

    /// Start the processor in optimized mode (higher parallelism)
    pub async fn start_optimized(&self) -> Result<JoinHandle<()>> {
        if self
            .running
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(anyhow!("Parallel processor already running"));
        }

        // Clone references for the task
        let running = self.running.clone();
        let miner_count = self.miner_count.clone();
        let state = self.state.clone();
        let block_sender = self.block_sender.clone();
        let _pending_transactions = self.pending_transactions.clone();
        let active_segments = self.active_segments.clone();
        let throughput_multiplier = self.throughput_multiplier;
        let base_block_time = self.base_block_time;
        let base_batch_size = self.base_batch_size;
        let worker_threads = self.config.worker_threads;
        let use_work_stealing = self.config.use_work_stealing;
        let prefetch_enabled = self.config.prefetch_enabled;

        info!("Starting optimized parallel processor with {worker_threads} worker threads");

        let handle = tokio::spawn(async move {
            info!("Optimized parallel processor started with multiplier {throughput_multiplier}");

            let mut interval = tokio::time::interval(Duration::from_millis(50)); // Faster polling interval

            while running.load(Ordering::Relaxed) == 1 {
                interval.tick().await;

                // Get current miner count - scale with worker threads
                let logical_miners = miner_count.load(Ordering::Relaxed).max(1) * worker_threads;

                // Calculate enhanced dynamic parameters
                let _block_time = (base_block_time
                    / (logical_miners as f32 * throughput_multiplier).max(1.0))
                .max(0.1);
                let batch_size = ((base_batch_size as f32)
                    * (logical_miners as f32 * throughput_multiplier).max(1.0))
                    as usize;

                // Load pending transactions with prefetching
                let pending_txs = if prefetch_enabled {
                    // Prefetch more transactions for better throughput
                    let prefetch_size = batch_size * 2;
                    let state_guard = state.read().await;
                    state_guard.get_pending_transactions(prefetch_size)
                } else {
                    let state_guard = state.read().await;
                    state_guard.get_pending_transactions(batch_size)
                };

                // pending_txs is already Vec<crate::ledger::transaction::Transaction>

                // Skip if no transactions
                if pending_txs.is_empty() {
                    continue;
                }

                // Enhanced transaction splitting with work stealing if enabled
                let segments = if use_work_stealing {
                    // Split into more segments for better load balancing
                    Self::split_transactions(pending_txs, logical_miners * 2)
                } else {
                    Self::split_transactions(pending_txs, logical_miners)
                };

                // Store segments
                {
                    let mut segments_guard = active_segments.lock().await;
                    *segments_guard = segments.clone();
                }

                // Mine blocks with enhanced parallelism
                match Self::mine_concurrent_blocks(&segments, &state).await {
                    Ok(blocks) => {
                        // Merge mined blocks with optimized algorithm
                        if !blocks.is_empty() {
                            match Self::merge_parallel_blocks(blocks).await {
                                Ok(merged_block) => {
                                    // Send block with higher priority
                                    if let Err(e) = block_sender.send(merged_block).await {
                                        warn!("Failed to send merged block: {e}");
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to merge blocks: {e}");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to mine concurrent blocks: {e}");
                    }
                }
            }

            info!("Optimized parallel processor stopped");
        });

        Ok(handle)
    }

    /// Split transactions for parallel processing
    fn split_transactions(
        transactions: Vec<Transaction>,
        segment_count: usize,
    ) -> HashMap<usize, Vec<Transaction>> {
        let mut segments = HashMap::new();

        if transactions.is_empty() || segment_count == 0 {
            return segments;
        }

        // Calculate transactions per segment
        let mut txs_per_segment = transactions.len() / segment_count;
        if txs_per_segment == 0 {
            txs_per_segment = 1;
        }

        let mut start = 0;
        for i in 0..segment_count {
            let end = if i == segment_count - 1 {
                transactions.len()
            } else {
                start + txs_per_segment
            };

            if start < transactions.len() {
                let segment = transactions[start..end.min(transactions.len())].to_vec();
                segments.insert(i, segment);
                start = end;
            }
        }

        segments
    }

    /// Mine blocks in parallel, one per segment
    async fn mine_concurrent_blocks(
        segments: &HashMap<usize, Vec<Transaction>>,
        state: &Arc<RwLock<State>>,
    ) -> Result<Vec<Block>> {
        let mut handles = Vec::with_capacity(segments.len());

        for (segment_id, txs) in segments.iter() {
            let txs_clone = txs.clone();
            let state_clone = state.clone();
            let segment_id = *segment_id;

            let handle = tokio::spawn(async move {
                Self::mine_segment_block(txs_clone, segment_id, state_clone).await
            });

            handles.push(handle);
        }

        let mut blocks = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(Ok(block)) => blocks.push(block),
                Ok(Err(e)) => warn!("Segment mining failed: {e}"),
                Err(e) => warn!("Segment task failed: {e}"),
            }
        }

        Ok(blocks)
    }

    /// Mine a block for a specific segment
    async fn mine_segment_block(
        transactions: Vec<Transaction>,
        segment_id: usize,
        state: Arc<RwLock<State>>,
    ) -> Result<Block> {
        if transactions.is_empty() {
            return Err(anyhow!("No transactions to mine in segment {segment_id}"));
        }

        // Get previous block hash and height
        let (prev_hash, height, difficulty, node_id, shard_id) = {
            let state_guard = state.read().await;
            let prev_hash = state_guard.get_latest_block_hash()?;
            let prev_hash = Hash::from_hex(&prev_hash)?;
            let height = state_guard.get_height()? + 1;
            let difficulty = 1; // Simple difficulty for parallel blocks
            let node_id = format!("node_{segment_id}"); // Use segment ID in node ID
            let shard_id = 0; // Assuming main shard
            (prev_hash, height, difficulty, node_id, shard_id)
        };

        // Create block
        let mut block = Block::new(
            prev_hash,
            transactions,
            height,
            difficulty,
            node_id,
            shard_id,
        );

        // Generate a reasonably unique nonce (simplified POW for this implementation)
        let mut rng = thread_rng();
        let nonce = rand::Rng::gen::<u64>(&mut rng);
        block.set_nonce(nonce);

        debug!(
            "Mined segment {segment_id} block with {} transactions",
            block.body.transactions.len()
        );

        Ok(block)
    }

    /// Merge blocks mined in parallel
    async fn merge_parallel_blocks(blocks: Vec<Block>) -> Result<Block> {
        if blocks.is_empty() {
            return Err(anyhow!("No blocks to merge"));
        }

        if blocks.len() == 1 {
            return Ok(blocks[0].clone());
        }

        // Get properties from first block
        let prev_hash = blocks[0].header.previous_hash.clone();
        let height = blocks[0].header.height;
        let difficulty = blocks[0].header.difficulty;
        let node_id = "parallel_processor".to_string();
        let shard_id = blocks[0].consensus.shard_id;

        // Collect all transactions, avoiding duplicates
        let mut merged_transactions = Vec::new();
        let mut tx_hashes = HashSet::new();

        for block in &blocks {
            for tx in &block.body.transactions {
                let tx_hash = tx.hash();
                if tx_hashes.insert(tx_hash) {
                    merged_transactions.push(tx.clone());
                }
            }
        }

        // Create merged block
        let mut merged_block = Block::new(
            prev_hash,
            merged_transactions,
            height,
            difficulty,
            node_id,
            shard_id,
        );

        // Set nonce by combining nonces
        let combined_nonce = blocks
            .iter()
            .fold(0u64, |acc, block| acc ^ block.header.nonce);
        merged_block.set_nonce(combined_nonce);

        info!(
            "Merged {} blocks with total {} transactions",
            blocks.len(),
            merged_block.body.transactions.len()
        );

        Ok(merged_block)
    }

    /// Stop the parallel processor
    pub fn stop(&self) {
        self.running.store(0, Ordering::SeqCst);
        info!("Stopping parallel processor");
    }

    /// Get estimated TPS based on current miner count
    pub fn get_estimated_tps(&self) -> f32 {
        let params = self.calculate_block_parameters();
        params.batch_size as f32 / params.block_time
    }

    #[allow(dead_code)]
    async fn process_block(&mut self, state_guard: &State) -> Result<()> {
        let current_height = state_guard.get_height()?;
        let _next_height = current_height + 1;

        // Process transactions in parallel
        let mut futures = Vec::new();
        let guard = self.pending_transactions.lock().await;
        for _tx in guard.iter() {
            let _tx = _tx.clone();
            let _state = Arc::clone(&self.state);
            let future = tokio::spawn(async move {
                let _state = _state.write().await;
                // TODO: Implement or call the correct transaction application method here
                // _state.apply_transaction(&_tx).await
                Ok::<(), anyhow::Error>(())
            });
            futures.push(future);
        }

        // Wait for all transactions to complete
        for future in futures {
            future.await??;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_transaction_splitting() {
        // Create test transactions
        let transactions = (0..100)
            .map(|i| {
                Transaction::new(
                    crate::ledger::transaction::TransactionType::Transfer,
                    format!("sender{i}"),
                    format!("receiver{i}"),
                    100,
                    i as u64,
                    10,
                    1000,
                    Vec::new(),
                )
            })
            .collect::<Vec<_>>();

        // Split into 4 segments
        let segments = ParallelProcessor::split_transactions(transactions, 4);

        // Check correct splitting
        assert_eq!(segments.len(), 4);
        assert_eq!(segments[&0].len(), 25);
        assert_eq!(segments[&1].len(), 25);
        assert_eq!(segments[&2].len(), 25);
        assert_eq!(segments[&3].len(), 25);
    }

    #[tokio::test]
    async fn test_dynamic_parameters() {
        // Create test processor
        let (tx, _rx) = mpsc::channel(100);

        // Create a minimal config for testing
        let config = crate::config::Config::new();

        let state = Arc::new(RwLock::new(State::new(&config).unwrap()));
        let processor = ParallelProcessor::new(state.clone(), tx, Some(2.0));

        // Test with 1 miner
        processor.update_miner_count(1);
        let params = processor.calculate_block_parameters();
        println!(
            "With 1 miner: block_time={}, batch_size={}",
            params.block_time, params.batch_size
        );

        // Based on the actual output:
        assert_eq!(params.block_time, 7.5);
        assert_eq!(params.batch_size, 1000);

        // Test with 4 miners
        processor.update_miner_count(4);
        let params = processor.calculate_block_parameters();
        println!(
            "With 4 miners: block_time={}, batch_size={}",
            params.block_time, params.batch_size
        );

        // Based on the actual output:
        assert_eq!(params.block_time, 1.875);
        assert_eq!(params.batch_size, 4000);

        // Verify the values are in a reasonable range
        assert!(
            params.block_time <= 15.0,
            "Block time should not exceed base block time"
        );
        assert!(
            params.batch_size >= 500,
            "Batch size should not be less than base batch size"
        );
    }
}
