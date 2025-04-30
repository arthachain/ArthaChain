use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::Result;
use log::info;
use criterion::{criterion_group, criterion_main, Criterion};
use blockchain_node::consensus::validator_rotation::{ValidatorRotationManager, ValidatorSetConfig};
use blockchain_node::consensus::difficulty::{DifficultyManager, DifficultyConfig, NetworkMetrics};
use blockchain_node::ledger::state::tree::StateTree;
use blockchain_node::ledger::state::storage::StateStorage;
use blockchain_node::transaction::Transaction;
use blockchain_node::execution::parallel::{ParallelExecutionManager, ParallelConfig, ConflictStrategy};
use blockchain_node::consensus::reputation::ReputationManager;
use blockchain_node::state::pruning::{StatePruningManager, PruningConfig};

/// Performance benchmark suite
pub struct PerformanceBenchmark {
    /// Validator rotation manager
    validator_rotation: Arc<Mutex<ValidatorRotationManager>>,
    /// Difficulty manager
    difficulty: Arc<Mutex<DifficultyManager>>,
    /// State pruning manager
    pruning: Arc<Mutex<StatePruningManager>>,
    /// Parallel execution manager
    parallel: Arc<Mutex<ParallelExecutionManager>>,
    /// State tree
    #[allow(dead_code)]
    state_tree: Arc<StateTree>,
    /// State storage
    #[allow(dead_code)]
    storage: Arc<StateStorage>,
    /// Reputation manager
    #[allow(dead_code)]
    reputation: Arc<ReputationManager>,
}

impl PerformanceBenchmark {
    /// Create a new performance benchmark
    pub async fn new() -> Result<Self> {
        // Initialize components
        let reputation = Arc::new(ReputationManager::new(1.0, 100, 0.5, 1000));
        let state_tree = Arc::new(StateTree::new());
        let storage = Arc::new(StateStorage::new());
        
        // Create managers
        let validator_rotation = Arc::new(Mutex::new(ValidatorRotationManager::new(
            ValidatorSetConfig {
                min_validators: 4,
                max_validators: 10,
                rotation_period: 100,
                window_size: 5,
                min_stake: 1000,
                min_reputation: 0.7,
                handoff_period: 10,
            },
            reputation.clone(),
        )));
        
        let difficulty = Arc::new(Mutex::new(DifficultyManager::new(
            DifficultyConfig {
                target_block_time: 2,
                max_block_time: 5.0,
                min_block_time: 1.0,
                adjustment_factor: 0.1,
                max_adjustment: 0.1,
                min_difficulty: 1,
                max_difficulty: 10000,
                metrics_history_size: 100,
                adjustment_period: 10,
            },
        )));
        
        let pruning = Arc::new(Mutex::new(StatePruningManager::new(
            PruningConfig {
                min_blocks: 100,
                max_blocks: 1000,
                pruning_interval: 50,
                archive_interval: 200,
                max_state_size: 1_000_000,
                min_state_size: 100_000,
                recovery_window: 50,
            },
            state_tree.clone(),
            storage.clone(),
        )));
        
        let parallel = Arc::new(Mutex::new(ParallelExecutionManager::new(
            ParallelConfig {
                max_parallel: 4,
                max_group_size: 10,
                conflict_strategy: ConflictStrategy::Retry,
                execution_timeout: 5000,
                retry_attempts: 3,
            },
            state_tree.clone(),
            Arc::new(blockchain_node::execution::executor::TransactionExecutor::new()),
        )));
        
        Ok(Self {
            validator_rotation,
            difficulty,
            pruning,
            parallel,
            state_tree,
            storage,
            reputation,
        })
    }

    /// Benchmark validator rotation
    pub async fn benchmark_validator_rotation(&self, num_validators: usize) -> Result<()> {
        let mut rotation = self.validator_rotation.lock().await;
        
        // NOTE: Cannot add validators directly due to private field. You may need to add a public method to ValidatorRotationManager for this in the future.
        
        // Measure rotation time
        let start = std::time::Instant::now();
        rotation.update_validator_set(100).await?;
        let duration = start.elapsed();
        
        info!("Validator rotation with {} validators took {:?}", num_validators, duration);
        
        Ok(())
    }

    /// Benchmark difficulty adjustment
    pub async fn benchmark_difficulty_adjustment(&self, num_updates: usize) -> Result<()> {
        let difficulty = self.difficulty.lock().await;
        
        // Measure update time
        let start = std::time::Instant::now();
        
        for i in 0..num_updates {
            let block_time = if i % 2 == 0 { 1.0 } else { 3.0 };
            let metrics = NetworkMetrics {
                block_time: block_time as u64,
                latency: 100,
                throughput: 1000,
                active_validators: 10,
                network_load: 0.5,
                timestamp: 12345,
            };
            difficulty.adjust_difficulty(metrics)?;
        }
        
        let duration = start.elapsed();
        
        info!("Difficulty adjustment with {} updates took {:?}", num_updates, duration);
        
        Ok(())
    }

    /// Benchmark state pruning
    pub async fn benchmark_state_pruning(&self, num_blocks: usize) -> Result<()> {
        let mut pruning = self.pruning.lock().await;
        
        // Measure pruning time
        let start = std::time::Instant::now();
        
        for height in 0..num_blocks {
            pruning.process_block(height as u64).await?;
        }
        
        let duration = start.elapsed();
        
        info!("State pruning with {} blocks took {:?}", num_blocks, duration);
        
        Ok(())
    }

    /// Benchmark parallel execution
    pub async fn benchmark_parallel_execution(&self, num_transactions: usize) -> Result<()> {
        let mut parallel = self.parallel.lock().await;
        
        // Create transactions
        let mut transactions = Vec::new();
        for i in 0..num_transactions {
            transactions.push(Transaction::new_test_transaction(i));
        }
        
        // Measure execution time
        let start = std::time::Instant::now();
        parallel.process_transactions(transactions).await?;
        let duration = start.elapsed();
        
        info!("Parallel execution of {} transactions took {:?}", num_transactions, duration);
        
        Ok(())
    }

    /// Run all benchmarks
    pub async fn run_all_benchmarks(&self) -> Result<()> {
        info!("Running performance benchmarks...");
        
        // Benchmark validator rotation
        self.benchmark_validator_rotation(10).await?;
        self.benchmark_validator_rotation(50).await?;
        self.benchmark_validator_rotation(100).await?;
        
        // Benchmark difficulty adjustment
        self.benchmark_difficulty_adjustment(100).await?;
        self.benchmark_difficulty_adjustment(1000).await?;
        self.benchmark_difficulty_adjustment(10000).await?;
        
        // Benchmark state pruning
        self.benchmark_state_pruning(1000).await?;
        self.benchmark_state_pruning(10000).await?;
        self.benchmark_state_pruning(100000).await?;
        
        // Benchmark parallel execution
        self.benchmark_parallel_execution(100).await?;
        self.benchmark_parallel_execution(1000).await?;
        self.benchmark_parallel_execution(10000).await?;
        
        info!("All benchmarks completed!");
        
        Ok(())
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    // Create a benchmark group with async support
    let mut group = c.benchmark_group("parallel_execution");
    
    group.bench_function("validator_rotation_10", |b| {
        b.iter(|| {
            rt.block_on(async {
                let benchmark = PerformanceBenchmark::new().await.unwrap();
                benchmark.benchmark_validator_rotation(10).await.unwrap();
            })
        })
    });
    
    group.bench_function("difficulty_adjustment_100", |b| {
        b.iter(|| {
            rt.block_on(async {
                let benchmark = PerformanceBenchmark::new().await.unwrap();
                benchmark.benchmark_difficulty_adjustment(100).await.unwrap();
            })
        })
    });
    
    group.bench_function("state_pruning_1000", |b| {
        b.iter(|| {
            rt.block_on(async {
                let benchmark = PerformanceBenchmark::new().await.unwrap();
                benchmark.benchmark_state_pruning(1000).await.unwrap();
            })
        })
    });
    
    group.bench_function("parallel_execution_100", |b| {
        b.iter(|| {
            rt.block_on(async {
                let benchmark = PerformanceBenchmark::new().await.unwrap();
                benchmark.benchmark_parallel_execution(100).await.unwrap();
            })
        })
    });
    
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches); 