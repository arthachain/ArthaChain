use anyhow::Result;
use blockchain_node::consensus::reputation::{ReputationConfig, ReputationManager};
use blockchain_node::execution::parallel::{
    ConflictStrategy, ParallelConfig, ParallelExecutionManager,
};
use blockchain_node::ledger::state::storage::StateStorage;
use blockchain_node::ledger::state::tree::StateTree;
use blockchain_node::transaction::Transaction;
use criterion::{criterion_group, criterion_main, Criterion};
use log::info;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Performance benchmark suite
pub struct PerformanceBenchmark {
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
        let reputation_config = ReputationConfig {
            min_reputation: 0.3,
            initial_reputation: 0.5,
            max_adjustment: 0.1,
            decay_factor: 0.99,
            decay_interval_secs: 3600,
        };
        let reputation = Arc::new(ReputationManager::new(reputation_config));
        let state_tree = Arc::new(StateTree::new());
        let storage = Arc::new(StateStorage::new());

        // Configure parallel execution
        let parallel_config = ParallelConfig {
            max_parallel: 4,
            max_group_size: 10,
            conflict_strategy: ConflictStrategy::Retry,
            execution_timeout: 5000,
            retry_attempts: 3,
            enable_work_stealing: true,
            enable_simd: true,
            worker_threads: 0, // Auto
            simd_batch_size: 32,
            memory_pool_size: 1024 * 1024 * 256, // 256MB pre-allocated memory
        };

        let parallel = Arc::new(Mutex::new(ParallelExecutionManager::new(
            parallel_config,
            state_tree.clone(),
            Arc::new(blockchain_node::execution::executor::TransactionExecutor::new()),
        )));

        Ok(Self {
            parallel,
            state_tree,
            storage,
            reputation,
        })
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

        info!(
            "Parallel execution of {} transactions took {:?}",
            num_transactions, duration
        );

        Ok(())
    }

    /// Run all benchmarks
    pub async fn run_all_benchmarks(&self) -> Result<()> {
        info!("Running performance benchmarks...");

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

    group.bench_function("parallel_execution_100", |b| {
        b.iter(|| {
            rt.block_on(async {
                let benchmark = PerformanceBenchmark::new().await.unwrap();
                benchmark.benchmark_parallel_execution(100).await.unwrap();
            })
        })
    });

    group.bench_function("parallel_execution_1000", |b| {
        b.iter(|| {
            rt.block_on(async {
                let benchmark = PerformanceBenchmark::new().await.unwrap();
                benchmark.benchmark_parallel_execution(1000).await.unwrap();
            })
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
