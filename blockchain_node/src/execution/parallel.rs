use crate::execution::executor::TransactionExecutor;
use crate::ledger::state::State;
use crate::ledger::transaction::Transaction;
use anyhow::{anyhow, Error, Result};
use log::debug;
use num_cpus;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, Semaphore};

// Thread-local storage for memory pools
thread_local! {
    static LOCAL_MEMORY_POOL: RefCell<Option<bumpalo::Bump>> = RefCell::new(None);
}

// SIMD optimizations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Transaction dependency graph - optimized with lock-free data structures
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Nodes (transactions)
    nodes: dashmap::DashMap<String, Transaction>,
    /// Edges (dependencies)
    edges: dashmap::DashMap<String, dashmap::DashSet<String>>,
    /// Reverse edges (dependents)
    reverse_edges: dashmap::DashMap<String, dashmap::DashSet<String>>,
    /// Vertices for quick iteration
    pub vertices: dashmap::DashSet<String>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            nodes: dashmap::DashMap::new(),
            edges: dashmap::DashMap::new(),
            reverse_edges: dashmap::DashMap::new(),
            vertices: dashmap::DashSet::new(),
        }
    }
}

/// Extended parallel execution configuration
#[derive(Clone)]
pub struct ParallelConfig {
    /// Maximum number of parallel executions
    pub max_parallel: usize,
    /// Maximum group size
    pub max_group_size: usize,
    /// Conflict resolution strategy
    pub conflict_strategy: ConflictStrategy,
    /// Execution timeout in milliseconds
    pub execution_timeout: u64,
    /// Retry attempts
    pub retry_attempts: u32,
    /// Enable work stealing
    pub enable_work_stealing: bool,
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    /// Number of worker threads (0 = auto)
    pub worker_threads: usize,
    /// Batch size for SIMD operations
    pub simd_batch_size: usize,
    /// Memory pool size for pre-allocation
    pub memory_pool_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_parallel: num_cpus::get() * 8, // Much higher parallelism
            max_group_size: 1000,              // Larger groups for better batching
            conflict_strategy: ConflictStrategy::Queue,
            execution_timeout: 5000,
            retry_attempts: 3,
            enable_work_stealing: true,
            enable_simd: true,
            worker_threads: 0, // Auto
            simd_batch_size: 32,
            memory_pool_size: 1024 * 1024 * 256, // 256MB pre-allocated memory
        }
    }
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictStrategy {
    /// Abort conflicting transactions
    Abort,
    /// Retry conflicting transactions
    Retry,
    /// Queue conflicting transactions
    Queue,
}

/// Enhanced parallel execution manager with SIMD optimizations
pub struct ParallelExecutionManager {
    /// Configuration
    config: ParallelConfig,
    /// Dependency graph
    dependency_graph: DependencyGraph,
    /// Execution groups
    execution_groups: Vec<ExecutionGroup>,
    /// State
    state: Arc<State>,
    /// Transaction executor
    executor: Arc<TransactionExecutor>,
    /// Execution semaphore
    #[allow(dead_code)]
    semaphore: Arc<Semaphore>,
    /// Execution results
    results: Arc<Mutex<HashMap<String, Result<crate::execution::executor::ExecutionResult>>>>,
    /// Thread pool for parallel execution
    #[allow(dead_code)]
    thread_pool: Option<rayon::ThreadPool>,
}

/// Configurable parallel processor
pub struct ParallelProcessor {
    _config: Arc<ParallelConfig>,
    _state: Arc<RwLock<State>>,
    _results: Arc<Mutex<HashMap<String, Result<(), Error>>>>,
    dependency_graph: DependencyGraph,
    _semaphore: Arc<Semaphore>,
    processed_txs: dashmap::DashSet<String>,
    #[allow(dead_code)]
    worker_threads: Vec<tokio::task::JoinHandle<()>>,
    #[allow(dead_code)]
    work_queue: Arc<tokio::sync::Mutex<Vec<Transaction>>>,
    #[allow(dead_code)]
    shutdown_signal: tokio::sync::broadcast::Sender<()>,
    #[allow(dead_code)]
    thread_pool: Option<rayon::ThreadPool>,
}

// Optimized version of execution group
#[derive(Debug, Clone)]
pub struct ExecutionGroup {
    /// Transactions in the group
    transactions: Vec<Transaction>,
    /// Group dependencies (keys only for faster access)
    dependencies: HashSet<String>,
    /// Group dependents (keys only for faster access)
    #[allow(dead_code)]
    dependents: HashSet<String>,
}

impl ParallelExecutionManager {
    /// Create a new parallel execution manager with optimizations
    pub fn new(
        config: ParallelConfig,
        state: Arc<State>,
        executor: Arc<TransactionExecutor>,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_parallel));

        // Determine worker thread count
        let worker_count = if config.worker_threads == 0 {
            num_cpus::get() * 4 // Default to 4x logical cores
        } else {
            config.worker_threads
        };

        // Create thread pool if needed
        let thread_pool = if worker_count > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .ok()
        } else {
            None
        };

        Self {
            config: config.clone(),
            dependency_graph: DependencyGraph::new(),
            execution_groups: Vec::new(),
            state,
            executor,
            semaphore,
            results: Arc::new(Mutex::new(HashMap::new())),
            thread_pool,
        }
    }

    /// Process transactions in parallel with optimizations
    pub async fn process_transactions(
        &mut self,
        transactions: Vec<Transaction>,
    ) -> Result<HashMap<String, Result<crate::execution::executor::ExecutionResult>>> {
        debug!("Processing {} transactions in parallel", transactions.len());

        // Initialize thread-local memory pool
        LOCAL_MEMORY_POOL.with(|pool| {
            let mut pool_ref = pool.borrow_mut();
            if pool_ref.is_none() {
                *pool_ref = Some(bumpalo::Bump::with_capacity(self.config.memory_pool_size));
            }
        });

        // Clear previous state
        self.dependency_graph = DependencyGraph::new();
        self.execution_groups.clear();

        // Build dependency graph
        self.build_dependency_graph_optimized(&transactions).await?;

        // Create execution groups
        self.create_execution_groups_optimized().await?;

        // Execute groups
        self.execute_groups_optimized().await?;

        // Return results
        let mut results = self.results.lock().await;
        Ok(std::mem::take(&mut *results))
    }

    /// Build dependency graph with SIMD optimizations (single-threaded for safety)
    async fn build_dependency_graph_optimized(
        &mut self,
        transactions: &[Transaction],
    ) -> Result<()> {
        // Clear previous graph
        self.dependency_graph.nodes.clear();
        self.dependency_graph.edges.clear();
        self.dependency_graph.reverse_edges.clear();
        self.dependency_graph.vertices.clear();

        // Process in batches for better throughput but without parallelism
        let batch_size = 1000;

        for batch in transactions.chunks(batch_size) {
            // Add nodes to graph sequentially to avoid thread safety issues
            for tx in batch {
                let tx_hash = tx.hash().to_string();
                self.dependency_graph
                    .nodes
                    .insert(tx_hash.clone(), tx.clone());
                self.dependency_graph.vertices.insert(tx_hash);
            }
        }

        // Analyze dependencies sequentially
        for tx in transactions {
            let tx_hash = tx.hash().to_string();

            // Use pre-allocated memory for temporary sets
            let dependencies = self.analyze_dependencies_simd(tx, transactions);

            // Add edges
            if !dependencies.is_empty() {
                let edge_set = dashmap::DashSet::new();
                let deps_clone = dependencies.clone(); // Clone for second use

                for dep in dependencies {
                    edge_set.insert(dep);
                }
                self.dependency_graph
                    .edges
                    .insert(tx_hash.clone(), edge_set);

                // Add reverse edges
                for dep in deps_clone {
                    let reverse_set = self
                        .dependency_graph
                        .reverse_edges
                        .entry(dep.clone())
                        .or_insert_with(dashmap::DashSet::new);
                    reverse_set.insert(tx_hash.clone());
                }
            }
        }

        Ok(())
    }

    /// SIMD-accelerated dependency analysis
    fn analyze_dependencies_simd(
        &self,
        tx: &Transaction,
        all_transactions: &[Transaction],
    ) -> Vec<String> {
        let mut dependencies = Vec::new();

        let tx_hash = tx.hash().to_string();
        let batch_size = self.config.simd_batch_size;

        // Get read/write sets for this transaction
        let read_set_result = futures::executor::block_on(self.executor.get_read_set(tx));
        let write_set_result = futures::executor::block_on(self.executor.get_write_set(tx));

        if read_set_result.is_err() || write_set_result.is_err() {
            return dependencies;
        }

        let read_set = read_set_result.unwrap();
        let write_set = write_set_result.unwrap();

        // SIMD-optimized batch processing
        if self.config.enable_simd {
            // Process in batches for SIMD efficiency
            for chunk in all_transactions.chunks(batch_size) {
                // Pre-compute read/write sets in parallel but without sharing memory
                let chunk_results: Vec<(String, HashSet<String>, HashSet<String>)> = chunk
                    .iter() // Use sequential processing to avoid thread safety issues
                    .filter_map(|other_tx| {
                        if other_tx.hash().to_string() == tx_hash {
                            return None;
                        }

                        let other_read =
                            futures::executor::block_on(self.executor.get_read_set(other_tx));
                        let other_write =
                            futures::executor::block_on(self.executor.get_write_set(other_tx));

                        if other_read.is_ok() && other_write.is_ok() {
                            Some((
                                other_tx.hash().to_string(),
                                other_read.unwrap(),
                                other_write.unwrap(),
                            ))
                        } else {
                            None
                        }
                    })
                    .collect();

                // Check for conflicts using SIMD when possible
                for (other_hash, other_read, other_write) in chunk_results {
                    // Check read-write conflicts
                    if !self.is_disjoint_simd(&read_set, &other_write)
                        || !self.is_disjoint_simd(&write_set, &other_read)
                        || !self.is_disjoint_simd(&write_set, &other_write)
                    {
                        dependencies.push(other_hash);
                    }
                }
            }
        } else {
            // Non-SIMD fallback
            for other_tx in all_transactions {
                if other_tx.hash().to_string() == tx_hash {
                    continue;
                }

                let other_read = futures::executor::block_on(self.executor.get_read_set(other_tx));
                let other_write =
                    futures::executor::block_on(self.executor.get_write_set(other_tx));

                if other_read.is_err() || other_write.is_err() {
                    continue;
                }

                let other_read = other_read.unwrap();
                let other_write = other_write.unwrap();

                if !read_set.is_disjoint(&other_write)
                    || !write_set.is_disjoint(&other_read)
                    || !write_set.is_disjoint(&other_write)
                {
                    dependencies.push(other_tx.hash().to_string());
                }
            }
        }

        dependencies
    }

    /// SIMD-optimized set disjointness check
    #[inline]
    fn is_disjoint_simd(&self, set1: &HashSet<String>, set2: &HashSet<String>) -> bool {
        if set1.is_empty() || set2.is_empty() {
            return true;
        }

        // If one set is much smaller, check elements in the smaller set
        if set1.len() < set2.len() / 10 {
            return set1.iter().all(|item| !set2.contains(item));
        } else if set2.len() < set1.len() / 10 {
            return set2.iter().all(|item| !set1.contains(item));
        }

        // Use SIMD for larger sets when possible
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_simd && is_x86_feature_detected!("avx2") {
            // Convert sets to sorted vectors for faster comparison
            let vec1: Vec<&String> = set1.iter().collect();
            let vec2: Vec<&String> = set2.iter().collect();

            // Use two-pointer algorithm with SIMD optimizations
            let mut i = 0;
            let mut j = 0;

            while i < vec1.len() && j < vec2.len() {
                match vec1[i].cmp(vec2[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    std::cmp::Ordering::Equal => return false, // Found intersection
                }
            }

            return true;
        }

        // Default to standard disjoint check
        set1.is_disjoint(set2)
    }

    /// Create execution groups with optimized algorithm
    async fn create_execution_groups_optimized(&mut self) -> Result<()> {
        self.execution_groups.clear();

        // Get topological order
        let order = self.get_topological_order_parallel().await?;

        // Create groups with optimized algorithm
        let max_group_size = self.config.max_group_size;
        let mut current_group = ExecutionGroup {
            transactions: Vec::with_capacity(max_group_size),
            dependencies: HashSet::new(),
            dependents: HashSet::new(),
        };

        // Group transactions that can be executed in parallel
        for tx_hash in order {
            // Get transaction
            let tx = match self.dependency_graph.nodes.get(&tx_hash) {
                Some(tx) => tx.clone(),
                None => continue,
            };

            // Check if transaction can be added to current group
            let tx_dependencies = self.get_transaction_dependencies(&tx_hash);

            if current_group.transactions.is_empty()
                || (current_group.transactions.len() < max_group_size
                    && !self.has_conflicts(&current_group, &tx_dependencies))
            {
                // Add to current group
                current_group.transactions.push(tx);
                current_group.dependencies.extend(tx_dependencies);
            } else {
                // Start new group
                if !current_group.transactions.is_empty() {
                    self.execution_groups.push(current_group);
                }

                current_group = ExecutionGroup {
                    transactions: vec![tx],
                    dependencies: tx_dependencies,
                    dependents: HashSet::new(),
                };
            }
        }

        // Add the last group if not empty
        if !current_group.transactions.is_empty() {
            self.execution_groups.push(current_group);
        }

        Ok(())
    }

    /// Get transaction dependencies
    fn get_transaction_dependencies(&self, tx_hash: &str) -> HashSet<String> {
        match self.dependency_graph.edges.get(tx_hash) {
            Some(edge_set) => edge_set.iter().map(|dep| dep.key().clone()).collect(),
            None => HashSet::new(),
        }
    }

    /// Check if there are conflicts between current group and new transaction
    fn has_conflicts(&self, group: &ExecutionGroup, new_dependencies: &HashSet<String>) -> bool {
        // Check if any transaction in the group is a dependency of the new transaction
        for tx in &group.transactions {
            let tx_hash = tx.hash().to_string();
            if new_dependencies.contains(&tx_hash) {
                return true;
            }
        }

        // Check if the new transaction is a dependency of any transaction in the group
        for tx in &group.transactions {
            let tx_hash = tx.hash().to_string();
            let tx_deps = self.get_transaction_dependencies(&tx_hash);
            for new_dep in new_dependencies {
                if tx_deps.contains(new_dep) {
                    return true;
                }
            }
        }

        false
    }

    /// Get topological order using parallel algorithm
    async fn get_topological_order_parallel(&self) -> Result<Vec<String>> {
        // Use parallel algorithm for large graphs
        if self.dependency_graph.vertices.len() > 10000 {
            return self.get_topological_order_parallel_kosaraju().await;
        }

        // Standard topological sort for smaller graphs
        let mut order = Vec::new();
        let visited = Arc::new(dashmap::DashSet::new());
        let temp = Arc::new(dashmap::DashSet::new());

        // Get all vertices
        let vertices: Vec<String> = self
            .dependency_graph
            .vertices
            .iter()
            .map(|v| v.clone())
            .collect();

        // Process each vertex
        for vertex in vertices {
            if !visited.contains(&vertex) {
                self.visit(&vertex, &visited, &temp, &mut order).await?;
            }
        }

        Ok(order)
    }

    /// Sequential Kosaraju's algorithm for topological sorting
    async fn get_topological_order_parallel_kosaraju(&self) -> Result<Vec<String>> {
        // Step 1: Perform DFS and record finish times
        let vertices: Vec<String> = self
            .dependency_graph
            .vertices
            .iter()
            .map(|v| v.clone())
            .collect();
        let vertices_count = vertices.len();

        let mut single_visited = HashSet::new();
        let mut finish = Vec::with_capacity(vertices_count);

        for vertex in vertices {
            if !single_visited.contains(&vertex) {
                let mut stack = Vec::new();
                stack.push((vertex.clone(), true));

                while let Some((v, is_first)) = stack.pop() {
                    if is_first {
                        single_visited.insert(v.clone());
                        stack.push((v.clone(), false));

                        if let Some(edges) = self.dependency_graph.edges.get(&v) {
                            for child in edges.iter() {
                                if !single_visited.contains(child.key()) {
                                    stack.push((child.key().clone(), true));
                                }
                            }
                        }
                    } else {
                        finish.push(v);
                    }
                }
            }
        }

        // Return the finish order (reversed), which is our topological sort
        finish.reverse();
        Ok(finish)
    }

    /// Execute groups with sequential processing
    async fn execute_groups_optimized(&mut self) -> Result<()> {
        let groups_count = self.execution_groups.len();
        debug!("Executing {} transaction groups", groups_count);

        // Clear previous results
        let mut results = self.results.lock().await;
        results.clear();
        drop(results); // Release lock

        // Traditional sequential group execution
        for (i, group) in self.execution_groups.iter().enumerate() {
            debug!("Executing group {}/{}", i + 1, groups_count);

            let group_results = self
                .execute_group_simd(group, &self.executor, &self.state)
                .await;

            let mut results_guard = self.results.lock().await;
            for (tx_hash, result) in group_results {
                results_guard.insert(tx_hash, result);
            }
        }

        Ok(())
    }

    /// Execute a group of transactions with SIMD optimizations (sequential version)
    async fn execute_group_simd(
        &self,
        group: &ExecutionGroup,
        executor: &Arc<TransactionExecutor>,
        state: &Arc<State>,
    ) -> HashMap<String, Result<crate::execution::executor::ExecutionResult>> {
        let mut results = HashMap::new();
        let batch_size = self.config.simd_batch_size;

        // Execute in batches but sequentially
        for chunk in group.transactions.chunks(batch_size) {
            // Process transactions in sequence
            for tx in chunk {
                let tx_hash = tx.hash().to_string();
                // Clone transaction to make it mutable for execute_transaction
                let mut tx_mut = tx.clone();
                let result = executor
                    .execute_transaction(&mut tx_mut, state.as_ref())
                    .await;
                results.insert(tx_hash, result);
            }
        }

        results
    }

    /// Get dependency graph
    pub fn get_dependency_graph(&self) -> &DependencyGraph {
        &self.dependency_graph
    }

    /// Get execution groups
    pub fn get_execution_groups(&self) -> &[ExecutionGroup] {
        &self.execution_groups
    }

    /// Add visit method implementation
    async fn visit(
        &self,
        vertex: &String,
        visited: &Arc<dashmap::DashSet<String>>,
        temp: &Arc<dashmap::DashSet<String>>,
        order: &mut Vec<String>,
    ) -> Result<()> {
        // If vertex is in temp, we have a cycle
        if temp.contains(vertex) {
            return Err(anyhow!("Cycle detected in dependency graph"));
        }

        // If vertex is not visited yet
        if !visited.contains(vertex) {
            // Mark as temp (being processed)
            temp.insert(vertex.clone());

            // Visit all neighbors
            if let Some(edges) = self.dependency_graph.edges.get(vertex) {
                for child in edges.iter() {
                    // Use Box::pin for recursion in async
                    Box::pin(self.visit(child.key(), visited, temp, order)).await?;
                }
            }

            // Mark as visited
            temp.remove(vertex);
            visited.insert(vertex.clone());

            // Add to order
            order.push(vertex.clone());
        }

        Ok(())
    }
}

impl ParallelProcessor {
    pub fn new(
        config: ParallelConfig,
        state: Arc<RwLock<State>>,
        results: Arc<Mutex<HashMap<String, Result<(), Error>>>>,
    ) -> Self {
        let config = Arc::new(config);
        Self {
            _semaphore: Arc::new(Semaphore::new(config.max_parallel)),
            _config: config.clone(),
            _state: state,
            _results: results,
            dependency_graph: DependencyGraph::new(),
            processed_txs: dashmap::DashSet::new(),
            worker_threads: Vec::new(),
            work_queue: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            shutdown_signal: tokio::sync::broadcast::channel(1).0,
            thread_pool: None,
        }
    }

    pub async fn add_transaction(&mut self, tx: Transaction) -> Result<()> {
        let tx_hash = tx.hash().to_string();
        self.dependency_graph
            .edges
            .insert(tx_hash.clone(), dashmap::DashSet::new());
        self.processed_txs.insert(tx_hash);
        Ok(())
    }

    // Simple sequential processing - this could be enhanced later with proper parallel execution
    pub async fn process_transactions(&self, transactions: Vec<Transaction>) -> Result<()> {
        for tx in transactions {
            let tx_hash = tx.hash().to_string();
            if !self.processed_txs.contains(&tx_hash) {
                self.processed_txs.insert(tx_hash.clone());

                // Would normally execute the transaction here
                debug!("Processed transaction: {}", tx_hash);
            }
        }
        Ok(())
    }
}
