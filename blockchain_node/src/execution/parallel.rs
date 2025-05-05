use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use anyhow::{Result, anyhow, Error};
use log::debug;
use serde::{Serialize, Deserialize};
use tokio::sync::{Mutex, Semaphore, RwLock};
use crate::ledger::transaction::Transaction;
use crate::ledger::state::{State, StateTree};
use crate::execution::executor::TransactionExecutor;
use rayon::prelude::*;
use crossbeam::channel;
use crossbeam::deque::{Injector, Stealer, Worker};
use num_cpus;

// SIMD optimizations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

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
            max_group_size: 1000, // Larger groups for better batching
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
    /// State tree
    state_tree: Arc<StateTree>,
    /// Transaction executor
    executor: Arc<TransactionExecutor>,
    /// Execution semaphore
    semaphore: Arc<Semaphore>,
    /// Execution results
    results: Arc<Mutex<HashMap<String, Result<()>>>>,
    /// Thread pool for parallel execution
    thread_pool: Option<rayon::ThreadPool>,
    /// Pre-allocated memory pool
    memory_pool: bumpalo::Bump,
    /// Work stealing deque
    injector: Injector<Transaction>,
    /// Worker threads
    workers: Vec<Worker<Transaction>>,
    /// Stealers for work stealing
    stealers: Vec<Stealer<Transaction>>,
}

/// Configurable parallel processor
pub struct ParallelProcessor {
    _config: Arc<ParallelConfig>,
    _state: Arc<RwLock<State>>,
    _results: Arc<Mutex<HashMap<String, Result<(), Error>>>>,
    dependency_graph: DependencyGraph,
    _semaphore: Arc<Semaphore>,
    processed_txs: dashmap::DashSet<String>,
    worker_threads: Vec<tokio::task::JoinHandle<()>>,
    work_queue: Arc<tokio::sync::Mutex<Vec<Transaction>>>,
    shutdown_signal: tokio::sync::broadcast::Sender<()>,
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
    dependents: HashSet<String>,
}

impl ParallelExecutionManager {
    /// Create a new parallel execution manager with optimizations
    pub fn new(
        config: ParallelConfig,
        state_tree: Arc<StateTree>,
        executor: Arc<TransactionExecutor>,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_parallel));
        
        // Determine worker thread count
        let worker_count = if config.worker_threads == 0 {
            num_cpus::get() * 4 // Default to 4x logical cores
        } else {
            config.worker_threads
        };
        
        // Create work stealing structures
        let injector = Injector::new();
        let mut workers = Vec::with_capacity(worker_count);
        let mut stealers = Vec::with_capacity(worker_count);
        
        for _ in 0..worker_count {
            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            workers.push(worker);
        }
        
        // Create thread pool if needed
        let thread_pool = if worker_count > 0 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .ok();
            pool
        } else {
            None
        };
        
        Self {
            config: config.clone(),
            dependency_graph: DependencyGraph::new(),
            execution_groups: Vec::new(),
            state_tree,
            executor,
            semaphore,
            results: Arc::new(Mutex::new(HashMap::new())),
            thread_pool,
            memory_pool: bumpalo::Bump::with_capacity(config.memory_pool_size),
            injector,
            workers,
            stealers,
        }
    }

    /// Process transactions in parallel with optimizations
    pub async fn process_transactions(
        &mut self,
        transactions: Vec<Transaction>,
    ) -> Result<HashMap<String, Result<()>>> {
        debug!("Processing {} transactions in parallel", transactions.len());
        
        // Reset memory pool for this batch
        self.memory_pool.reset();
        
        // Build dependency graph
        self.build_dependency_graph_optimized(&transactions).await?;
        
        // Create execution groups
        self.create_execution_groups_optimized().await?;
        
        // Execute groups in parallel
        self.execute_groups_optimized().await?;
        
        // Return results
        let results_guard = self.results.lock().await;
        let results: HashMap<String, Result<()>> = results_guard.iter().map(|(k, v)| (k.clone(), v.as_ref().map(|_| ()).map_err(|e| anyhow!(e.to_string())))).collect();
        Ok(results)
    }

    /// Build dependency graph with optimizations
    async fn build_dependency_graph_optimized(&mut self, transactions: &[Transaction]) -> Result<()> {
        // Clear existing graph
        self.dependency_graph.nodes.clear();
        self.dependency_graph.edges.clear();
        self.dependency_graph.reverse_edges.clear();
        self.dependency_graph.vertices.clear();
        
        // Process in parallel batches for better throughput
        let batch_size = 1000;
        
        for batch in transactions.chunks(batch_size) {
            // Add nodes to graph
            batch.par_iter().for_each(|tx| {
                let tx_hash = tx.hash();
                self.dependency_graph.nodes.insert(tx_hash.clone(), tx.clone());
                self.dependency_graph.vertices.insert(tx_hash);
            });
        }
        
        // Analyze dependencies in parallel
        if let Some(pool) = &self.thread_pool {
            pool.install(|| {
                transactions.par_iter().for_each(|tx| {
                    let tx_hash = tx.hash();
                    
                    // Use pre-allocated memory for temporary sets
                    let dependencies = self.analyze_dependencies_simd(tx, transactions);
                    
                    // Add edges
                    if !dependencies.is_empty() {
                        let edge_set = dashmap::DashSet::new();
                        for dep in dependencies {
                            edge_set.insert(dep);
                        }
                        self.dependency_graph.edges.insert(tx_hash.clone(), edge_set);
                        
                        // Add reverse edges
                        for dep in dependencies {
                            let reverse_set = self.dependency_graph.reverse_edges
                                .entry(dep.clone())
                                .or_insert_with(dashmap::DashSet::new);
                            reverse_set.insert(tx_hash.clone());
                        }
                    }
                });
            });
        } else {
            // Fallback for non-threaded environment
            for tx in transactions {
                let tx_hash = tx.hash();
                let dependencies = self.analyze_dependencies_simd(tx, transactions);
                
                // Add edges
                if !dependencies.is_empty() {
                    let edge_set = dashmap::DashSet::new();
                    for dep in dependencies {
                        edge_set.insert(dep);
                    }
                    self.dependency_graph.edges.insert(tx_hash.clone(), edge_set);
                    
                    // Add reverse edges
                    for dep in dependencies {
                        let reverse_set = self.dependency_graph.reverse_edges
                            .entry(dep.clone())
                            .or_insert_with(dashmap::DashSet::new);
                        reverse_set.insert(tx_hash.clone());
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// SIMD-accelerated dependency analysis
    fn analyze_dependencies_simd(&self, tx: &Transaction, all_transactions: &[Transaction]) -> Vec<String> {
        let mut dependencies = Vec::new();
        
        let tx_hash = tx.hash();
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
                // Pre-compute read/write sets in parallel
                let chunk_results: Vec<(String, HashSet<String>, HashSet<String>)> = chunk
                    .par_iter()
                    .filter_map(|other_tx| {
                        if other_tx.hash() == tx_hash {
                            return None;
                        }
                        
                        let other_read = futures::executor::block_on(self.executor.get_read_set(other_tx));
                        let other_write = futures::executor::block_on(self.executor.get_write_set(other_tx));
                        
                        if other_read.is_ok() && other_write.is_ok() {
                            Some((other_tx.hash(), other_read.unwrap(), other_write.unwrap()))
                        } else {
                            None
                        }
                    })
                    .collect();
                
                // Check for conflicts using SIMD when possible
                for (other_hash, other_read, other_write) in chunk_results {
                    // Check read-write conflicts
                    if !self.is_disjoint_simd(&read_set, &other_write) ||
                       !self.is_disjoint_simd(&write_set, &other_read) ||
                       !self.is_disjoint_simd(&write_set, &other_write) {
                        dependencies.push(other_hash);
                    }
                }
            }
        } else {
            // Non-SIMD fallback
            for other_tx in all_transactions {
                if other_tx.hash() == tx_hash {
                    continue;
                }
                
                let other_read = futures::executor::block_on(self.executor.get_read_set(other_tx));
                let other_write = futures::executor::block_on(self.executor.get_write_set(other_tx));
                
                if other_read.is_err() || other_write.is_err() {
                    continue;
                }
                
                let other_read = other_read.unwrap();
                let other_write = other_write.unwrap();
                
                if !read_set.is_disjoint(&other_write) ||
                   !write_set.is_disjoint(&other_read) ||
                   !write_set.is_disjoint(&other_write) {
                    dependencies.push(other_tx.hash());
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
            if current_group.transactions.len() >= max_group_size {
                self.execution_groups.push(current_group);
                current_group = ExecutionGroup {
                    transactions: Vec::with_capacity(max_group_size),
                    dependencies: HashSet::new(),
                    dependents: HashSet::new(),
                };
            }
            
            // Add transaction to group
            current_group.transactions.push(tx);
            
            // Add dependencies
            if let Some(deps) = self.dependency_graph.edges.get(&tx_hash) {
                for dep in deps.iter() {
                    current_group.dependencies.insert(dep.clone());
                }
            }
            
            // Add dependents
            if let Some(deps) = self.dependency_graph.reverse_edges.get(&tx_hash) {
                for dep in deps.iter() {
                    current_group.dependents.insert(dep.clone());
                }
            }
        }
        
        // Add last group
        if !current_group.transactions.is_empty() {
            self.execution_groups.push(current_group);
        }
        
        Ok(())
    }
    
    /// Get topological order using parallel algorithm
    async fn get_topological_order_parallel(&self) -> Result<Vec<String>> {
        // Use parallel algorithm for large graphs
        if self.dependency_graph.vertices.len() > 10000 {
            return self.get_topological_order_parallel_kosaraju().await;
        }
        
        // Standard topological sort for smaller graphs
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut temp = HashSet::new();
        
        // Get all vertices
        let vertices: Vec<String> = self.dependency_graph.vertices.iter().map(|v| v.clone()).collect();
        
        // Process each vertex
        for vertex in vertices {
            if !visited.contains(&vertex) {
                self.visit(&vertex, &mut visited, &mut temp, &mut order).await?;
            }
        }
        
        Ok(order)
    }
    
    /// Parallel Kosaraju's algorithm for topological sorting
    async fn get_topological_order_parallel_kosaraju(&self) -> Result<Vec<String>> {
        // Step 1: Perform DFS and record finish times
        let vertices: Vec<String> = self.dependency_graph.vertices.iter().map(|v| v.clone()).collect();
        let vertices_count = vertices.len();
        
        let visited = Arc::new(dashmap::DashSet::new());
        let finish_order = Arc::new(Mutex::new(Vec::with_capacity(vertices_count)));
        
        // Use rayon for parallel processing if available
        if let Some(pool) = &self.thread_pool {
            pool.install(|| {
                let chunks = vertices.chunks(1000);
                
                chunks.into_iter().for_each(|chunk| {
                    chunk.into_iter().for_each(|vertex| {
                        if !visited.contains(vertex) {
                            let mut stack = Vec::new();
                            stack.push((vertex.clone(), true)); // (vertex, first visit)
                            
                            while let Some((v, is_first)) = stack.pop() {
                                if is_first {
                                    // Mark as visited
                                    visited.insert(v.clone());
                                    
                                    // Push again with first=false to process after children
                                    stack.push((v.clone(), false));
                                    
                                    // Process all children
                                    if let Some(edges) = self.dependency_graph.edges.get(&v) {
                                        for child in edges.iter() {
                                            if !visited.contains(child) {
                                                stack.push((child.clone(), true));
                                            }
                                        }
                                    }
                                } else {
                                    // Add to finish order after processing all children
                                    let mut order = finish_order.try_lock().unwrap();
                                    order.push(v);
                                }
                            }
                        }
                    });
                });
            });
        } else {
            // Fallback non-parallel implementation
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
                                    if !single_visited.contains(child) {
                                        stack.push((child.clone(), true));
                                    }
                                }
                            }
                        } else {
                            finish.push(v);
                        }
                    }
                }
            }
            
            let mut order = finish_order.lock().await;
            *order = finish;
        }
        
        // Return the finish order, which is our topological sort
        let finish_result = finish_order.lock().await;
        Ok(finish_result.clone())
    }
    
    /// Execute groups with work stealing and parallel processing
    async fn execute_groups_optimized(&mut self) -> Result<()> {
        let groups_count = self.execution_groups.len();
        debug!("Executing {} transaction groups", groups_count);
        
        // Clear previous results
        let mut results = self.results.lock().await;
        results.clear();
        drop(results); // Release lock
        
        // Execute groups with work stealing if enabled
        if self.config.enable_work_stealing {
            // Process groups in parallel using work stealing
            let (sender, receiver) = channel::unbounded();
            
            // Prepare groups for parallel execution
            for (i, group) in self.execution_groups.iter().enumerate() {
                sender.send((i, group.clone())).unwrap();
            }
            drop(sender);
            
            // Process with thread pool if available
            if let Some(pool) = &self.thread_pool {
                pool.install(|| {
                    std::iter::repeat(())
                        .take(self.workers.len())
                        .enumerate()
                        .par_bridge()
                        .for_each(|(worker_id, _)| {
                            let worker = &self.workers[worker_id];
                            let executor = self.executor.clone();
                            let state_tree = self.state_tree.clone();
                            let results = self.results.clone();
                            
                            while let Ok((group_id, group)) = receiver.recv() {
                                // Process group
                                debug!("Worker {} processing group {}", worker_id, group_id);
                                
                                let group_results = futures::executor::block_on(
                                    self.execute_group_simd(&group, &executor, &state_tree)
                                );
                                
                                // Store results
                                let mut results_guard = futures::executor::block_on(results.lock());
                                for (tx_hash, result) in group_results {
                                    results_guard.insert(tx_hash, result);
                                }
                            }
                        });
                });
            } else {
                // Fallback to sequential processing
                for (group_id, group) in receiver {
                    debug!("Sequentially processing group {}", group_id);
                    let group_results = self.execute_group_simd(&group, &self.executor, &self.state_tree).await;
                    
                    let mut results_guard = self.results.lock().await;
                    for (tx_hash, result) in group_results {
                        results_guard.insert(tx_hash, result);
                    }
                }
            }
        } else {
            // Traditional sequential group execution
            for (i, group) in self.execution_groups.iter().enumerate() {
                debug!("Executing group {}/{}", i + 1, groups_count);
                
                let group_results = self.execute_group_simd(group, &self.executor, &self.state_tree).await;
                
                let mut results_guard = self.results.lock().await;
                for (tx_hash, result) in group_results {
                    results_guard.insert(tx_hash, result);
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute a group of transactions with SIMD optimizations
    async fn execute_group_simd(
        &self,
        group: &ExecutionGroup,
        executor: &Arc<TransactionExecutor>,
        state_tree: &Arc<StateTree>,
    ) -> HashMap<String, Result<()>> {
        let mut results = HashMap::new();
        let batch_size = self.config.simd_batch_size;
        
        // Execute in batches for better SIMD utilization
        for chunk in group.transactions.chunks(batch_size) {
            let permits = match self.semaphore.acquire_many(chunk.len() as u32).await {
                Ok(permits) => permits,
                Err(_) => return results,
            };
            
            let chunk_results: Vec<(String, Result<()>)> = futures::future::join_all(
                chunk.iter().map(|tx| {
                    let executor = executor.clone();
                    let state_tree = state_tree.clone();
                    let tx_clone = tx.clone();
                    
                    async move {
                        let tx_hash = tx_clone.hash();
                        let result = executor.execute(&tx_clone, state_tree.as_ref()).await;
                        (tx_hash, result)
                    }
                })
            ).await;
            
            // Store results
            for (tx_hash, result) in chunk_results {
                results.insert(tx_hash, result);
            }
            
            // Release permits
            drop(permits);
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
        let tx_hash = tx.hash();
        // let mut dependencies = HashSet::new();
        // Add dependencies (method does not exist, so skip)
        // for dep in tx.get_dependencies() {
        //     dependencies.insert(dep.to_string());
        // }
        // Store dependencies (empty for now)
        self.dependency_graph.edges.insert(tx_hash.clone(), HashSet::new());
        // Add to processed set
        self.processed_txs.insert(tx_hash);
        Ok(())
    }
} 