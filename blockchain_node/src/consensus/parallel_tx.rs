//! Parallel Transaction Processing Module
//!
//! This module implements advanced parallel transaction processing with dependency tracking,
//! conflict resolution, and optimized execution strategies for blockchain consensus.

use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use chrono::Utc;
use rayon::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use futures::future::join_all;
use tokio::time::{timeout, Duration, Instant};
use std::hash::{Hash, Hasher};

/// Transaction identifier
pub type TxId = Vec<u8>;

/// Transaction dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxDependencyGraph {
    /// Transaction vertices
    pub vertices: HashMap<TxId, TxVertex>,
    /// Transaction edges (adjacency list)
    pub edges: HashMap<TxId, HashSet<TxId>>,
    /// Reverse edges for efficient dependency tracking
    pub reverse_edges: HashMap<TxId, HashSet<TxId>>,
}

impl TxDependencyGraph {
    pub fn new() -> Self {
        Self {
            vertices: HashMap::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
        }
    }

    /// Add a transaction to the graph
    pub fn add_transaction(&mut self, tx: TxVertex) -> Result<()> {
        let tx_id = tx.tx_hash.clone();
        
        // Update dependencies based on read/write conflicts
        self.update_dependencies(&tx)?;
        
        // Insert vertex
        self.vertices.insert(tx_id.clone(), tx);
        
        // Initialize edge lists if not present
        self.edges.entry(tx_id.clone()).or_insert_with(HashSet::new);
        self.reverse_edges.entry(tx_id).or_insert_with(HashSet::new);
        
        Ok(())
    }

    /// Update dependencies for a new transaction
    fn update_dependencies(&mut self, new_tx: &TxVertex) -> Result<()> {
        let new_tx_id = &new_tx.tx_hash;
        
        for (existing_tx_id, existing_tx) in &self.vertices {
            // Check for conflicts
            let conflicts = self.detect_conflicts(new_tx, existing_tx);
            
            if !conflicts.is_empty() {
                // Add dependency edge based on conflict type and priority
                match self.resolve_conflict_priority(new_tx, existing_tx, &conflicts) {
                    ConflictResolution::NewTxWaits => {
                        // New transaction waits for existing one
                        self.add_edge(existing_tx_id.clone(), new_tx_id.clone());
                    }
                    ConflictResolution::ExistingTxWaits => {
                        // Existing transaction waits for new one (rare case)
                        self.add_edge(new_tx_id.clone(), existing_tx_id.clone());
                    }
                    ConflictResolution::Abort => {
                        return Err(anyhow!("Transaction conflict cannot be resolved"));
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Add an edge between transactions
    fn add_edge(&mut self, from: TxId, to: TxId) {
        self.edges.entry(from.clone()).or_insert_with(HashSet::new).insert(to.clone());
        self.reverse_edges.entry(to).or_insert_with(HashSet::new).insert(from);
    }

    /// Remove an edge between transactions
    fn remove_edge(&mut self, from: &TxId, to: &TxId) {
        if let Some(edges) = self.edges.get_mut(from) {
            edges.remove(to);
        }
        if let Some(reverse_edges) = self.reverse_edges.get_mut(to) {
            reverse_edges.remove(from);
        }
    }

    /// Detect conflicts between two transactions
    fn detect_conflicts(&self, tx1: &TxVertex, tx2: &TxVertex) -> Vec<ConflictType> {
        let mut conflicts = Vec::new();
        
        // Read-Write conflict
        if !tx1.read_set.is_disjoint(&tx2.write_set) {
            conflicts.push(ConflictType::ReadWrite);
        }
        
        // Write-Read conflict
        if !tx1.write_set.is_disjoint(&tx2.read_set) {
            conflicts.push(ConflictType::WriteRead);
        }
        
        // Write-Write conflict
        if !tx1.write_set.is_disjoint(&tx2.write_set) {
            conflicts.push(ConflictType::WriteWrite);
        }
        
        conflicts
    }

    /// Resolve conflict priority between transactions
    fn resolve_conflict_priority(&self, tx1: &TxVertex, tx2: &TxVertex, conflicts: &[ConflictType]) -> ConflictResolution {
        // Priority-based resolution
        if tx1.priority > tx2.priority {
            return ConflictResolution::ExistingTxWaits;
        }
        if tx2.priority > tx1.priority {
            return ConflictResolution::NewTxWaits;
        }
        
        // Timestamp-based resolution (older transactions have priority)
        if tx1.timestamp < tx2.timestamp {
            return ConflictResolution::ExistingTxWaits;
        }
        if tx2.timestamp < tx1.timestamp {
            return ConflictResolution::NewTxWaits;
        }
        
        // For write-write conflicts, use abort strategy
        if conflicts.contains(&ConflictType::WriteWrite) {
            return ConflictResolution::Abort;
        }
        
        // Default: new transaction waits
        ConflictResolution::NewTxWaits
    }

    /// Get transactions with no dependencies (ready to execute)
    pub fn get_ready_transactions(&self) -> Vec<TxId> {
        self.vertices
            .iter()
            .filter(|(tx_id, vertex)| {
                vertex.status == TxStatus::Pending && 
                self.reverse_edges.get(*tx_id).map_or(true, |deps| deps.is_empty())
            })
            .map(|(tx_id, _)| tx_id.clone())
            .collect()
    }

    /// Mark transaction as completed and update dependencies
    pub fn complete_transaction(&mut self, tx_id: &TxId) -> Result<()> {
        // Update transaction status
        if let Some(vertex) = self.vertices.get_mut(tx_id) {
            vertex.status = TxStatus::Completed;
        }
        
        // Remove outgoing edges (dependents are now free)
        if let Some(dependents) = self.edges.remove(tx_id) {
            for dependent in dependents {
                self.remove_edge(tx_id, &dependent);
            }
        }
        
        // Remove from reverse edges
        self.reverse_edges.remove(tx_id);
        
        Ok(())
    }

    /// Detect cycles in the dependency graph
    pub fn has_deadlock(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        
        for tx_id in self.vertices.keys() {
            if !visited.contains(tx_id) {
                if self.has_cycle_util(tx_id, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        
        false
    }

    /// Utility function for cycle detection using DFS
    fn has_cycle_util(&self, tx_id: &TxId, visited: &mut HashSet<TxId>, rec_stack: &mut HashSet<TxId>) -> bool {
        visited.insert(tx_id.clone());
        rec_stack.insert(tx_id.clone());
        
        if let Some(neighbors) = self.edges.get(tx_id) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if self.has_cycle_util(neighbor, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(neighbor) {
                    return true;
                }
            }
        }
        
        rec_stack.remove(tx_id);
        false
    }

    /// Get topological ordering of transactions
    pub fn topological_sort(&self) -> Result<Vec<TxId>> {
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        
        // Calculate in-degrees
        for tx_id in self.vertices.keys() {
            in_degree.insert(tx_id.clone(), 0);
        }
        
        for edges in self.edges.values() {
            for to in edges {
                *in_degree.get_mut(to).unwrap() += 1;
            }
        }
        
        // Find nodes with in-degree 0
        for (tx_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(tx_id.clone());
            }
        }
        
        // Process queue
        while let Some(tx_id) = queue.pop_front() {
            result.push(tx_id.clone());
            
            if let Some(neighbors) = self.edges.get(&tx_id) {
                for neighbor in neighbors {
                    let degree = in_degree.get_mut(neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        
        if result.len() != self.vertices.len() {
            return Err(anyhow!("Cycle detected in dependency graph"));
        }
        
        Ok(result)
    }
}

/// Transaction vertex with enhanced metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxVertex {
    /// Transaction hash
    pub tx_hash: TxId,
    /// Transaction data
    pub data: Vec<u8>,
    /// Read set (resources read by transaction)
    pub read_set: HashSet<Vec<u8>>,
    /// Write set (resources modified by transaction)
    pub write_set: HashSet<Vec<u8>>,
    /// Transaction status
    pub status: TxStatus,
    /// Transaction priority (higher values = higher priority)
    pub priority: u32,
    /// Transaction timestamp
    pub timestamp: u64,
    /// Estimated execution time in milliseconds
    pub estimated_exec_time: u64,
    /// Gas limit
    pub gas_limit: u64,
    /// Transaction type
    pub tx_type: TransactionType,
    /// Retry count
    pub retry_count: u32,
    /// Maximum retries allowed
    pub max_retries: u32,
}

impl TxVertex {
    pub fn new(tx_hash: TxId, data: Vec<u8>, read_set: HashSet<Vec<u8>>, write_set: HashSet<Vec<u8>>) -> Self {
        Self {
            tx_hash,
            data,
            read_set,
            write_set,
            status: TxStatus::Pending,
            priority: 0,
            timestamp: Utc::now().timestamp() as u64,
            estimated_exec_time: 1000, // Default 1 second
            gas_limit: 1_000_000,
            tx_type: TransactionType::Regular,
            retry_count: 0,
            max_retries: 3,
        }
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_gas_limit(mut self, gas_limit: u64) -> Self {
        self.gas_limit = gas_limit;
        self
    }

    pub fn with_type(mut self, tx_type: TransactionType) -> Self {
        self.tx_type = tx_type;
        self
    }

    pub fn with_exec_time(mut self, exec_time: u64) -> Self {
        self.estimated_exec_time = exec_time;
        self
    }
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TxStatus {
    /// Pending execution
    Pending,
    /// Ready for execution
    Ready,
    /// Currently executing
    Executing,
    /// Successfully completed
    Completed,
    /// Failed execution
    Failed,
    /// Aborted due to conflict
    Aborted,
    /// Waiting for dependencies
    Waiting,
}

/// Transaction type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionType {
    /// Regular transaction
    Regular,
    /// High priority system transaction
    System,
    /// Smart contract deployment
    ContractDeployment,
    /// Smart contract call
    ContractCall,
    /// Cross-shard transaction
    CrossShard,
}

/// Conflict types between transactions
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictType {
    /// Read-Write conflict
    ReadWrite,
    /// Write-Read conflict
    WriteRead,
    /// Write-Write conflict
    WriteWrite,
}

/// Conflict resolution strategies
#[derive(Debug, Clone)]
pub enum ConflictResolution {
    /// New transaction waits for existing one
    NewTxWaits,
    /// Existing transaction waits for new one
    ExistingTxWaits,
    /// Abort one of the transactions
    Abort,
}

/// Parallel transaction processor with advanced capabilities
pub struct ParallelTxProcessor {
    /// Dependency graph
    graph: Arc<RwLock<TxDependencyGraph>>,
    /// Maximum parallel transactions
    max_parallel_txs: usize,
    /// Conflict resolution strategy
    conflict_resolution: ConflictResolutionStrategy,
    /// Execution semaphore
    execution_semaphore: Arc<Semaphore>,
    /// Performance metrics
    metrics: Arc<RwLock<ExecutionMetrics>>,
    /// Thread pool size
    thread_pool_size: usize,
    /// Execution timeout
    execution_timeout: Duration,
    /// Retry configuration
    retry_config: RetryConfig,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// First come, first served
    FCFS,
    /// Priority based
    Priority,
    /// Random selection
    Random,
    /// Shortest job first
    ShortestJobFirst,
    /// Optimistic concurrency control
    OptimisticCC,
    /// Pessimistic concurrency control
    PessimisticCC,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retries per transaction
    pub max_retries: u32,
    /// Base delay between retries in milliseconds
    pub base_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Maximum delay between retries
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            backoff_multiplier: 2.0,
            max_delay_ms: 5000,
        }
    }
}

/// Execution metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionMetrics {
    /// Total transactions processed
    pub total_transactions: u64,
    /// Successfully completed transactions
    pub completed_transactions: u64,
    /// Failed transactions
    pub failed_transactions: u64,
    /// Aborted transactions
    pub aborted_transactions: u64,
    /// Average execution time
    pub avg_execution_time_ms: f64,
    /// Peak parallelism achieved
    pub peak_parallelism: usize,
    /// Total conflicts detected
    pub total_conflicts: u64,
    /// Deadlocks detected
    pub deadlocks_detected: u64,
    /// Total retries
    pub total_retries: u64,
    /// Throughput (transactions per second)
    pub throughput_tps: f64,
}

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Transaction ID
    pub tx_id: TxId,
    /// Execution status
    pub status: TxStatus,
    /// Result data
    pub result_data: Option<Vec<u8>>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Gas used
    pub gas_used: u64,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Retry count
    pub retry_count: u32,
}

impl ParallelTxProcessor {
    /// Create a new parallel transaction processor
    pub fn new(max_parallel_txs: usize, conflict_resolution: ConflictResolutionStrategy) -> Self {
        Self {
            graph: Arc::new(RwLock::new(TxDependencyGraph::new())),
            max_parallel_txs,
            conflict_resolution,
            execution_semaphore: Arc::new(Semaphore::new(max_parallel_txs)),
            metrics: Arc::new(RwLock::new(ExecutionMetrics::default())),
            thread_pool_size: num_cpus::get(),
            execution_timeout: Duration::from_secs(30),
            retry_config: RetryConfig::default(),
        }
    }

    /// Create processor with custom configuration
    pub fn with_config(
        max_parallel_txs: usize,
        conflict_resolution: ConflictResolutionStrategy,
        execution_timeout: Duration,
        retry_config: RetryConfig,
    ) -> Self {
        Self {
            graph: Arc::new(RwLock::new(TxDependencyGraph::new())),
            max_parallel_txs,
            conflict_resolution,
            execution_semaphore: Arc::new(Semaphore::new(max_parallel_txs)),
            metrics: Arc::new(RwLock::new(ExecutionMetrics::default())),
            thread_pool_size: num_cpus::get(),
            execution_timeout,
            retry_config,
        }
    }

    /// Add a transaction to the processing queue
    pub async fn add_transaction(&self, tx: TxVertex) -> Result<()> {
        let mut graph = self.graph.write().await;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_transactions += 1;
        }
        
        graph.add_transaction(tx)?;
        Ok(())
    }

    /// Add multiple transactions as a batch
    pub async fn add_transaction_batch(&self, transactions: Vec<TxVertex>) -> Result<()> {
        let mut graph = self.graph.write().await;
        
        for tx in transactions {
            graph.add_transaction(tx)?;
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_transactions += graph.vertices.len() as u64;
        }
        
        Ok(())
    }

    /// Get ready transactions based on conflict resolution strategy
    pub async fn get_ready_transactions(&self) -> Vec<TxId> {
        let graph = self.graph.read().await;
        let mut ready_txs = graph.get_ready_transactions();
        
        // Apply conflict resolution strategy
        match self.conflict_resolution {
            ConflictResolutionStrategy::FCFS => {
                // Sort by timestamp (oldest first)
                ready_txs.sort_by_key(|tx_id| {
                    graph.vertices.get(tx_id).map(|v| v.timestamp).unwrap_or(0)
                });
            }
            ConflictResolutionStrategy::Priority => {
                // Sort by priority (highest first)
                ready_txs.sort_by_key(|tx_id| {
                    std::cmp::Reverse(graph.vertices.get(tx_id).map(|v| v.priority).unwrap_or(0))
                });
            }
            ConflictResolutionStrategy::ShortestJobFirst => {
                // Sort by estimated execution time (shortest first)
                ready_txs.sort_by_key(|tx_id| {
                    graph.vertices.get(tx_id).map(|v| v.estimated_exec_time).unwrap_or(u64::MAX)
                });
            }
            ConflictResolutionStrategy::Random => {
                let mut rng = thread_rng();
                ready_txs.shuffle(&mut rng);
            }
            ConflictResolutionStrategy::OptimisticCC | ConflictResolutionStrategy::PessimisticCC => {
                // Use priority-based ordering for concurrency control
                ready_txs.sort_by_key(|tx_id| {
                    std::cmp::Reverse(graph.vertices.get(tx_id).map(|v| v.priority).unwrap_or(0))
                });
            }
        }
        
        ready_txs
    }

    /// Execute transactions in parallel
    pub async fn execute_transactions(&self) -> Result<Vec<ExecutionResult>> {
        let start_time = Instant::now();
        let ready_txs = self.get_ready_transactions().await;
        
        if ready_txs.is_empty() {
            return Ok(Vec::new());
        }
        
        // Check for deadlocks
        {
            let graph = self.graph.read().await;
            if graph.has_deadlock() {
                let mut metrics = self.metrics.write().await;
                metrics.deadlocks_detected += 1;
                return Err(anyhow!("Deadlock detected in transaction dependency graph"));
            }
        }
        
        // Limit parallel execution
        let batch_size = ready_txs.len().min(self.max_parallel_txs);
        let batch: Vec<_> = ready_txs.into_iter().take(batch_size).collect();
        
        // Update peak parallelism
        {
            let mut metrics = self.metrics.write().await;
            metrics.peak_parallelism = metrics.peak_parallelism.max(batch.len());
        }
        
        // Execute transactions in parallel
        let mut handles = Vec::new();
        
        for tx_id in batch {
            let graph = self.graph.clone();
            let semaphore = self.execution_semaphore.clone();
            let timeout = self.execution_timeout;
            let retry_config = self.retry_config.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                
                // Get transaction details
                let tx_vertex = {
                    let graph_read = graph.read().await;
                    graph_read.vertices.get(&tx_id).cloned()
                };
                
                if let Some(mut tx) = tx_vertex {
                    // Mark as executing
                    {
                        let mut graph_write = graph.write().await;
                        if let Some(vertex) = graph_write.vertices.get_mut(&tx_id) {
                            vertex.status = TxStatus::Executing;
                        }
                    }
                    
                    // Execute with retry logic
                    let result = Self::execute_with_retry(&tx_id, &mut tx, timeout, retry_config).await;
                    
                    // Update transaction status
                    {
                        let mut graph_write = graph.write().await;
                        if let Some(vertex) = graph_write.vertices.get_mut(&tx_id) {
                            vertex.status = result.status.clone();
                            vertex.retry_count = result.retry_count;
                        }
                        
                        // Mark as completed if successful
                        if result.status == TxStatus::Completed {
                            let _ = graph_write.complete_transaction(&tx_id);
                        }
                    }
                    
                    result
                } else {
                    ExecutionResult {
                        tx_id,
                        status: TxStatus::Failed,
                        result_data: None,
                        execution_time_ms: 0,
                        gas_used: 0,
                        error_message: Some("Transaction not found".to_string()),
                        retry_count: 0,
                    }
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all executions to complete
        let results = join_all(handles).await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            let execution_time = start_time.elapsed();
            
            for result in &results {
                match result.status {
                    TxStatus::Completed => metrics.completed_transactions += 1,
                    TxStatus::Failed => metrics.failed_transactions += 1,
                    TxStatus::Aborted => metrics.aborted_transactions += 1,
                    _ => {}
                }
                metrics.total_retries += result.retry_count as u64;
            }
            
            // Update average execution time
            let total_exec_time: u64 = results.iter().map(|r| r.execution_time_ms).sum();
            if !results.is_empty() {
                metrics.avg_execution_time_ms = total_exec_time as f64 / results.len() as f64;
            }
            
            // Calculate throughput
            if execution_time.as_millis() > 0 {
                metrics.throughput_tps = (results.len() as f64 * 1000.0) / execution_time.as_millis() as f64;
            }
        }
        
        Ok(results)
    }

    /// Execute a single transaction with retry logic
    async fn execute_with_retry(
        tx_id: &TxId,
        tx: &mut TxVertex,
        timeout: Duration,
        retry_config: RetryConfig,
    ) -> ExecutionResult {
        let mut retry_count = 0;
        let mut last_error = None;
        
        while retry_count <= retry_config.max_retries {
            let start_time = Instant::now();
            
            // Execute transaction with timeout
            let execution_result = timeout(timeout, Self::execute_single_transaction(tx_id, tx)).await;
            
            match execution_result {
                Ok(Ok(result_data)) => {
                    // Success
                    return ExecutionResult {
                        tx_id: tx_id.clone(),
                        status: TxStatus::Completed,
                        result_data: Some(result_data),
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                        gas_used: tx.gas_limit / 2, // Simulate gas usage
                        error_message: None,
                        retry_count,
                    };
                }
                Ok(Err(e)) => {
                    // Execution error
                    last_error = Some(e.to_string());
                    
                    // Check if retryable
                    if retry_count < retry_config.max_retries && Self::is_retryable_error(&e) {
                        retry_count += 1;
                        tx.retry_count = retry_count;
                        
                        // Calculate delay
                        let delay_ms = std::cmp::min(
                            retry_config.base_delay_ms * (retry_config.backoff_multiplier.powi(retry_count as i32) as u64),
                            retry_config.max_delay_ms,
                        );
                        
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                        continue;
                    }
                }
                Err(_) => {
                    // Timeout
                    last_error = Some("Execution timeout".to_string());
                    break;
                }
            }
        }
        
        // All retries exhausted or non-retryable error
        ExecutionResult {
            tx_id: tx_id.clone(),
            status: TxStatus::Failed,
            result_data: None,
            execution_time_ms: 0,
            gas_used: 0,
            error_message: last_error,
            retry_count,
        }
    }

    /// Execute a single transaction (mock implementation)
    async fn execute_single_transaction(tx_id: &TxId, tx: &TxVertex) -> Result<Vec<u8>> {
        // Simulate execution time based on transaction type and complexity
        let exec_time = match tx.tx_type {
            TransactionType::System => Duration::from_millis(50),
            TransactionType::Regular => Duration::from_millis(100),
            TransactionType::ContractCall => Duration::from_millis(200),
            TransactionType::ContractDeployment => Duration::from_millis(500),
            TransactionType::CrossShard => Duration::from_millis(1000),
        };
        
        tokio::time::sleep(exec_time).await;
        
        // Simulate occasional failures for testing retry logic
        if tx_id.len() % 17 == 0 {
            return Err(anyhow!("Simulated execution error"));
        }
        
        // Return mock result
        Ok(format!("Result for tx: {}", hex::encode(tx_id)).into_bytes())
    }

    /// Check if an error is retryable
    fn is_retryable_error(error: &anyhow::Error) -> bool {
        let error_str = error.to_string().to_lowercase();
        
        // Define retryable error patterns
        error_str.contains("timeout") ||
        error_str.contains("network") ||
        error_str.contains("temporary") ||
        error_str.contains("resource unavailable") ||
        error_str.contains("simulated execution error") // For testing
    }

    /// Get execution metrics
    pub async fn get_metrics(&self) -> ExecutionMetrics {
        self.metrics.read().await.clone()
    }

    /// Reset metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = ExecutionMetrics::default();
    }

    /// Get transaction status
    pub async fn get_transaction_status(&self, tx_id: &TxId) -> Option<TxStatus> {
        let graph = self.graph.read().await;
        graph.vertices.get(tx_id).map(|vertex| vertex.status.clone())
    }

    /// Get transaction dependencies
    pub async fn get_transaction_dependencies(&self, tx_id: &TxId) -> HashSet<TxId> {
        let graph = self.graph.read().await;
        graph.reverse_edges.get(tx_id).cloned().unwrap_or_default()
    }

    /// Check for deadlocks
    pub async fn has_deadlock(&self) -> bool {
        let graph = self.graph.read().await;
        graph.has_deadlock()
    }

    /// Get topological order of pending transactions
    pub async fn get_execution_order(&self) -> Result<Vec<TxId>> {
        let graph = self.graph.read().await;
        graph.topological_sort()
    }

    /// Clear completed transactions from the graph
    pub async fn cleanup_completed_transactions(&self) -> Result<u64> {
        let mut graph = self.graph.write().await;
        let mut removed_count = 0;
        
        let completed_txs: Vec<TxId> = graph.vertices
            .iter()
            .filter(|(_, vertex)| vertex.status == TxStatus::Completed)
            .map(|(tx_id, _)| tx_id.clone())
            .collect();
        
        for tx_id in completed_txs {
            graph.vertices.remove(&tx_id);
            graph.edges.remove(&tx_id);
            graph.reverse_edges.remove(&tx_id);
            removed_count += 1;
        }
        
        Ok(removed_count)
    }

    /// Abort a transaction
    pub async fn abort_transaction(&self, tx_id: &TxId) -> Result<()> {
        let mut graph = self.graph.write().await;
        
        if let Some(vertex) = graph.vertices.get_mut(tx_id) {
            vertex.status = TxStatus::Aborted;
            
            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.aborted_transactions += 1;
            }
            
            // Remove from graph
            graph.vertices.remove(tx_id);
            graph.edges.remove(tx_id);
            graph.reverse_edges.remove(tx_id);
            
            Ok(())
        } else {
            Err(anyhow!("Transaction not found"))
        }
    }

    /// Process transactions continuously
    pub async fn run_continuous_processing(&self) -> Result<()> {
        loop {
            let results = self.execute_transactions().await?;
            
            if results.is_empty() {
                // No ready transactions, wait before next check
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            
            // Cleanup completed transactions periodically
            if rand::random::<f64>() < 0.1 {
                let _ = self.cleanup_completed_transactions().await;
            }
        }
    }
}

// Legacy compatibility structures (maintained for backward compatibility)
pub struct TransactionVertex {
    pub read_set: HashSet<Vec<u8>>,
    pub write_set: HashSet<Vec<u8>>,
    pub dependencies: HashSet<Vec<u8>>,
}

pub struct TransactionGraph {
    pub vertices: HashMap<Vec<u8>, TransactionVertex>,
    pub edges: HashMap<Vec<u8>, HashSet<Vec<u8>>>,
}

impl TransactionGraph {
    pub fn new() -> Self {
        Self {
            vertices: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    pub async fn add_transaction(&mut self, tx_hash: Vec<u8>, read_set: HashSet<Vec<u8>>, write_set: HashSet<Vec<u8>>) -> Result<()> {
        let vertex = TransactionVertex {
            read_set: read_set.clone(),
            write_set: write_set.clone(),
            dependencies: HashSet::new(),
        };
        
        self.vertices.insert(tx_hash.clone(), vertex);
        self.edges.insert(tx_hash.clone(), HashSet::new());
        
        self.update_dependencies(&tx_hash).await?;
        Ok(())
    }

    async fn update_dependencies(&mut self, tx_hash: &[u8]) -> Result<()> {
        let vertex = self.vertices.get(tx_hash).ok_or_else(|| anyhow!("Transaction not found"))?;
        let read_set = vertex.read_set.clone();
        let write_set = vertex.write_set.clone();

        let mut conflicts = Vec::new();
        for (other_hash, other_vertex) in &self.vertices {
            if other_hash.as_slice() == tx_hash {
                continue;
            }

            let has_conflict = !read_set.is_disjoint(&other_vertex.write_set) ||
                              !write_set.is_disjoint(&other_vertex.read_set) ||
                              !write_set.is_disjoint(&other_vertex.write_set);

            if has_conflict {
                conflicts.push(other_hash.clone());
            }
        }

        if let Some(edges) = self.edges.get_mut(tx_hash) {
            for conflict in &conflicts {
                edges.insert(conflict.clone());
            }
        }

        if let Some(vertex) = self.vertices.get_mut(tx_hash) {
            for conflict in conflicts {
                vertex.dependencies.insert(conflict);
            }
        }

        Ok(())
    }
}

impl Default for TransactionGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dependency_graph() {
        let mut graph = TxDependencyGraph::new();
        
        // Create transactions with conflicting read/write sets
        let tx1 = TxVertex::new(
            vec![1, 2, 3],
            vec![1, 2, 3],
            HashSet::from([vec![1], vec![2]]),
            HashSet::from([vec![3]]),
        ).with_priority(1);
        
        let tx2 = TxVertex::new(
            vec![4, 5, 6],
            vec![4, 5, 6],
            HashSet::from([vec![3]]), // Conflicts with tx1's write set
            HashSet::from([vec![4]]),
        ).with_priority(2);
        
        graph.add_transaction(tx1).unwrap();
        graph.add_transaction(tx2).unwrap();
        
        // Check dependencies
        let ready_txs = graph.get_ready_transactions();
        assert!(!ready_txs.is_empty());
        
        // Complete first transaction
        let first_tx = ready_txs[0].clone();
        graph.complete_transaction(&first_tx).unwrap();
        
        // Check if second transaction is now ready
        let ready_txs_after = graph.get_ready_transactions();
        assert!(ready_txs_after.len() >= 1);
    }

    #[tokio::test]
    async fn test_parallel_processor() {
        let processor = ParallelTxProcessor::new(4, ConflictResolutionStrategy::Priority);
        
        // Add test transactions
        let tx1 = TxVertex::new(
            vec![1],
            vec![1],
            HashSet::new(),
            HashSet::from([vec![1]]),
        ).with_priority(10);
        
        let tx2 = TxVertex::new(
            vec![2],
            vec![2],
            HashSet::new(),
            HashSet::from([vec![2]]),
        ).with_priority(5);
        
        processor.add_transaction(tx1).await.unwrap();
        processor.add_transaction(tx2).await.unwrap();
        
        // Execute transactions
        let results = processor.execute_transactions().await.unwrap();
        assert!(!results.is_empty());
        
        // Check metrics
        let metrics = processor.get_metrics().await;
        assert!(metrics.total_transactions >= 2);
    }

    #[tokio::test]
    async fn test_deadlock_detection() {
        let mut graph = TxDependencyGraph::new();
        
        // Create circular dependency
        let tx1 = TxVertex::new(vec![1], vec![1], HashSet::from([vec![1]]), HashSet::from([vec![2]]));
        let tx2 = TxVertex::new(vec![2], vec![2], HashSet::from([vec![2]]), HashSet::from([vec![1]]));
        
        graph.add_transaction(tx1).unwrap();
        graph.add_transaction(tx2).unwrap();
        
        // Should detect deadlock
        assert!(graph.has_deadlock());
    }

    #[tokio::test]
    async fn test_conflict_resolution() {
        let processor = ParallelTxProcessor::new(2, ConflictResolutionStrategy::ShortestJobFirst);
        
        let fast_tx = TxVertex::new(
            vec![1],
            vec![1],
            HashSet::new(),
            HashSet::from([vec![1]]),
        ).with_exec_time(100);
        
        let slow_tx = TxVertex::new(
            vec![2],
            vec![2],
            HashSet::new(),
            HashSet::from([vec![2]]),
        ).with_exec_time(1000);
        
        processor.add_transaction(slow_tx).await.unwrap();
        processor.add_transaction(fast_tx).await.unwrap();
        
        let ready_txs = processor.get_ready_transactions().await;
        // Fast transaction should come first
        let graph = processor.graph.read().await;
        let first_tx = graph.vertices.get(&ready_txs[0]).unwrap();
        assert_eq!(first_tx.estimated_exec_time, 100);
    }

    #[test]
    fn test_retry_config() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay_ms, 100);
        assert_eq!(config.backoff_multiplier, 2.0);
        assert_eq!(config.max_delay_ms, 5000);
    }

    #[tokio::test]
    async fn test_transaction_graph_legacy() {
        let mut graph = TransactionGraph::new();
        
        let read_set1 = HashSet::from([vec![1, 2]]);
        let write_set1 = HashSet::from([vec![3, 4]]);
        
        let read_set2 = HashSet::from([vec![3]]); // Conflicts with write_set1
        let write_set2 = HashSet::from([vec![5]]);
        
        graph.add_transaction(vec![1, 2, 3], read_set1, write_set1).await.unwrap();
        graph.add_transaction(vec![4, 5, 6], read_set2, write_set2).await.unwrap();
        
        // Check that dependencies were established
        let tx1_vertex = graph.vertices.get(&vec![1, 2, 3]).unwrap();
        let tx2_vertex = graph.vertices.get(&vec![4, 5, 6]).unwrap();
        
        assert!(!tx1_vertex.dependencies.is_empty() || !tx2_vertex.dependencies.is_empty());
    }
} 