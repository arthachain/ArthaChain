use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use anyhow::{Result, anyhow, Error};
use log::debug;
use serde::{Serialize, Deserialize};
use tokio::sync::{Mutex, Semaphore, RwLock};
use crate::ledger::transaction::Transaction;
use crate::ledger::state::{State, StateTree};
use crate::execution::executor::TransactionExecutor;

/// Transaction dependency graph
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Nodes (transactions)
    nodes: HashMap<String, Transaction>,
    /// Edges (dependencies)
    edges: HashMap<String, HashSet<String>>,
    /// Reverse edges (dependents)
    reverse_edges: HashMap<String, HashSet<String>>,
    pub vertices: HashSet<String>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            vertices: HashSet::new(),
        }
    }
}

/// Transaction execution group
#[derive(Debug, Clone)]
pub struct ExecutionGroup {
    /// Transactions in the group
    transactions: Vec<Transaction>,
    /// Group dependencies
    dependencies: HashSet<String>,
    /// Group dependents
    dependents: HashSet<String>,
}

/// Parallel execution configuration
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

/// Parallel execution manager
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
}

impl ParallelExecutionManager {
    /// Create a new parallel execution manager
    pub fn new(
        config: ParallelConfig,
        state_tree: Arc<StateTree>,
        executor: Arc<TransactionExecutor>,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_parallel));
        let config_clone = config.clone();
        Self {
            config: config_clone,
            dependency_graph: DependencyGraph {
                nodes: HashMap::new(),
                edges: HashMap::new(),
                reverse_edges: HashMap::new(),
                vertices: HashSet::new(),
            },
            execution_groups: Vec::new(),
            state_tree,
            executor,
            semaphore,
            results: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Process transactions in parallel
    pub async fn process_transactions(
        &mut self,
        transactions: Vec<Transaction>,
    ) -> Result<HashMap<String, Result<()>>> {
        debug!("Processing {} transactions in parallel", transactions.len());
        
        // Build dependency graph
        self.build_dependency_graph(&transactions).await?;
        
        // Create execution groups
        self.create_execution_groups().await?;
        
        // Execute groups in parallel
        self.execute_groups().await?;
        
        // Instead of cloning the results HashMap, collect into a new map
        let results_guard = self.results.lock().await;
        let results: HashMap<String, Result<()>> = results_guard.iter().map(|(k, v)| (k.clone(), v.as_ref().map(|_| ()).map_err(|e| anyhow!(e.to_string())))).collect();
        Ok(results)
    }

    /// Build dependency graph
    async fn build_dependency_graph(&mut self, transactions: &[Transaction]) -> Result<()> {
        // Clear existing graph
        self.dependency_graph.nodes.clear();
        self.dependency_graph.edges.clear();
        self.dependency_graph.reverse_edges.clear();
        
        // Add nodes
        for tx in transactions {
            self.dependency_graph.nodes.insert(tx.hash(), tx.clone());
        }
        
        // Add edges
        for tx in transactions {
            let tx_hash = tx.hash().clone();
            let mut dependencies = HashSet::new();
            
            // Check for read/write conflicts
            for other_tx in transactions {
                if other_tx.hash() == tx_hash {
                    continue;
                }
                
                if self.has_conflict(tx, &other_tx).await? {
                    dependencies.insert(other_tx.hash());
                }
            }
            
            // Add edges
            self.dependency_graph.edges.insert(tx_hash.clone(), dependencies.clone());
            
            // Add reverse edges
            for dep in &dependencies {
                self.dependency_graph.reverse_edges
                    .entry(dep.clone())
                    .or_insert_with(HashSet::new)
                    .insert(tx_hash.clone());
            }
        }
        
        Ok(())
    }

    /// Check for transaction conflicts
    async fn has_conflict(&self, tx1: &Transaction, tx2: &Transaction) -> Result<bool> {
        // Get read/write sets
        let read_set1 = self.executor.get_read_set(tx1).await?;
        let write_set1 = self.executor.get_write_set(tx1).await?;
        let read_set2 = self.executor.get_read_set(tx2).await?;
        let write_set2 = self.executor.get_write_set(tx2).await?;
        
        // Check for conflicts
        let has_conflict = !read_set1.is_disjoint(&write_set2) ||
            !write_set1.is_disjoint(&read_set2) ||
            !write_set1.is_disjoint(&write_set2);
        
        Ok(has_conflict)
    }

    /// Create execution groups
    async fn create_execution_groups(&mut self) -> Result<()> {
        self.execution_groups.clear();
        
        // Get topological order
        let order = self.get_topological_order().await?;
        
        // Create groups
        let mut current_group = ExecutionGroup {
            transactions: Vec::new(),
            dependencies: HashSet::new(),
            dependents: HashSet::new(),
        };
        
        for tx_hash in order {
            let tx = self.dependency_graph.nodes.get(&tx_hash).unwrap().clone();
            
            // Check if transaction can be added to current group
            if current_group.transactions.len() >= self.config.max_group_size {
                self.execution_groups.push(current_group);
                current_group = ExecutionGroup {
                    transactions: Vec::new(),
                    dependencies: HashSet::new(),
                    dependents: HashSet::new(),
                };
            }
            
            // Add transaction to group
            current_group.transactions.push(tx);
            
            // Add dependencies
            if let Some(deps) = self.dependency_graph.edges.get(&tx_hash) {
                for dep in deps {
                    current_group.dependencies.insert(dep.clone());
                }
            }
            
            // Add dependents
            if let Some(deps) = self.dependency_graph.reverse_edges.get(&tx_hash) {
                for dep in deps {
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

    /// Get topological order of transactions
    async fn get_topological_order(&self) -> Result<Vec<String>> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut temp = HashSet::new();
        
        for tx_hash in self.dependency_graph.nodes.keys() {
            if !visited.contains(tx_hash) {
                self.visit(tx_hash, &mut visited, &mut temp, &mut order).await?;
            }
        }
        
        order.reverse();
        Ok(order)
    }

    /// Visit node for topological sort
    async fn visit(
        &self,
        tx_hash: &String,
        visited: &mut HashSet<String>,
        temp: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) -> Result<()> {
        if temp.contains(tx_hash) {
            return Err(anyhow!("Cycle detected in transaction dependencies"));
        }
        
        if visited.contains(tx_hash) {
            return Ok(());
        }
        
        temp.insert(tx_hash.clone());
        
        if let Some(deps) = self.dependency_graph.edges.get(tx_hash) {
            for dep in deps {
                Box::pin(self.visit(dep, visited, temp, order)).await?;
            }
        }
        
        temp.remove(tx_hash);
        visited.insert(tx_hash.clone());
        order.push(tx_hash.clone());
        
        Ok(())
    }

    /// Execute groups in parallel
    async fn execute_groups(&mut self) -> Result<()> {
        let mut handles = Vec::new();
        
        for group in &self.execution_groups {
            let semaphore = self.semaphore.clone();
            let executor = self.executor.clone();
            let state_tree = self.state_tree.clone();
            let results = self.results.clone();
            let transactions = group.transactions.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                
                for tx in transactions {
                    let result = executor.execute_transaction(&tx, &state_tree).await;
                    results.lock().await.insert(tx.hash(), result);
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all groups to complete
        for handle in handles {
            handle.await?;
        }
        
        Ok(())
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

pub struct ParallelProcessor {
    _config: Arc<ParallelConfig>,
    _state: Arc<RwLock<State>>,
    _results: Arc<Mutex<HashMap<String, Result<(), Error>>>>,
    dependency_graph: DependencyGraph,
    _semaphore: Arc<Semaphore>,
    processed_txs: HashSet<String>,
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
            processed_txs: HashSet::new(),
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