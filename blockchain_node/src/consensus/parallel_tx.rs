use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, HashSet};
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use chrono::Utc;
use rayon::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use futures;

/// Transaction dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxDependencyGraph {
    /// Transaction vertices
    pub vertices: HashMap<Vec<u8>, TxVertex>,
    /// Transaction edges
    pub edges: HashMap<Vec<u8>, HashSet<Vec<u8>>>,
}

/// Transaction vertex
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxVertex {
    /// Transaction hash
    pub tx_hash: Vec<u8>,
    /// Transaction data
    pub data: Vec<u8>,
    /// Read set
    pub read_set: HashSet<Vec<u8>>,
    /// Write set
    pub write_set: HashSet<Vec<u8>>,
    /// Status
    pub status: TxStatus,
    /// Dependencies
    pub dependencies: HashSet<Vec<u8>>,
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TxStatus {
    /// Pending
    Pending,
    /// Ready for execution
    Ready,
    /// Executing
    Executing,
    /// Completed
    Completed,
    /// Failed
    Failed,
}

/// Parallel transaction processor
pub struct ParallelTxProcessor {
    graph: Arc<RwLock<TxDependencyGraph>>,
    max_parallel_txs: usize,
    conflict_resolution: ConflictResolutionStrategy,
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
}

impl ParallelTxProcessor {
    /// Create a new parallel transaction processor
    pub fn new(max_parallel_txs: usize, conflict_resolution: ConflictResolutionStrategy) -> Self {
        Self {
            graph: Arc::new(RwLock::new(TxDependencyGraph {
                vertices: HashMap::new(),
                edges: HashMap::new(),
            })),
            max_parallel_txs,
            conflict_resolution,
        }
    }

    /// Add a transaction to the graph
    pub async fn add_transaction(
        &self,
        tx_hash: Vec<u8>,
        data: Vec<u8>,
        read_set: HashSet<Vec<u8>>,
        write_set: HashSet<Vec<u8>>,
    ) -> Result<()> {
        let mut graph = self.graph.write().await;
        
        // Create vertex
        let vertex = TxVertex {
            tx_hash: tx_hash.clone(),
            data,
            read_set: read_set.clone(),
            write_set: write_set.clone(),
            status: TxStatus::Pending,
            dependencies: HashSet::new(),
        };
        
        // Add vertex
        graph.vertices.insert(tx_hash.clone(), vertex);
        
        // Add edges
        graph.edges.insert(tx_hash.clone(), HashSet::new());
        
        // Update dependencies
        self.update_dependencies(&mut graph, &tx_hash).await?;
        
        Ok(())
    }

    /// Update transaction dependencies
    async fn update_dependencies(&self, graph: &mut TxDependencyGraph, tx_hash: &[u8]) -> Result<()> {
        let vertex = graph.vertices.get(tx_hash).ok_or_else(|| anyhow!("Transaction not found"))?;
        
        let read_set = vertex.read_set.clone();
        let write_set = vertex.write_set.clone();

        // Collect conflicts first to avoid borrowing issues
        let mut conflicts = Vec::new();
        for (other_hash, other_vertex) in &graph.vertices {
            if other_hash.as_slice() == tx_hash {
                continue;
            }

            let has_read_write_conflict = !read_set.is_disjoint(&other_vertex.write_set);
            let has_write_read_conflict = !write_set.is_disjoint(&other_vertex.read_set);
            let has_write_write_conflict = !write_set.is_disjoint(&other_vertex.write_set);

            if has_read_write_conflict || has_write_read_conflict || has_write_write_conflict {
                conflicts.push(other_hash.clone());
            }
        }

        // Update edges and dependencies
        if let Some(edges) = graph.edges.get_mut(tx_hash) {
            for conflict in &conflicts {
                edges.insert(conflict.clone());
            }
        }

        if let Some(vertex) = graph.vertices.get_mut(tx_hash) {
            for conflict in conflicts {
                vertex.dependencies.insert(conflict);
            }
        }
        
        Ok(())
    }

    /// Get ready transactions
    pub async fn get_ready_transactions(&self) -> Vec<Vec<u8>> {
        let graph = self.graph.read().await;
        
        // Find transactions with no dependencies
        let ready: Vec<_> = graph.vertices
            .iter()
            .filter(|(_, v)| v.status == TxStatus::Pending && v.dependencies.is_empty())
            .map(|(h, _)| h.clone())
            .collect();
        
        // Apply conflict resolution strategy
        match self.conflict_resolution {
            ConflictResolutionStrategy::FCFS => {
                ready
            }
            ConflictResolutionStrategy::Priority => {
                let mut sorted = ready;
                sorted.sort_by(|a, b| {
                    let a_deps = graph.edges.get(a).map(|deps| deps.len()).unwrap_or(0);
                    let b_deps = graph.edges.get(b).map(|deps| deps.len()).unwrap_or(0);
                    b_deps.cmp(&a_deps)
                });
                sorted
            }
            ConflictResolutionStrategy::Random => {
                let mut rng = thread_rng();
                let mut shuffled = ready;
                shuffled.shuffle(&mut rng);
                shuffled
            }
        }
    }

    /// Execute transactions in parallel
    pub async fn execute_transactions(&self) -> Result<()> {
        let ready_txs = self.get_ready_transactions().await;
        
        // Limit parallel execution
        let batch_size = ready_txs.len().min(self.max_parallel_txs);
        let batch: Vec<_> = ready_txs.into_iter().take(batch_size).collect();
        
        // Execute transactions in parallel
        let results: Vec<_> = batch.par_iter()
            .map(|tx_hash| {
                // Simulate transaction execution with hash and timestamp
                let _timestamp = Utc::now();
                self.execute_single_transaction(tx_hash)
            })
            .collect();

        // Wait for all results
        let completed_results = futures::future::join_all(results).await;
        
        // Update transaction statuses
        let mut graph = self.graph.write().await;
        for (tx_hash, result) in batch.iter().zip(completed_results) {
            if let Some(vertex) = graph.vertices.get_mut(tx_hash) {
                vertex.status = if result.is_ok() {
                    TxStatus::Completed
                } else {
                    TxStatus::Failed
                };
            }
        }

        Ok(())
    }

    /// Execute a single transaction
    async fn execute_single_transaction(&self, tx_hash: &[u8]) -> Result<()> {
        // Get transaction data
        let graph = self.graph.read().await;
        let _vertex = graph.vertices.get(tx_hash)
            .ok_or_else(|| anyhow!("Transaction not found"))?;

        // TODO: Implement actual transaction execution logic here
        // For now, just simulate success
        Ok(())
    }

    /// Clean up completed transactions
    #[allow(dead_code)]
    async fn cleanup_completed_transactions(&self, graph: &mut TxDependencyGraph) -> Result<()> {
        // Find completed transactions
        let completed: Vec<_> = graph.vertices
            .iter()
            .filter(|(_, v)| v.status == TxStatus::Completed)
            .map(|(h, _)| h.clone())
            .collect();
        
        // Remove completed transactions
        for tx_hash in completed {
            graph.vertices.remove(&tx_hash);
            graph.edges.remove(&tx_hash);
            
            // Remove from other transactions' dependencies
            for edges in graph.edges.values_mut() {
                edges.remove(&tx_hash);
            }
            for vertex in graph.vertices.values_mut() {
                vertex.dependencies.remove(&tx_hash);
            }
        }
        
        Ok(())
    }

    /// Get transaction status
    pub async fn get_transaction_status(&self, tx_hash: &[u8]) -> Option<TxStatus> {
        let graph = self.graph.read().await;
        graph.vertices.get(tx_hash).map(|v| v.status.clone())
    }

    /// Get transaction dependencies
    pub async fn get_transaction_dependencies(&self, tx_hash: &[u8]) -> HashSet<Vec<u8>> {
        let graph = self.graph.read().await;
        graph.vertices.get(tx_hash)
            .map(|v| v.dependencies.clone())
            .unwrap_or_default()
    }
}

#[derive(Debug)]
pub struct TransactionVertex {
    pub read_set: HashSet<Vec<u8>>,
    pub write_set: HashSet<Vec<u8>>,
    pub dependencies: HashSet<Vec<u8>>,
}

#[derive(Debug)]
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
            read_set,
            write_set,
            dependencies: HashSet::new(),
        };
        self.vertices.insert(tx_hash.clone(), vertex);
        self.edges.insert(tx_hash.clone(), HashSet::new());
        
        self.update_dependencies(&tx_hash).await
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

            let has_read_write_conflict = !read_set.is_disjoint(&other_vertex.write_set);
            let has_write_read_conflict = !write_set.is_disjoint(&other_vertex.read_set);
            let has_write_write_conflict = !write_set.is_disjoint(&other_vertex.write_set);

            if has_read_write_conflict || has_write_read_conflict || has_write_write_conflict {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestTransaction {
        read_set: HashSet<Vec<u8>>,
        write_set: HashSet<Vec<u8>>,
        _dependencies: Vec<u64>,
    }

    #[tokio::test]
    async fn test_transaction_graph() {
        let mut graph = TransactionGraph::new();
        
        let tx1 = TestTransaction {
            read_set: vec![vec![1]].into_iter().collect(),
            write_set: vec![vec![2]].into_iter().collect(),
            _dependencies: Vec::new(),
        };
        
        let tx2 = TestTransaction {
            read_set: vec![vec![2]].into_iter().collect(),
            write_set: vec![vec![3]].into_iter().collect(),
            _dependencies: Vec::new(),
        };
        
        assert!(graph.add_transaction(
            vec![1], 
            tx1.read_set.clone(), 
            tx1.write_set.clone()
        ).await.is_ok());
        
        assert!(graph.add_transaction(
            vec![2], 
            tx2.read_set.clone(), 
            tx2.write_set.clone()
        ).await.is_ok());
        
        let executable = graph.vertices.iter()
            .filter(|(_, v)| v.dependencies.is_empty())
            .map(|(h, _)| h.clone())
            .collect::<Vec<_>>();
            
        assert!(!executable.is_empty());
    }
} 