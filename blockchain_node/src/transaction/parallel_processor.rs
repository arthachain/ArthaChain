use crate::ledger::transaction::Transaction;
use crate::network::telemetry::NetworkMetrics;
use crate::types::{BlockHeight, TransactionHash};
use futures::stream::{self, StreamExt};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

const MAX_PARALLEL_TXS: usize = 1000;
const BATCH_SIZE: usize = 100;

pub struct ParallelProcessor {
    // Memory pool for pending transactions
    mempool: Arc<RwLock<MemPool>>,
    // Track transaction dependencies
    dependency_graph: Arc<RwLock<DependencyGraph>>,
    // Limit parallel execution
    semaphore: Arc<Semaphore>,
    // Network metrics
    metrics: Arc<RwLock<NetworkMetrics>>,
}

struct MemPool {
    // Transactions by priority
    by_priority: BTreeMap<u64, HashSet<TransactionHash>>,
    // Transactions by account
    by_account: HashMap<String, HashSet<TransactionHash>>,
    // Transaction details
    transactions: HashMap<TransactionHash, Transaction>,
    // Size tracking
    current_size: usize,
    max_size: usize,
}

struct DependencyGraph {
    // Track which transactions depend on each other
    edges: HashMap<TransactionHash, HashSet<TransactionHash>>,
    // Track readiness of transactions
    ready: HashSet<TransactionHash>,
}

impl ParallelProcessor {
    pub fn new(metrics: Arc<RwLock<NetworkMetrics>>) -> Self {
        Self {
            mempool: Arc::new(RwLock::new(MemPool::new())),
            dependency_graph: Arc::new(RwLock::new(DependencyGraph::new())),
            semaphore: Arc::new(Semaphore::new(MAX_PARALLEL_TXS)),
            metrics,
        }
    }

    pub async fn add_transaction(&self, tx: Transaction) -> anyhow::Result<()> {
        let tx_hash = tx.hash();

        // Update mempool
        let mut mempool = self.mempool.write().await;
        mempool.add_transaction(tx.clone())?;

        // Update dependency graph
        let mut graph = self.dependency_graph.write().await;
        graph.add_transaction(&tx).await?;

        // Record metrics
        if let Ok(mut metrics) = self.metrics.try_write() {
            metrics.record_mempool_add(tx_hash.clone());
        }
        Ok(())
    }

    pub async fn process_transactions(
        &self,
        block_height: BlockHeight,
    ) -> anyhow::Result<Vec<Transaction>> {
        let ready_txs = self.get_ready_transactions().await?;

        // Process transactions in parallel batches
        let batch_results = stream::iter(ready_txs.chunks(BATCH_SIZE))
            .map(|batch| self.process_batch(batch.to_vec(), block_height))
            .buffer_unordered(4) // Process up to 4 batches concurrently
            .collect::<Vec<_>>()
            .await;

        // Combine results and handle errors
        let mut processed = Vec::new();
        for result in batch_results {
            match result {
                Ok(batch) => processed.extend(batch),
                Err(e) => {
                    if let Ok(mut metrics) = self.metrics.try_write() {
                        metrics.record_batch_error(e.to_string());
                    }
                    eprintln!("Batch processing error: {}", e);
                    continue;
                }
            }
        }

        Ok(processed)
    }

    async fn process_batch(
        &self,
        batch: Vec<Transaction>,
        block_height: BlockHeight,
    ) -> anyhow::Result<Vec<Transaction>> {
        let batch_len = batch.len();
        let permits = self.semaphore.acquire_many(batch_len as u32).await?;

        let batch_results = stream::iter(batch)
            .map(|tx| self.process_single_transaction(tx, block_height))
            .buffer_unordered(batch_len) // Process all transactions in batch concurrently
            .collect::<Vec<_>>()
            .await;

        drop(permits); // Release semaphore permits

        // Filter successful transactions
        let processed: Vec<_> = batch_results.into_iter().filter_map(|r| r.ok()).collect();

        Ok(processed)
    }

    async fn process_single_transaction(
        &self,
        tx: Transaction,
        block_height: BlockHeight,
    ) -> anyhow::Result<Transaction> {
        let tx_hash = tx.hash();

        // Execute transaction
        let result = tx.clone().execute(block_height).await?;

        // Update mempool and dependency graph
        let mut mempool = self.mempool.write().await;
        let mut graph = self.dependency_graph.write().await;

        mempool.remove_transaction(&tx_hash)?;
        graph.remove_transaction(&tx_hash).await?;

        // Record metrics
        if let Ok(mut metrics) = self.metrics.try_write() {
            metrics.record_transaction_processed(tx_hash);
        }
        Ok(result)
    }

    async fn get_ready_transactions(&self) -> anyhow::Result<Vec<Transaction>> {
        let mempool = self.mempool.read().await;
        let graph = self.dependency_graph.read().await;

        let mut ready = Vec::new();
        for tx_hash in graph.ready.iter() {
            if let Some(tx) = mempool.transactions.get(tx_hash) {
                ready.push(tx.clone());
            }
        }

        Ok(ready)
    }
}

impl MemPool {
    fn new() -> Self {
        Self {
            by_priority: BTreeMap::new(),
            by_account: HashMap::new(),
            transactions: HashMap::new(),
            current_size: 0,
            max_size: 1024 * 1024 * 1024, // 1GB
        }
    }

    fn add_transaction(&mut self, tx: Transaction) -> anyhow::Result<()> {
        let tx_hash = tx.hash();
        let _priority = tx.priority();
        let account = tx.account().to_string();

        // Check size limits
        if self.current_size + tx.size() > self.max_size {
            self.evict_low_priority_transactions()?;
        }

        // Add to indices
        self.by_priority
            .entry(_priority)
            .or_default()
            .insert(tx_hash.clone());

        self.by_account
            .entry(account)
            .or_default()
            .insert(tx_hash.clone());

        let tx_size = tx.size();
        self.transactions.insert(tx_hash, tx);
        self.current_size += tx_size;

        Ok(())
    }

    fn remove_transaction(&mut self, tx_hash: &TransactionHash) -> anyhow::Result<()> {
        if let Some(tx) = self.transactions.remove(tx_hash) {
            let priority = tx.priority();
            let account = tx.account().to_string();

            if let Some(set) = self.by_priority.get_mut(&priority) {
                set.remove(tx_hash);
                if set.is_empty() {
                    self.by_priority.remove(&priority);
                }
            }

            if let Some(set) = self.by_account.get_mut(&account) {
                set.remove(tx_hash);
                if set.is_empty() {
                    self.by_account.remove(&account);
                }
            }

            self.current_size -= tx.size();
        }

        Ok(())
    }

    fn evict_low_priority_transactions(&mut self) -> anyhow::Result<()> {
        let mut space_freed = 0;
        let space_needed = self.current_size - self.max_size;

        // Remove lowest priority transactions first
        let priorities_to_remove: Vec<u64> = self.by_priority.keys().cloned().collect();

        for priority in priorities_to_remove {
            if let Some(hashes) = self.by_priority.get(&priority).cloned() {
                for tx_hash in hashes.iter() {
                    if let Some(tx) = self.transactions.get(tx_hash) {
                        space_freed += tx.size();
                        let tx_hash_clone = tx_hash.clone();
                        self.remove_transaction(&tx_hash_clone)?;

                        if space_freed >= space_needed {
                            return Ok(());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl DependencyGraph {
    fn new() -> Self {
        Self {
            edges: HashMap::new(),
            ready: HashSet::new(),
        }
    }

    async fn add_transaction(&mut self, tx: &Transaction) -> anyhow::Result<()> {
        let tx_hash = tx.hash();

        // Check dependencies
        let deps = tx.dependencies();
        if deps.is_empty() {
            self.ready.insert(tx_hash);
        } else {
            for dep in deps {
                self.edges.entry(dep).or_default().insert(tx_hash.clone());
            }
        }

        Ok(())
    }

    async fn remove_transaction(&mut self, tx_hash: &TransactionHash) -> anyhow::Result<()> {
        self.ready.remove(tx_hash);

        // Update dependencies
        if let Some(dependents) = self.edges.remove(tx_hash) {
            for dependent in dependents {
                // Check if dependent is now ready
                let mut is_ready = true;
                for (_, deps) in self.edges.iter() {
                    if deps.contains(&dependent) {
                        is_ready = false;
                        break;
                    }
                }

                if is_ready {
                    self.ready.insert(dependent);
                }
            }
        }

        Ok(())
    }
}
