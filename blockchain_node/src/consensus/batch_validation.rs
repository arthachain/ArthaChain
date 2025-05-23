use crate::consensus::validation::{ValidationEngine, ValidationRequest, ValidationResult};
use crate::ledger::transaction::Transaction;
use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, Semaphore};

/// Configuration for batch validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchValidationConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Number of concurrent batches
    pub concurrent_batches: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Maximum pending transactions
    pub max_pending_transactions: usize,
    /// Prioritize transactions by fee
    pub prioritize_by_fee: bool,
    /// Minimum transactions to trigger validation
    pub min_transactions_to_validate: usize,
    /// Maximum validation workers
    pub max_validation_workers: usize,
}

impl Default for BatchValidationConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 500,
            concurrent_batches: 4,
            batch_timeout_ms: 5000,
            max_pending_transactions: 10000,
            prioritize_by_fee: true,
            min_transactions_to_validate: 50,
            max_validation_workers: 8,
        }
    }
}

/// Status of a transaction batch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchStatus {
    /// Not yet validated
    Pending,
    /// Validation in progress
    InProgress,
    /// Successfully validated
    Valid,
    /// Validation failed
    Invalid,
    /// Validation timed out
    TimedOut,
}

/// Result of batch validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchValidationResult {
    /// Batch ID
    pub batch_id: u64,
    /// Status of the batch
    pub status: BatchStatus,
    /// Number of valid transactions
    pub valid_count: usize,
    /// Number of invalid transactions
    pub invalid_count: usize,
    /// Validation results by transaction hash
    pub results: HashMap<Vec<u8>, ValidationResult>,
    /// Time taken for validation in milliseconds
    pub duration_ms: u64,
    /// Error message if validation failed
    pub error: Option<String>,
}

/// Batch validator for transaction batches
pub struct BatchValidator {
    /// Configuration
    config: RwLock<BatchValidationConfig>,
    /// Validation engine
    validation_engine: Arc<ValidationEngine>,
    /// Pending transactions
    pending_transactions: RwLock<Vec<Transaction>>,
    /// Batch results
    batch_results: RwLock<HashMap<u64, BatchValidationResult>>,
    /// Next batch ID
    next_batch_id: RwLock<u64>,
    /// Running flag
    running: RwLock<bool>,
    /// Transaction receiver
    tx_receiver: Option<mpsc::Receiver<Transaction>>,
    /// Result sender
    result_sender: Option<mpsc::Sender<BatchValidationResult>>,
    /// Active batch count semaphore
    active_batches: Arc<Semaphore>,
}

impl BatchValidator {
    /// Create a new batch validator
    pub fn new(
        config: BatchValidationConfig,
        validation_engine: Arc<ValidationEngine>,
        tx_receiver: Option<mpsc::Receiver<Transaction>>,
        result_sender: Option<mpsc::Sender<BatchValidationResult>>,
    ) -> Self {
        Self {
            config: RwLock::new(config.clone()),
            validation_engine,
            pending_transactions: RwLock::new(Vec::new()),
            batch_results: RwLock::new(HashMap::new()),
            next_batch_id: RwLock::new(0),
            running: RwLock::new(false),
            tx_receiver,
            result_sender,
            active_batches: Arc::new(Semaphore::new(config.concurrent_batches)),
        }
    }

    /// Start the batch validator
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(anyhow!("Batch validator already running"));
        }

        *running = true;

        // Start the transaction processing loop if we have a receiver
        if let Some(receiver) = self.tx_receiver.take() {
            self.start_transaction_processing(receiver);
        }

        info!("Batch validator started");
        Ok(())
    }

    /// Stop the batch validator
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(anyhow!("Batch validator not running"));
        }

        *running = false;
        info!("Batch validator stopped");
        Ok(())
    }

    /// Start the transaction processing loop
    fn start_transaction_processing(&self, mut receiver: mpsc::Receiver<Transaction>) {
        let self_clone = Arc::new(self.clone());

        tokio::spawn(async move {
            while let Some(tx) = receiver.recv().await {
                let is_running = *self_clone.running.read().await;
                if !is_running {
                    break;
                }

                // Add transaction to pending queue
                self_clone.add_transaction(tx).await;

                // Check if we should process a batch
                let should_process = {
                    let config = self_clone.config.read().await;
                    let pending = self_clone.pending_transactions.read().await;
                    pending.len() >= config.min_transactions_to_validate
                };

                if should_process {
                    if let Err(e) = self_clone.process_batch().await {
                        warn!("Error processing transaction batch: {}", e);
                    }
                }
            }
        });

        // Start a timer to periodically process batches regardless of queue size
        let self_clone2 = self_clone.clone();
        tokio::spawn(async move {
            let mut interval = {
                let config = self_clone2.config.read().await;
                tokio::time::interval(std::time::Duration::from_millis(config.batch_timeout_ms))
            };

            loop {
                interval.tick().await;

                let is_running = *self_clone2.running.read().await;
                if !is_running {
                    break;
                }

                let has_pending = {
                    let pending = self_clone2.pending_transactions.read().await;
                    !pending.is_empty()
                };

                if has_pending {
                    if let Err(e) = self_clone2.process_batch().await {
                        warn!("Error processing timed batch: {}", e);
                    }
                }
            }
        });
    }

    /// Add a transaction to the pending queue
    pub async fn add_transaction(&self, tx: Transaction) -> Result<()> {
        let mut pending = self.pending_transactions.write().await;
        let config = self.config.read().await;

        // Check if we're at capacity
        if pending.len() >= config.max_pending_transactions {
            // If we prioritize by fee, maybe replace a lower fee transaction
            if config.prioritize_by_fee {
                // Find the transaction with the lowest fee
                if let Some(min_idx) = pending
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, t)| t.fee)
                    .map(|(i, _)| i)
                {
                    // Only replace if the new transaction has a higher fee
                    if tx.fee > pending[min_idx].fee {
                        pending[min_idx] = tx;
                        return Ok(());
                    }
                }
            }

            return Err(anyhow!("Pending transaction queue is full"));
        }

        // Add to pending queue
        pending.push(tx);
        Ok(())
    }

    /// Process a batch of transactions
    pub async fn process_batch(&self) -> Result<BatchValidationResult> {
        // Acquire semaphore permit
        let permit = self.active_batches.clone().acquire_owned().await?;

        // Create a batch of transactions
        let batch = {
            let config = self.config.read().await;
            let mut pending = self.pending_transactions.write().await;

            // Sort by fee if prioritized
            if config.prioritize_by_fee {
                pending.sort_by(|a, b| b.fee.cmp(&a.fee));
            }

            // Take up to max_batch_size transactions
            let batch_size = pending.len().min(config.max_batch_size);
            pending.drain(0..batch_size).collect::<Vec<_>>()
        };

        if batch.is_empty() {
            drop(permit);
            return Err(anyhow!("No transactions to process"));
        }

        // Create batch ID
        let batch_id = {
            let mut next_id = self.next_batch_id.write().await;
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Start batch validation
        let start_time = std::time::Instant::now();

        let mut batch_result = BatchValidationResult {
            batch_id,
            status: BatchStatus::InProgress,
            valid_count: 0,
            invalid_count: 0,
            results: HashMap::new(),
            duration_ms: 0,
            error: None,
        };

        // Store initial result
        {
            let mut results = self.batch_results.write().await;
            results.insert(batch_id, batch_result.clone());
        }

        // Process the batch
        let transactions = batch.clone();
        let validation_engine = self.validation_engine.clone();
        let config = self.config.read().await.clone();

        // Create a validation request for the batch
        let batch_handle = tokio::spawn(async move {
            let mut results = HashMap::new();
            let mut valid_count = 0;
            let mut invalid_count = 0;

            // Set up a worker pool for validation
            let (tx, mut rx) = mpsc::channel(config.max_validation_workers);

            // Submit validation tasks
            for transaction in transactions {
                let tx_hash = transaction.hash.clone();
                let tx_clone = transaction.clone();
                let validation_engine_clone = validation_engine.clone();
                let tx_sender = tx.clone();

                tokio::spawn(async move {
                    let result = match validation_engine_clone
                        .validate(ValidationRequest::Transaction(tx_clone))
                        .await
                    {
                        Ok(result) => result,
                        Err(e) => {
                            // Create a failed validation result
                            let mut result = ValidationResult::default();
                            result.status = crate::consensus::validation::ValidationStatus::Invalid;
                            result.error = Some(e.to_string());
                            result
                        }
                    };

                    let _ = tx_sender.send((tx_hash, result)).await;
                });
            }

            // Drop the sender to close the channel when all tasks are submitted
            drop(tx);

            // Collect results
            while let Some((tx_hash, result)) = rx.recv().await {
                match result.status {
                    crate::consensus::validation::ValidationStatus::Valid => {
                        valid_count += 1;
                    }
                    _ => {
                        invalid_count += 1;
                    }
                }

                results.insert(tx_hash, result);
            }

            // Determine batch status
            let status = if valid_count == transactions.len() {
                BatchStatus::Valid
            } else if invalid_count == transactions.len() {
                BatchStatus::Invalid
            } else {
                // Mixed results, we'll consider the batch invalid
                BatchStatus::Invalid
            };

            (status, valid_count, invalid_count, results, None)
        });

        // Wait for the batch to complete (with timeout)
        let (status, valid_count, invalid_count, results, error) = match tokio::time::timeout(
            std::time::Duration::from_millis(config.batch_timeout_ms),
            batch_handle,
        )
        .await
        {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => (
                BatchStatus::Invalid,
                0,
                batch.len(),
                HashMap::new(),
                Some(format!("Batch processing error: {}", e)),
            ),
            Err(_) => (
                BatchStatus::TimedOut,
                0,
                batch.len(),
                HashMap::new(),
                Some("Batch processing timed out".to_string()),
            ),
        };

        // Update batch result
        batch_result.status = status;
        batch_result.valid_count = valid_count;
        batch_result.invalid_count = invalid_count;
        batch_result.results = results;
        batch_result.error = error;
        batch_result.duration_ms = start_time.elapsed().as_millis() as u64;

        // Store final result
        {
            let mut batch_results = self.batch_results.write().await;
            batch_results.insert(batch_id, batch_result.clone());
        }

        // Send result if we have a sender
        if let Some(sender) = &self.result_sender {
            let _ = sender.send(batch_result.clone()).await;
        }

        // Log result
        info!(
            "Processed batch {}: {} valid, {} invalid, {}ms",
            batch_id, valid_count, invalid_count, batch_result.duration_ms
        );

        // Drop the permit to allow other batches to proceed
        drop(permit);

        Ok(batch_result)
    }

    /// Get batch validation result
    pub async fn get_batch_result(&self, batch_id: u64) -> Option<BatchValidationResult> {
        let results = self.batch_results.read().await;
        results.get(&batch_id).cloned()
    }

    /// Get all batch results
    pub async fn get_all_batch_results(&self) -> Vec<BatchValidationResult> {
        let results = self.batch_results.read().await;
        results.values().cloned().collect()
    }

    /// Get validation result for a transaction
    pub async fn get_transaction_result(&self, tx_hash: &[u8]) -> Option<ValidationResult> {
        let results = self.batch_results.read().await;

        for batch in results.values() {
            if let Some(result) = batch.results.get(tx_hash) {
                return Some(result.clone());
            }
        }

        None
    }

    /// Get pending transaction count
    pub async fn get_pending_count(&self) -> usize {
        let pending = self.pending_transactions.read().await;
        pending.len()
    }

    /// Update configuration
    pub async fn update_config(&self, config: BatchValidationConfig) -> Result<()> {
        let mut cfg = self.config.write().await;

        // Update the active batches semaphore if needed
        if config.concurrent_batches != cfg.concurrent_batches {
            let current_permits = self.active_batches.available_permits();
            let new_permits = config.concurrent_batches.saturating_sub(current_permits);

            if new_permits > 0 {
                self.active_batches.add_permits(new_permits);
            }
        }

        // Update config
        *cfg = config;

        Ok(())
    }
}

impl Clone for BatchValidator {
    fn clone(&self) -> Self {
        // This is a partial clone for internal use
        Self {
            config: RwLock::new(self.config.try_read().unwrap_or_default().clone()),
            validation_engine: self.validation_engine.clone(),
            pending_transactions: RwLock::new(Vec::new()),
            batch_results: RwLock::new(HashMap::new()),
            next_batch_id: RwLock::new(0),
            running: RwLock::new(false),
            tx_receiver: None,
            result_sender: None,
            active_batches: self.active_batches.clone(),
        }
    }
}
