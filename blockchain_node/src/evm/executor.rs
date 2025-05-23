use crate::evm::runtime::EvmRuntime;
use crate::evm::types::{EvmConfig, EvmError, EvmExecutionResult, EvmTransaction};
use crate::storage::HybridStorage;
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// Executor for EVM transactions
pub struct EvmExecutor {
    /// EVM runtime instance
    runtime: Mutex<EvmRuntime>,
    /// Transaction queue
    transaction_queue: mpsc::Sender<EvmTransaction>,
    /// Configuration
    config: EvmConfig,
}

impl EvmExecutor {
    /// Create a new EVM executor
    pub fn new(storage: Arc<HybridStorage>, config: EvmConfig) -> Self {
        let (tx_sender, mut tx_receiver) = mpsc::channel(100);

        // Create the EVM runtime
        let runtime = EvmRuntime::new(storage.clone(), config.clone());

        // Create the executor
        let executor = Self {
            runtime: Mutex::new(runtime),
            transaction_queue: tx_sender,
            config,
        };

        // Spawn a task to process transactions
        let runtime_mutex = executor.runtime.clone();
        tokio::spawn(async move {
            while let Some(tx) = tx_receiver.recv().await {
                let mut runtime = runtime_mutex.lock().unwrap();

                // Set block context from current time
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                runtime.set_block_context(0, now); // Block number would come from the blockchain

                // Execute the transaction
                match runtime.execute(tx).await {
                    Ok(result) => {
                        info!(
                            "EVM transaction executed: success={}, gas_used={}",
                            result.success, result.gas_used
                        );

                        // Here is where we would integrate with the blockchain's consensus
                        // and transaction processing systems
                    }
                    Err(e) => {
                        error!("Failed to execute EVM transaction: {:?}", e);
                    }
                }

                // Clear caches to free memory
                runtime.clear_cache();
            }
        });

        executor
    }

    /// Submit a transaction for execution
    pub async fn submit_transaction(&self, tx: EvmTransaction) -> Result<(), anyhow::Error> {
        self.transaction_queue
            .send(tx)
            .await
            .map_err(|e| anyhow!("Failed to submit transaction: {}", e))
    }

    /// Execute a transaction immediately (synchronously)
    pub async fn execute_transaction_sync(
        &self,
        tx: EvmTransaction,
    ) -> Result<EvmExecutionResult, EvmError> {
        let mut runtime = self.runtime.lock().unwrap();

        // Set block context from current time
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        runtime.set_block_context(0, now); // Block number would come from the blockchain

        // Execute the transaction
        let result = runtime.execute(tx).await?;

        // Clear caches to free memory
        runtime.clear_cache();

        Ok(result)
    }

    /// Get a clone of the config
    pub fn get_config(&self) -> EvmConfig {
        self.config.clone()
    }

    /// Get a reference to the runtime
    pub fn get_runtime(&self) -> &Mutex<EvmRuntime> {
        &self.runtime
    }
}
