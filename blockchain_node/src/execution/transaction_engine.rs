use crate::execution::executor::{ContractExecutor, ExecutionResult, TransactionExecutor};
use crate::ledger::state::State;
use crate::ledger::transaction::Transaction;
// use crate::wasm::{ContractExecutor, WasmConfig};
use anyhow::Result;
use log::{debug, error, info};
use std::sync::Arc;

/// Placeholder for WasmConfig when wasm feature is disabled
#[derive(Debug, Clone)]
pub struct WasmConfig {
    // Placeholder fields
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {}
    }
}

/// Configuration for the transaction engine
pub struct TransactionEngineConfig {
    /// Maximum concurrent transactions to process
    pub max_concurrent_txs: usize,
    /// Gas price adjustment factor
    pub gas_price_adjustment: f64,
    /// Maximum gas limit allowed per transaction
    pub max_gas_limit: u64,
    /// Minimum gas price allowed
    pub min_gas_price: u64,
    /// WASM configuration
    pub wasm_config: WasmConfig,
    /// Enable WASM smart contracts
    pub enable_wasm: bool,
}

impl Default for TransactionEngineConfig {
    fn default() -> Self {
        Self {
            max_concurrent_txs: 100,
            gas_price_adjustment: 1.0,
            max_gas_limit: 10_000_000,
            min_gas_price: 1,
            wasm_config: WasmConfig::default(),
            enable_wasm: true,
        }
    }
}

/// Main transaction execution engine
pub struct TransactionEngine {
    /// Transaction executor
    executor: TransactionExecutor,
    /// WASM contract executor (optional)
    wasm_executor: Option<Arc<ContractExecutor>>,
    /// Configuration
    config: TransactionEngineConfig,
    /// State reference
    state: Arc<State>,
}

impl TransactionEngine {
    /// Create a new transaction engine
    pub fn new(state: Arc<State>, config: TransactionEngineConfig) -> Result<Self> {
        // Create WASM executor if enabled
        let wasm_executor = if config.enable_wasm {
            let executor = ContractExecutor::new();
            Some(Arc::new(executor))
        } else {
            None
        };

        // Create transaction executor
        let executor = TransactionExecutor::new(
            wasm_executor.clone(),
            config.gas_price_adjustment,
            config.max_gas_limit,
            config.min_gas_price,
        );

        Ok(Self {
            executor,
            wasm_executor,
            config,
            state,
        })
    }

    /// Process a single transaction
    pub async fn process_transaction(&self, tx: &mut Transaction) -> Result<ExecutionResult> {
        debug!(
            "Processing transaction: {}",
            hex::encode(tx.hash().as_ref())
        );
        self.executor.execute_transaction(tx, &self.state).await
    }

    /// Process multiple transactions in parallel
    pub async fn process_transactions(
        &self,
        txs: &mut [Transaction],
    ) -> Result<Vec<ExecutionResult>> {
        info!("Processing {} transactions", txs.len());

        use futures::stream::{self, StreamExt};

        // Create a stream of transactions
        let results = stream::iter(txs.iter_mut())
            .map(|tx| self.process_transaction(tx))
            .buffer_unordered(self.config.max_concurrent_txs)
            .collect::<Vec<_>>()
            .await;

        // Check results
        let mut final_results = Vec::with_capacity(results.len());
        for result in results {
            match result {
                Ok(execution_result) => final_results.push(execution_result),
                Err(e) => {
                    error!("Transaction processing error: {}", e);
                    final_results
                        .push(ExecutionResult::Failure(format!("Processing error: {}", e)));
                }
            }
        }

        info!("Completed processing {} transactions", txs.len());
        Ok(final_results)
    }

    /// Apply transactions to a block
    pub async fn apply_transactions_to_block(
        &self,
        txs: &mut [Transaction],
        block_height: u64,
    ) -> Result<()> {
        debug!(
            "Applying {} transactions to block at height {}",
            txs.len(),
            block_height
        );

        // Process transactions
        let results = self.process_transactions(txs).await?;

        // Verify all succeeded
        for (i, result) in results.iter().enumerate() {
            match result {
                ExecutionResult::Success => {
                    // Transaction succeeded, no action needed
                }
                _ => {
                    // In a real implementation, we'd handle failures differently
                    // For now, we'll just log them
                    error!("Transaction {} failed: {:?}", i, result);
                }
            }
        }

        // Update block height in state
        self.state.set_height(block_height)?;

        Ok(())
    }

    /// Get the transaction executor
    pub fn get_executor(&self) -> &TransactionExecutor {
        &self.executor
    }

    /// Get the WASM executor
    pub fn get_wasm_executor(&self) -> Option<&Arc<ContractExecutor>> {
        self.wasm_executor.as_ref()
    }

    /// Get the state
    pub fn get_state(&self) -> &Arc<State> {
        &self.state
    }

    /// Get the configuration
    pub fn get_config(&self) -> &TransactionEngineConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::ledger::transaction::TransactionType;

    #[tokio::test]
    async fn test_transaction_engine() {
        // Create state
        let config = Config::default();
        let state = Arc::new(State::new(&config).unwrap());

        // Initialize state - increase balance to cover transfer + gas
        state.set_balance("sender", 50000).unwrap(); // Enough for 1000 transfer + 21000 gas + buffer
        state.set_balance("recipient", 0).unwrap();

        // Create engine
        let engine_config = TransactionEngineConfig::default();
        let engine = TransactionEngine::new(state.clone(), engine_config).unwrap();

        // Create transaction
        let mut tx = Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient".to_string(),
            1000,
            0,
            1,
            21000,
            vec![],
        );
        // Set signature after creation
        tx.signature = vec![1, 2, 3, 4];

        // Process transaction
        let result = engine.process_transaction(&mut tx).await.unwrap();

        // Verify result
        match result {
            ExecutionResult::Success => {
                // Check state updates - sender should have original - amount - gas_fee
                let expected_sender_balance = 50000 - 1000 - 21000; // 28000
                assert_eq!(
                    state.get_balance("sender").unwrap(),
                    expected_sender_balance
                );
                assert_eq!(state.get_balance("recipient").unwrap(), 1000);
                assert_eq!(state.get_nonce("sender").unwrap(), 1);
            }
            _ => panic!("Transaction processing failed: {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_batch_processing() {
        // Create state
        let config = Config::default();
        let state = Arc::new(State::new(&config).unwrap());

        // Initialize state
        state.set_balance("sender", 100000).unwrap();
        state.set_balance("recipient1", 0).unwrap();
        state.set_balance("recipient2", 0).unwrap();
        state.set_balance("recipient3", 0).unwrap();

        // Create engine
        let engine_config = TransactionEngineConfig::default();
        let engine = TransactionEngine::new(state.clone(), engine_config).unwrap();

        // Create transactions
        let mut txs = vec![
            {
                let mut tx = Transaction::new(
                    TransactionType::Transfer,
                    "sender".to_string(),
                    "recipient1".to_string(),
                    1000,
                    0,
                    1,
                    21000,
                    vec![],
                );
                tx.signature = vec![1, 2, 3, 4];
                tx
            },
            {
                let mut tx = Transaction::new(
                    TransactionType::Transfer,
                    "sender".to_string(),
                    "recipient2".to_string(),
                    2000,
                    1,
                    1,
                    21000,
                    vec![],
                );
                tx.signature = vec![1, 2, 3, 4];
                tx
            },
            {
                let mut tx = Transaction::new(
                    TransactionType::Transfer,
                    "sender".to_string(),
                    "recipient3".to_string(),
                    3000,
                    2,
                    1,
                    21000,
                    vec![],
                );
                tx.signature = vec![1, 2, 3, 4];
                tx
            },
        ];

        // Process transactions
        let results = engine.process_transactions(&mut txs).await.unwrap();

        // Verify results
        for (i, result) in results.iter().enumerate() {
            match result {
                ExecutionResult::Success => {}
                _ => panic!("Transaction {} failed: {:?}", i, result),
            }
        }

        // Check state updates
        assert_eq!(
            state.get_balance("sender").unwrap(),
            100000 - 1000 - 2000 - 3000 - (21000 * 3)
        );
        assert_eq!(state.get_balance("recipient1").unwrap(), 1000);
        assert_eq!(state.get_balance("recipient2").unwrap(), 2000);
        assert_eq!(state.get_balance("recipient3").unwrap(), 3000);
        assert_eq!(state.get_nonce("sender").unwrap(), 3);
    }
}
