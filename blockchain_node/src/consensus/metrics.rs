//! Consensus metrics for performance monitoring and optimization

use crate::types::{Address, ContractId, GasLimit, GasPrice, Hash};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Consensus performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusMetrics {
    /// Total transactions processed
    pub transactions_processed: u64,
    /// Average consensus time
    pub avg_consensus_time: Duration,
    /// Current TPS
    pub tps: f64,
    /// Validator participation rate
    pub participation_rate: f64,
    /// Block finalization time
    pub finalization_time: Duration,
}

impl Default for ConsensusMetrics {
    fn default() -> Self {
        Self {
            transactions_processed: 0,
            avg_consensus_time: Duration::from_millis(100),
            tps: 0.0,
            participation_rate: 100.0,
            finalization_time: Duration::from_secs(3),
        }
    }
}

/// Metrics collector for consensus operations
pub struct MetricsCollector {
    metrics: ConsensusMetrics,
    start_time: Instant,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: ConsensusMetrics::default(),
            start_time: Instant::now(),
        }
    }

    pub fn get_metrics(&self) -> &ConsensusMetrics {
        &self.metrics
    }

    pub fn record_transaction(&mut self) {
        self.metrics.transactions_processed += 1;
        let elapsed = self.start_time.elapsed();
        if elapsed.as_secs() > 0 {
            self.metrics.tps = self.metrics.transactions_processed as f64 / elapsed.as_secs_f64();
        }
    }

    pub fn update_consensus_time(&mut self, time: Duration) {
        self.metrics.avg_consensus_time = time;
    }

    pub fn update_participation_rate(&mut self, rate: f64) {
        self.metrics.participation_rate = rate;
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for smart contract execution
#[derive(Debug, Serialize, Deserialize)]
pub struct ContractMetrics {
    pub contracts_deployed: Mutex<u64>,
    pub contracts_executed: Mutex<u64>,
    pub total_gas_used: Mutex<u64>,
    pub avg_execution_time: Duration,
    pub successful_calls: Mutex<u64>,
    pub failed_calls: Mutex<u64>,
}

impl ContractMetrics {
    pub fn record_proposal_created(&self) {
        if let Ok(mut v) = self.contracts_executed.lock() {
            *v += 1;
        }
    }

    pub fn record_vote_cast(&self) {
        if let Ok(mut v) = self.successful_calls.lock() {
            *v += 1;
        }
    }

    pub fn record_proposal_executed(&self) {
        if let Ok(mut v) = self.contracts_executed.lock() {
            *v += 1;
        }
    }
}

impl Default for ContractMetrics {
    fn default() -> Self {
        Self {
            contracts_deployed: Mutex::new(0),
            contracts_executed: Mutex::new(0),
            total_gas_used: Mutex::new(0),
            avg_execution_time: Duration::from_millis(10),
            successful_calls: Mutex::new(0),
            failed_calls: Mutex::new(0),
        }
    }
}

/// Security-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub security_violations: u64,
    pub access_denied_count: u64,
    pub suspicious_transactions: u64,
    pub security_checks_passed: u64,
    pub security_checks_failed: u64,
}

impl Default for SecurityMetrics {
    fn default() -> Self {
        Self {
            security_violations: 0,
            access_denied_count: 0,
            suspicious_transactions: 0,
            security_checks_passed: 0,
            security_checks_failed: 0,
        }
    }
}

impl SecurityMetrics {
    pub fn record_validated_call(&self, _caller: Address, _target: Address) {}
    pub fn record_contract_registered(&self, _contract_id: ContractId) {}
    pub fn record_role_granted(&self, _account: Address, _role: String) {}
}

/// Gas-related metrics for optimization tracking
#[derive(Debug, Default)]
pub struct GasMetrics {
    /// Gas savings per contract
    contract_savings: Mutex<HashMap<Address, u64>>,
    /// Batch processing stats
    batches_processed: Mutex<u64>,
    /// Gas price updates
    price_updates: Mutex<Vec<(u64, u64)>>, // (block_number, price_value)
    /// Gas usage by contract and function
    gas_usage: Mutex<HashMap<(Address, Hash), u64>>,
}

impl GasMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_gas_savings(&self, contract: Address, savings: GasLimit) {
        if let Ok(mut savings_map) = self.contract_savings.lock() {
            *savings_map.entry(contract).or_insert(0) += savings.value();
        }
    }

    pub fn record_batch_processed(&self, batch_size: usize) {
        if let Ok(mut count) = self.batches_processed.lock() {
            *count += 1;
        }
    }

    pub fn record_gas_price_update(&self, block_number: u64, price: GasPrice) {
        if let Ok(mut updates) = self.price_updates.lock() {
            updates.push((block_number, price.value()));
            // Keep only recent updates
            if updates.len() > 1000 {
                updates.drain(0..100);
            }
        }
    }

    pub fn record_gas_usage(&self, contract: Address, function: Hash, gas_used: u64) {
        if let Ok(mut usage_map) = self.gas_usage.lock() {
            *usage_map.entry((contract, function)).or_insert(0) += gas_used;
        }
    }
}
