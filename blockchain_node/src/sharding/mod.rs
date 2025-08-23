use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

use crate::types::{Transaction, Address, Hash};

pub mod shard;

/// Status of cross shard operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossShardStatus {
    /// Transaction is pending
    Pending,
    /// Transaction is in progress
    InProgress,
    /// Transaction has been completed
    Completed,
    /// Transaction has failed
    Failed(String),
    /// Transaction has timed out
    TimedOut,
    /// Transaction has been confirmed
    Confirmed,
    /// Transaction has been rejected
    Rejected,
}

/// Cross-shard transaction reference for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossShardReference {
    pub tx_hash: String,
    pub involved_shards: Vec<u32>,
    pub status: CrossShardStatus,
    pub created_at_height: u64,
}

/// Shard ID type
pub type ShardId = u64;

/// Performance metrics for individual shards
#[derive(Debug)]
pub struct ShardPerformance {
    pub transactions_processed: AtomicU64,
    pub blocks_created: AtomicU64,
    pub average_processing_time: AtomicU64, // in nanoseconds
    pub current_load: AtomicU64,
    pub last_activity: AtomicU64, // timestamp
}

/// Types of shards with different specializations
#[derive(Debug, Clone, PartialEq)]
pub enum ShardType {
    HighPerformance,    // Optimized for speed
    StorageOptimized,   // Optimized for storage
    ComputeIntensive,   // Optimized for complex computations
    GeneralPurpose,     // Balanced performance
}

impl ShardType {
    /// Get optimization parameters for the shard type
    pub fn get_optimization_params(&self) -> ShardOptimizationParams {
        match self {
            ShardType::HighPerformance => ShardOptimizationParams {
                max_transactions_per_block: 10000,
                target_confirmation_time_ms: 50,
                parallel_workers: 8,
                memory_limit_mb: 2048,
            },
            ShardType::StorageOptimized => ShardOptimizationParams {
                max_transactions_per_block: 5000,
                target_confirmation_time_ms: 100,
                parallel_workers: 4,
                memory_limit_mb: 4096,
            },
            ShardType::ComputeIntensive => ShardOptimizationParams {
                max_transactions_per_block: 2000,
                target_confirmation_time_ms: 200,
                parallel_workers: 16,
                memory_limit_mb: 8192,
            },
            ShardType::GeneralPurpose => ShardOptimizationParams {
                max_transactions_per_block: 7500,
                target_confirmation_time_ms: 75,
                parallel_workers: 6,
                memory_limit_mb: 3072,
            },
        }
    }
}

/// Optimization parameters for different shard types
#[derive(Debug, Clone)]
pub struct ShardOptimizationParams {
    pub max_transactions_per_block: u64,
    pub target_confirmation_time_ms: u64,
    pub parallel_workers: u32,
    pub memory_limit_mb: u64,
}

/// Information about a specific shard
#[derive(Debug)]
pub struct ShardInfo {
    pub id: u64,
    pub shard_type: ShardType,
    pub performance: ShardPerformance,
    pub parallel_capacity: u32,
    pub validator_count: u32,
    pub is_active: bool,
}

/// Configuration for the sharding system
#[derive(Debug, Clone)]
pub struct ShardingConfig {
    pub total_shards: u64,
    pub parallel_processing: bool,
    pub dynamic_allocation: bool,
    pub performance_monitoring: bool,
    pub cross_shard_batching: bool,
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            total_shards: 16,
            parallel_processing: true,
            dynamic_allocation: true,
            performance_monitoring: true,
            cross_shard_batching: true,
            load_balancing: LoadBalancingStrategy::PerformanceBased,
        }
    }
}

/// Load balancing strategies for shard distribution
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    PerformanceBased,
    // TODO: Custom load balancing strategy - temporarily disabled due to Clone constraints
    // Custom(Box<dyn Fn(&HashMap<u64, ShardInfo>) -> u64 + Send + Sync>),
}

/// Shard load information
#[derive(Debug, Clone)]
pub struct ShardLoad {
    pub shard_id: ShardId,
    pub current_load: f64,
    pub capacity: f64,
    pub response_time_ms: u64,
    pub error_rate: f64,
}

/// Manager for the entire sharding system
pub struct ShardManager {
    shards: Arc<RwLock<HashMap<u64, ShardInfo>>>,
    config: ShardingConfig,
    performance_monitor: ShardPerformanceMonitor,
    load_balancer: ShardLoadBalancer,
}

impl ShardManager {
    /// Create a new shard manager
    pub fn new(config: ShardingConfig) -> Self {
        let shards = Arc::new(RwLock::new(HashMap::new()));
        let performance_monitor = ShardPerformanceMonitor::new();
        let load_balancer = ShardLoadBalancer::new(config.load_balancing.clone());

        Self {
            shards,
            config,
            performance_monitor,
            load_balancer,
        }
    }

    /// Register a new shard
    pub async fn register_shard(&self, shard_id: u64, shard_type: ShardType, parallel_capacity: u32) -> Result<(), String> {
        let info = ShardInfo {
            id: shard_id,
            shard_type: shard_type.clone(),
            performance: ShardPerformance {
                transactions_processed: AtomicU64::new(0),
                blocks_created: AtomicU64::new(0),
                average_processing_time: AtomicU64::new(0),
                current_load: AtomicU64::new(0),
                last_activity: AtomicU64::new(0),
            },
            parallel_capacity,
            validator_count: 0,
            is_active: true,
        };

        let mut shards = self.shards.write().await;
        shards.insert(shard_id, info);
        
        // Initialize monitoring for the new shard
        self.performance_monitor.initialize_shard_monitoring(shard_id).await;
        
        Ok(())
    }

    /// Get information about a specific shard
    pub async fn get_shard_info(&self, shard_id: u64) -> Option<ShardInfo> {
        let shards = self.shards.read().await;
        shards.get(&shard_id).map(|info| ShardInfo {
            id: info.id,
            shard_type: info.shard_type.clone(),
            performance: ShardPerformance {
                transactions_processed: AtomicU64::new(info.performance.transactions_processed.load(Ordering::Relaxed)),
                blocks_created: AtomicU64::new(info.performance.blocks_created.load(Ordering::Relaxed)),
                average_processing_time: AtomicU64::new(info.performance.average_processing_time.load(Ordering::Relaxed)),
                current_load: AtomicU64::new(info.performance.current_load.load(Ordering::Relaxed)),
                last_activity: AtomicU64::new(info.performance.last_activity.load(Ordering::Relaxed)),
            },
            parallel_capacity: info.parallel_capacity,
            validator_count: info.validator_count,
            is_active: info.is_active,
        })
    }

    /// Get the optimal shard for a transaction
    pub async fn get_optimal_shard(&self, _transaction_data: &TransactionData) -> Option<u64> {
        let shards = self.shards.read().await;
        self.load_balancer.get_optimal_shard(&shards)
    }

    /// Process a cross-shard transaction
    pub async fn process_cross_shard_transaction(
        &self,
        _source_shard: u64,
        _target_shard: u64,
        _transaction_data: &TransactionData,
    ) -> Result<(), String> {
        // Record cross-shard transaction
        self.performance_monitor.record_cross_shard_transaction(_source_shard, _target_shard).await;
        
        // Simulate cross-shard processing
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        Ok(())
    }

    /// Get all shards
    pub async fn get_all_shards(&self) -> Vec<ShardInfo> {
        let shards = self.shards.read().await;
        shards.values().map(|info| ShardInfo {
            id: info.id,
            shard_type: info.shard_type.clone(),
            performance: ShardPerformance {
                transactions_processed: AtomicU64::new(info.performance.transactions_processed.load(Ordering::Relaxed)),
                blocks_created: AtomicU64::new(info.performance.blocks_created.load(Ordering::Relaxed)),
                average_processing_time: AtomicU64::new(info.performance.average_processing_time.load(Ordering::Relaxed)),
                current_load: AtomicU64::new(info.performance.current_load.load(Ordering::Relaxed)),
                last_activity: AtomicU64::new(info.performance.last_activity.load(Ordering::Relaxed)),
            },
            parallel_capacity: info.parallel_capacity,
            validator_count: info.validator_count,
            is_active: info.is_active,
        }).collect()
    }

    /// Get sharding configuration
    pub fn get_config(&self) -> &ShardingConfig {
        &self.config
    }

    /// Get performance monitor
    pub fn get_performance_monitor(&self) -> &ShardPerformanceMonitor {
        &self.performance_monitor
    }

    /// Get load balancer
    pub fn get_load_balancer(&self) -> &ShardLoadBalancer {
        &self.load_balancer
    }
}

/// Monitor for shard performance metrics
pub struct ShardPerformanceMonitor {
    cross_shard_transactions: Arc<RwLock<HashMap<(u64, u64), u64>>>,
    global_metrics: Arc<RwLock<GlobalShardMetrics>>,
}

impl ShardPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            cross_shard_transactions: Arc::new(RwLock::new(HashMap::new())),
            global_metrics: Arc::new(RwLock::new(GlobalShardMetrics {
                total_transactions: 0,
                total_blocks: 0,
                average_tps: 0.0,
                peak_tps: 0,
            })),
        }
    }

    /// Initialize monitoring for a specific shard
    pub async fn initialize_shard_monitoring(&self, shard_id: u64) {
        let mut cross_shard = self.cross_shard_transactions.write().await;
        cross_shard.insert((shard_id, shard_id), 0);
    }

    /// Record a cross-shard transaction
    pub async fn record_cross_shard_transaction(&self, source_shard: u64, target_shard: u64) {
        let mut cross_shard = self.cross_shard_transactions.write().await;
        let key = (source_shard, target_shard);
        *cross_shard.entry(key).or_insert(0) += 1;
    }

    /// Update metrics for a specific shard
    pub async fn update_shard_metrics(&self, shard_id: u64, transactions: u64, blocks: u64) {
        let mut global = self.global_metrics.write().await;
        global.total_transactions += transactions;
        global.total_blocks += blocks;
        
        // Update TPS calculation
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        if current_time > 0 {
            global.average_tps = global.total_transactions as f64 / current_time as f64;
            if global.average_tps > global.peak_tps as f64 {
                global.peak_tps = global.average_tps as u64;
            }
        }
    }

    /// Get global shard metrics
    pub async fn get_global_metrics(&self) -> GlobalShardMetrics {
        self.global_metrics.read().await.clone()
    }
}

/// Global metrics across all shards
#[derive(Debug, Clone)]
pub struct GlobalShardMetrics {
    pub total_transactions: u64,
    pub total_blocks: u64,
    pub average_tps: f64,
    pub peak_tps: u64,
}

/// Load balancer for distributing transactions across shards
pub struct ShardLoadBalancer {
    strategy: LoadBalancingStrategy,
    current_shard_index: AtomicU64,
}

impl ShardLoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            current_shard_index: AtomicU64::new(0),
        }
    }

    /// Get the optimal shard based on the current strategy
    pub fn get_optimal_shard(&self, shards: &HashMap<u64, ShardInfo>) -> Option<u64> {
        if shards.is_empty() {
            return None;
        }

        match &self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let current = self.current_shard_index.fetch_add(1, Ordering::Relaxed);
                let shard_ids: Vec<u64> = shards.keys().cloned().collect();
                if !shard_ids.is_empty() {
                    Some(shard_ids[current as usize % shard_ids.len()])
                } else {
                    None
                }
            }
            LoadBalancingStrategy::LeastLoaded => {
                shards.iter()
                    .min_by_key(|(_, info)| info.performance.current_load.load(Ordering::Relaxed))
                    .map(|(id, _)| *id)
            }
            LoadBalancingStrategy::PerformanceBased => {
                shards.iter()
                    .max_by_key(|(_, info)| {
                        let tps = info.performance.transactions_processed.load(Ordering::Relaxed);
                        let load = info.performance.current_load.load(Ordering::Relaxed);
                        if load > 0 { tps / load } else { tps }
                    })
                    .map(|(id, _)| *id)
            }
            // TODO: Custom load balancing strategy - temporarily disabled
            // LoadBalancingStrategy::Custom(func) => {
            //     Some(func(shards))
            // }
        }
    }

    /// Calculate the current load of a shard
    pub fn calculate_shard_load(&self, shard: &ShardInfo) -> f64 {
        let current_load = shard.performance.current_load.load(Ordering::Relaxed) as f64;
        let capacity = shard.parallel_capacity as f64;
        
        if capacity > 0.0 {
            current_load / capacity
        } else {
            1.0 // Maximum load if capacity is 0
        }
    }

    /// Calculate performance score for a shard
    pub fn calculate_performance_score(&self, shard: &ShardInfo) -> f64 {
        let tps = shard.performance.transactions_processed.load(Ordering::Relaxed) as f64;
        let blocks = shard.performance.blocks_created.load(Ordering::Relaxed) as f64;
        let avg_time = shard.performance.average_processing_time.load(Ordering::Relaxed) as f64;
        
        // Higher score for higher TPS, more blocks, and lower processing time
        let time_factor = if avg_time > 0.0 { 1000.0 / avg_time } else { 0.0 };
        (tps * 0.4) + (blocks * 0.3) + (time_factor * 0.3)
    }

    /// Get current load for a specific shard
    pub fn get_shard_load(&self, shard: &ShardInfo) -> f64 {
        self.calculate_shard_load(shard)
    }

    /// Rebalance shards based on current performance
    pub async fn rebalance_shards(&self, shards: &mut HashMap<u64, ShardInfo>) {
        let mut shard_performance: Vec<(u64, f64)> = shards.iter()
            .map(|(id, info)| (*id, self.calculate_performance_score(info)))
            .collect();
        
        // Sort by performance score
        shard_performance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Adjust parallel capacity based on performance
        for (shard_id, _score) in shard_performance.iter().take(10) { // Top 10 performers
            if let Some(shard) = shards.get_mut(shard_id) {
                let current_capacity = shard.parallel_capacity;
                let new_capacity = (current_capacity as f64 * 1.1) as u32; // Increase by 10%
                shard.parallel_capacity = new_capacity.min(32); // Cap at 32
            }
        }
    }
}

/// Transaction data for shard assignment
#[derive(Debug, Clone)]
pub struct TransactionData {
    pub from: Address,
    pub to: Address,
    pub value: u64,
    pub nonce: u64,
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_type_optimization_params() {
        let high_perf = ShardType::HighPerformance;
        let params = high_perf.get_optimization_params();
        
        assert_eq!(params.max_transactions_per_block, 10000);
        assert_eq!(params.target_confirmation_time_ms, 50);
        assert_eq!(params.parallel_workers, 8);
    }

    #[tokio::test]
    async fn test_shard_manager_creation() {
        let config = ShardingConfig::default();
        let manager = ShardManager::new(config);
        
        assert_eq!(manager.get_config().total_shards, 16);
    }

    #[tokio::test]
    async fn test_shard_registration() {
        let config = ShardingConfig::default();
        let manager = ShardManager::new(config);
        
        let result = manager.register_shard(0, ShardType::HighPerformance, 16).await;
        
        assert!(result.is_ok());
    }
}
