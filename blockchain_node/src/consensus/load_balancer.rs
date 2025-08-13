use crate::types::{NodeId, ShardId};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct DynamicLoadBalancer {
    // Track shard capacities
    shard_capacities: Arc<RwLock<HashMap<ShardId, u64>>>,
    // Track current load
    current_loads: Arc<RwLock<HashMap<ShardId, u64>>>,
    // Track node assignments
    node_assignments: Arc<RwLock<HashMap<NodeId, ShardId>>>,
    // Track shard performance
    shard_performance: Arc<RwLock<HashMap<ShardId, f64>>>,
}

impl DynamicLoadBalancer {
    pub fn new() -> Self {
        Self {
            shard_capacities: Arc::new(RwLock::new(HashMap::new())),
            current_loads: Arc::new(RwLock::new(HashMap::new())),
            node_assignments: Arc::new(RwLock::new(HashMap::new())),
            shard_performance: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn update_shard_capacity(&self, shard: ShardId, capacity: u64) {
        let mut capacities = self.shard_capacities.write().await;
        capacities.insert(shard, capacity);
    }

    pub async fn update_current_load(&self, shard: ShardId, load: u64) {
        let mut loads = self.current_loads.write().await;
        loads.insert(shard, load);

        // Check if rebalancing is needed
        if self.should_rebalance(shard, load).await {
            self.trigger_rebalancing().await;
        }
    }

    async fn should_rebalance(&self, shard: ShardId, current_load: u64) -> bool {
        let capacities = self.shard_capacities.read().await;
        let loads = self.current_loads.read().await;

        if let Some(&capacity) = capacities.get(&shard) {
            // Check if load is above 80% of capacity
            if current_load as f64 > 0.8 * capacity as f64 {
                return true;
            }

            // Check for load imbalance across shards
            let avg_load: f64 = loads.values().sum::<u64>() as f64 / loads.len() as f64;
            let load_diff = (current_load as f64 - avg_load).abs();
            if load_diff / avg_load > 0.2 {
                return true;
            }
        }

        false
    }

    async fn trigger_rebalancing(&self) {
        let capacities = self.shard_capacities.read().await;
        let loads = self.current_loads.read().await;
        let mut assignments = self.node_assignments.write().await;

        // Calculate optimal distribution
        let total_load: u64 = loads.values().sum();
        let total_capacity: u64 = capacities.values().sum();

        for (node, current_shard) in assignments.iter_mut() {
            let optimal_shard = self
                .find_optimal_shard(
                    node,
                    *current_shard,
                    &loads,
                    &capacities,
                    total_load,
                    total_capacity,
                )
                .await;

            if optimal_shard != *current_shard {
                *current_shard = optimal_shard;
            }
        }
    }

    async fn find_optimal_shard(
        &self,
        node: &NodeId,
        current_shard: ShardId,
        loads: &HashMap<ShardId, u64>,
        capacities: &HashMap<ShardId, u64>,
        total_load: u64,
        total_capacity: u64,
    ) -> ShardId {
        let mut best_score = f64::MAX;
        let mut optimal_shard = current_shard;

        for (&shard, &capacity) in capacities.iter() {
            let load = loads.get(&shard).unwrap_or(&0);

            // Calculate migration score based on:
            // 1. Load distribution
            // 2. Network locality
            // 3. Migration cost
            let load_ratio = *load as f64 / capacity as f64;
            let locality_score = self.calculate_locality_score(node, shard).await;
            let migration_cost = if shard == current_shard { 0.0 } else { 1.0 };

            let score = 0.5 * load_ratio + 0.3 * locality_score + 0.2 * migration_cost;

            if score < best_score {
                best_score = score;
                optimal_shard = shard;
            }
        }

        optimal_shard
    }

    async fn calculate_locality_score(&self, node: &NodeId, shard: ShardId) -> f64 {
        // Calculate network locality score based on latency and topology
        // This is a simplified version - in production we would use actual network metrics
        let performance = self.shard_performance.read().await;
        performance.get(&shard).unwrap_or(&0.5).clone()
    }

    pub async fn record_transaction_latency(&self, shard: ShardId, latency: f64) {
        let mut performance = self.shard_performance.write().await;
        let current = performance.entry(shard).or_insert(0.0);
        // Exponential moving average
        *current = 0.9 * *current + 0.1 * latency;
    }
}
