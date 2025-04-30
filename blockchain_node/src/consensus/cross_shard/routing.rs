use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::consensus::metrics::NetworkMetrics;
use crate::types::{ShardId, NodeId, TransactionHash};

pub struct AdaptiveRouter {
    // Track latency between shards
    shard_latencies: Arc<RwLock<HashMap<(ShardId, ShardId), f64>>>,
    // Track load on each shard
    shard_loads: Arc<RwLock<HashMap<ShardId, f64>>>,
    // Track successful routes
    route_history: Arc<RwLock<BTreeMap<TransactionHash, Vec<ShardId>>>>,
    // Network metrics
    metrics: Arc<NetworkMetrics>,
}

impl AdaptiveRouter {
    pub fn new(metrics: Arc<NetworkMetrics>) -> Self {
        Self {
            shard_latencies: Arc::new(RwLock::new(HashMap::new())),
            shard_loads: Arc::new(RwLock::new(HashMap::new())),
            route_history: Arc::new(RwLock::new(BTreeMap::new())),
            metrics,
        }
    }

    pub async fn update_latency(&self, source: ShardId, target: ShardId, latency: f64) {
        let mut latencies = self.shard_latencies.write().await;
        latencies.insert((source, target), latency);
        
        // Update metrics
        self.metrics.record_shard_latency(source, target, latency);
    }

    pub async fn update_load(&self, shard: ShardId, load: f64) {
        let mut loads = self.shard_loads.write().await;
        loads.insert(shard, load);
        
        // Update metrics
        self.metrics.record_shard_load(shard, load);
    }

    pub async fn get_optimal_route(&self, source: ShardId, target: ShardId, tx_hash: TransactionHash) -> Vec<ShardId> {
        let latencies = self.shard_latencies.read().await;
        let loads = self.shard_loads.read().await;
        
        // Calculate optimal route based on latency and load
        let mut route = vec![source];
        let mut current = source;
        
        while current != target {
            let next = self.find_next_hop(current, target, &latencies, &loads).await;
            route.push(next);
            current = next;
        }
        
        // Store route for future reference
        let mut history = self.route_history.write().await;
        history.insert(tx_hash, route.clone());
        
        route
    }

    async fn find_next_hop(
        &self,
        current: ShardId,
        target: ShardId,
        latencies: &HashMap<(ShardId, ShardId), f64>,
        loads: &HashMap<ShardId, f64>
    ) -> ShardId {
        let mut best_score = f64::MAX;
        let mut best_next = target;

        // Weight factors for scoring
        const LATENCY_WEIGHT: f64 = 0.7;
        const LOAD_WEIGHT: f64 = 0.3;

        for (pair, latency) in latencies.iter() {
            if pair.0 != current {
                continue;
            }

            let next_shard = pair.1;
            let load = loads.get(&next_shard).unwrap_or(&0.0);
            
            // Calculate weighted score
            let score = LATENCY_WEIGHT * latency + LOAD_WEIGHT * load;
            
            if score < best_score {
                best_score = score;
                best_next = next_shard;
            }
        }

        best_next
    }

    pub async fn record_route_success(&self, tx_hash: TransactionHash) {
        let mut history = self.route_history.write().await;
        if let Some(route) = history.get(&tx_hash) {
            // Update success metrics for this route
            self.metrics.record_successful_route(route.clone());
        }
    }

    pub async fn record_route_failure(&self, tx_hash: TransactionHash) {
        let mut history = self.route_history.write().await;
        if let Some(route) = history.get(&tx_hash) {
            // Update failure metrics for this route
            self.metrics.record_failed_route(route.clone());
        }
    }
} 