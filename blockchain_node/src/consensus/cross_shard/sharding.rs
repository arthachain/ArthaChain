use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Shard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// Shard ID
    pub id: u32,
    /// Number of validator nodes
    pub validator_count: u32,
    /// Minimum stake requirement
    pub min_stake: u64,
    /// Maximum transactions per block
    pub max_txs_per_block: u32,
}

/// Shard status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardStatus {
    /// Shard is active and processing transactions
    Active,
    /// Shard is syncing with the network
    Syncing {
        current_block: u64,
        target_block: u64,
    },
    /// Shard is inactive or has failed
    Inactive { reason: String },
}

/// Cross-shard routing table
#[derive(Debug)]
pub struct RoutingTable {
    /// Routes between shards
    routes: HashMap<(u32, u32), Vec<u32>>,
    /// Shard statuses
    shard_status: HashMap<u32, ShardStatus>,
}

impl Default for RoutingTable {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingTable {
    /// Create a new routing table
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
            shard_status: HashMap::new(),
        }
    }

    /// Add a route between shards
    pub fn add_route(&mut self, from_shard: u32, to_shard: u32, intermediate_shards: Vec<u32>) {
        self.routes
            .insert((from_shard, to_shard), intermediate_shards);
    }

    /// Get the route between two shards
    pub fn get_route(&self, from_shard: u32, to_shard: u32) -> Option<&Vec<u32>> {
        self.routes.get(&(from_shard, to_shard))
    }

    /// Update shard status
    pub fn update_status(&mut self, shard_id: u32, status: ShardStatus) {
        self.shard_status.insert(shard_id, status);
    }

    /// Check if a shard is active
    pub fn is_shard_active(&self, shard_id: u32) -> bool {
        matches!(self.shard_status.get(&shard_id), Some(ShardStatus::Active))
    }
}

/// Shard manager for handling cross-shard operations
pub struct ShardManager {
    /// Shard configurations
    configs: HashMap<u32, ShardConfig>,
    /// Routing table
    routing: RoutingTable,
}

impl Default for ShardManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardManager {
    /// Create a new shard manager
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            routing: RoutingTable::new(),
        }
    }

    /// Register a new shard
    pub fn register_shard(&mut self, config: ShardConfig) {
        self.configs.insert(config.id, config.clone());
        self.routing.update_status(config.id, ShardStatus::Active);
    }

    /// Find the optimal route for cross-shard communication
    pub async fn find_route(&self, from_shard: u32, to_shard: u32) -> Result<Vec<u32>> {
        if let Some(route) = self.routing.get_route(from_shard, to_shard) {
            // Check if all shards in the route are active
            if route
                .iter()
                .all(|&shard| self.routing.is_shard_active(shard))
            {
                Ok(route.clone())
            } else {
                Err(anyhow::anyhow!("Some shards in the route are inactive"))
            }
        } else {
            Err(anyhow::anyhow!("No route found between shards"))
        }
    }
}
