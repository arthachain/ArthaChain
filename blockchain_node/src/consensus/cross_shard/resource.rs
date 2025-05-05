use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Resource types that can be shared across shards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    /// Computational resources
    Compute {
        cpu_cores: u32,
        memory_mb: u64,
    },
    /// Storage resources
    Storage {
        capacity_gb: u64,
        used_gb: u64,
    },
    /// Network bandwidth
    Bandwidth {
        mbps: u64,
        latency_ms: u32,
    },
}

/// Resource allocation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStatus {
    /// Resource is available
    Available,
    /// Resource is partially allocated
    PartiallyAllocated {
        used_percentage: f32,
    },
    /// Resource is fully allocated
    FullyAllocated,
}

/// Resource manager for cross-shard resource sharing
pub struct ResourceManager {
    /// Available resources per shard
    resources: HashMap<u32, Vec<ResourceType>>,
    /// Resource allocation status
    allocations: HashMap<u32, AllocationStatus>,
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            allocations: HashMap::new(),
        }
    }

    /// Register resources for a shard
    pub fn register_resources(&mut self, shard_id: u32, resources: Vec<ResourceType>) {
        self.resources.insert(shard_id, resources);
        self.allocations.insert(shard_id, AllocationStatus::Available);
    }

    /// Request resources from another shard
    pub async fn request_resources(&mut self, from_shard: u32, _to_shard: u32, _resource_type: ResourceType) -> Result<bool> {
        // Check if resources are available
        if let Some(resources) = self.resources.get(&from_shard) {
            // Check if requested resource type is available
            if resources.iter().any(|r| matches!(r, _resource_type)) {
                // Update allocation status
                self.allocations.insert(from_shard, AllocationStatus::PartiallyAllocated {
                    used_percentage: 0.5, // Example value
                });
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    /// Release resources back to the original shard
    pub async fn release_resources(&mut self, shard_id: u32) -> Result<()> {
        self.allocations.insert(shard_id, AllocationStatus::Available);
        Ok(())
    }
} 