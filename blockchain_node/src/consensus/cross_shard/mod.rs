pub mod coordinator;
pub mod integration;
pub mod merkle_proof;
pub mod protocol;
pub mod resource;
pub mod routing;
pub mod sharding;

// Re-export main types
pub use coordinator::{CoordinatorConfig, CrossShardCoordinator};
pub use integration::EnhancedCrossShardManager; // Fixed: use actual struct name
pub use merkle_proof::{MerkleProof, ProvenTransaction};
pub use protocol::{CrossShardTxType, TransactionCoordination};
pub use resource::ResourceManager;
pub use routing::AdaptiveRouter; // Fixed: use actual struct name
pub use sharding::ShardManager;

#[cfg(test)]
mod tests;
