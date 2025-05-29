pub mod parallel_processor;
pub mod svcp;

// Make cross_shard and reputation modules always available
pub mod cross_shard;
pub mod reputation;

#[cfg(not(skip_problematic_modules))]
pub mod byzantine;

#[cfg(not(skip_problematic_modules))]
pub mod adaptive;

// SVBFT modules
pub mod quantum_svbft;
pub mod svbft;
pub mod view_change;

#[cfg(not(skip_problematic_modules))]
pub mod leader_election;

pub mod batch;
#[cfg(not(skip_problematic_modules))]
pub mod batch_validation;
#[cfg(not(skip_problematic_modules))]
pub mod checkpoint;
#[cfg(not(skip_problematic_modules))]
pub mod dag;
#[cfg(not(skip_problematic_modules))]
pub mod incentives;
#[cfg(not(skip_problematic_modules))]
pub mod security;
#[cfg(not(skip_problematic_modules))]
pub mod types;
#[cfg(not(skip_problematic_modules))]
pub mod validation;
#[cfg(not(skip_problematic_modules))]
pub mod vote_aggregation;

// Re-exports
pub use batch::BatchProcessor;
#[cfg(not(skip_problematic_modules))]
pub use checkpoint::CheckpointManager;
#[cfg(not(skip_problematic_modules))]
pub use cross_shard::CrossShardManager;
// Quantum-resistant enhanced cross-shard coordinator
pub use cross_shard::{
    CoordinatorMessage, CrossShardCoordinator, EnhancedCrossShardManager, ParticipantHandler,
    TxPhase,
};
#[cfg(not(skip_problematic_modules))]
pub use dag::DagManager;
#[cfg(not(skip_problematic_modules))]
pub use incentives::IncentiveManager;
pub use parallel_processor::ParallelProcessor;
#[cfg(not(skip_problematic_modules))]
pub use security::SecurityManager;
pub use svcp::SVCPMiner;
#[cfg(not(skip_problematic_modules))]
pub use types::{ConsensusMessage, ConsensusState, ConsensusType};
#[cfg(not(skip_problematic_modules))]
pub use validation::ValidationEngine;
#[cfg(not(skip_problematic_modules))]
pub use vote_aggregation::VoteAggregator;

// AI enhanced detection engines
#[cfg(not(skip_problematic_modules))]
pub mod advanced_detection;
#[cfg(not(skip_problematic_modules))]
pub mod anomaly_detection;
#[cfg(not(skip_problematic_modules))]
pub mod fraud_detection;
#[cfg(not(skip_problematic_modules))]
pub mod social_graph;
#[cfg(not(skip_problematic_modules))]
pub mod weight_adjustment;

// Re-exports of AI engines
#[cfg(not(skip_problematic_modules))]
pub use advanced_detection::AdvancedDetectionEngine;
#[cfg(not(skip_problematic_modules))]
pub use anomaly_detection::AnomalyDetector;
#[cfg(not(skip_problematic_modules))]
pub use fraud_detection::FraudDetectionEngine;
#[cfg(not(skip_problematic_modules))]
pub use social_graph::SocialGraph;
#[cfg(not(skip_problematic_modules))]
pub use weight_adjustment::DynamicWeightAdjuster;

pub use quantum_svbft::QuantumSVBFTConsensus;
pub use view_change::ViewChangeManager;
