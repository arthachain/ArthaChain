pub mod svcp;
pub mod svbft;
pub mod sharding;
pub mod parallel_processor;
pub mod cross_shard;
pub mod reputation;
pub mod ltl;
pub mod petri_net;
pub mod verification;
pub mod view_change;
pub mod parallel_tx;
pub mod state_pruning;
pub mod difficulty;
pub mod validator_set;
pub mod validator_rotation;
pub mod proofs;
pub mod batch;
pub mod receipt;
pub mod social_graph;
pub mod weight_adjustment;
pub mod advanced_detection;

// Re-export commonly used types
pub use svcp::{SVCPConfig, SVCPMiner};
pub use svbft::{SVBFTConfig, SVBFTConsensus};
pub use social_graph::SocialGraph;
pub use weight_adjustment::DynamicWeightAdjuster;
pub use advanced_detection::AdvancedDetectionEngine; 