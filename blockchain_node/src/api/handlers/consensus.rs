#[cfg(not(skip_problematic_modules))]
use axum::{extract::Extension, Json};
#[cfg(not(skip_problematic_modules))]
use serde::Serialize;
#[cfg(not(skip_problematic_modules))]
use std::sync::Arc;

#[cfg(not(skip_problematic_modules))]
use crate::api::ApiError;
#[cfg(not(skip_problematic_modules))]
use crate::consensus::svbft::{ConsensusPhase, SVBFTConsensus};

use crate::consensus::svcp::SVCPMiner;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Response for consensus status
#[cfg(not(skip_problematic_modules))]
#[derive(Serialize)]
pub struct ConsensusStatusResponse {
    /// Current view number
    pub view: u64,
    /// Current phase
    pub phase: String,
    /// Current leader
    pub leader: String,
    /// Quorum size
    pub quorum_size: usize,
    /// Number of validators
    pub validator_count: usize,
    /// Latest finalized block height
    pub finalized_height: u64,
}

/// Get consensus status
#[cfg(not(skip_problematic_modules))]
pub async fn get_consensus_status(
    Extension(consensus): Extension<Option<Arc<SVBFTConsensus>>>,
) -> Result<Json<ConsensusStatusResponse>, ApiError> {
    let consensus = match consensus {
        Some(c) => c,
        None => {
            return Err(ApiError {
                status: 503,
                message: "Consensus engine not available".to_string(),
            });
        }
    };

    // Get consensus status from the SVBFT consensus engine
    let view = consensus.get_current_view().await;
    let leader = consensus
        .get_current_leader()
        .await
        .unwrap_or_else(|| "unknown".to_string());
    let quorum_size = consensus.get_quorum_size().await.unwrap_or(0);
    let phase = consensus
        .get_current_phase()
        .await
        .unwrap_or(ConsensusPhase::New);
    let finalized_blocks = consensus.get_finalized_blocks().await;

    // Find the highest finalized block
    let finalized_height = finalized_blocks
        .values()
        .map(|block| block.header.height)
        .max()
        .unwrap_or(0);

    // Convert phase to string
    let phase_str = match phase {
        ConsensusPhase::New => "New".to_string(),
        ConsensusPhase::Prepare => "Prepare".to_string(),
        ConsensusPhase::PreCommit => "PreCommit".to_string(),
        ConsensusPhase::Commit => "Commit".to_string(),
        ConsensusPhase::Decide => "Decide".to_string(),
    };

    Ok(Json(ConsensusStatusResponse {
        view,
        phase: phase_str,
        leader,
        quorum_size,
        validator_count: finalized_blocks.len(), // Approximate
        finalized_height,
    }))
}

/// Consensus status response
#[derive(serde::Serialize)]
pub struct ConsensusStatusResponse {
    /// Current difficulty
    pub difficulty: u64,
    /// Current proposers
    pub proposers: Vec<String>,
    /// This node is a proposer
    pub is_proposer: bool,
    /// Estimated TPS
    pub estimated_tps: f32,
    /// Consensus mechanism
    pub mechanism: String,
}

/// Get the current consensus status
pub async fn get_consensus_status(miner: &Arc<RwLock<SVCPMiner>>) -> ConsensusStatusResponse {
    let miner_guard = miner.read().await;

    let difficulty = miner_guard.get_difficulty().await;
    let proposers = miner_guard.get_proposers().await;
    let node_id = miner_guard.get_node_id();
    let estimated_tps = miner_guard.get_estimated_tps();

    ConsensusStatusResponse {
        difficulty,
        proposers: proposers.clone(),
        is_proposer: proposers.contains(&node_id),
        estimated_tps,
        mechanism: "SVCP".to_string(),
    }
}

#[cfg(skip_problematic_modules)]
pub async fn get_consensus_status_str() -> String {
    "Consensus engine not available in this build".to_string()
}
