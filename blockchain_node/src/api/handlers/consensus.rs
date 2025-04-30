use std::sync::Arc;
use axum::{extract::Extension, Json};
use serde::Serialize;

use crate::consensus::svbft::{SVBFTConsensus, ConsensusPhase};
use crate::api::ApiError;

/// Response for consensus status
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
    let leader = consensus.get_current_leader().await.unwrap_or_else(|| "unknown".to_string());
    let quorum_size = consensus.get_quorum_size().await.unwrap_or(0);
    let phase = consensus.get_current_phase().await.unwrap_or(ConsensusPhase::New);
    let finalized_blocks = consensus.get_finalized_blocks().await;
    
    // Find the highest finalized block
    let finalized_height = finalized_blocks.values()
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