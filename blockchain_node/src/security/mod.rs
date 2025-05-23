use log::info;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Mutex;

use crate::ai_engine::security::NodeScore;

/// SecurityManager handles validation and security checks
pub struct SecurityManager {
    /// Node scores by node ID
    node_scores: Arc<Mutex<HashMap<String, NodeScore>>>,
    /// Security policies
    security_policies: SecurityPolicies,
    /// Last update time
    last_update: SystemTime,
}

/// Security policies for the node
pub struct SecurityPolicies {
    /// Minimum score required for transaction validation
    min_score_for_validation: f32,
    /// Minimum score required for consensus participation
    min_score_for_consensus: f32,
    /// Minimum score required for block production
    min_score_for_block_production: f32,
    /// Ban threshold score
    ban_threshold: f32,
}

impl Default for SecurityPolicies {
    fn default() -> Self {
        Self {
            min_score_for_validation: 0.5,
            min_score_for_consensus: 0.6,
            min_score_for_block_production: 0.7,
            ban_threshold: 0.3,
        }
    }
}

impl SecurityManager {
    /// Create a new security manager
    pub fn new(node_scores: Arc<Mutex<HashMap<String, NodeScore>>>) -> Self {
        Self {
            node_scores,
            security_policies: SecurityPolicies::default(),
            last_update: SystemTime::now(),
        }
    }

    /// Check if a node is allowed to participate in validation
    pub async fn is_allowed_validator(&self, node_id: &str) -> bool {
        let scores = self.node_scores.lock().await;
        if let Some(score) = scores.get(node_id) {
            score.overall_score >= self.security_policies.min_score_for_validation
        } else {
            false
        }
    }

    /// Check if a node is allowed to participate in consensus
    pub async fn is_allowed_consensus_participant(&self, node_id: &str) -> bool {
        let scores = self.node_scores.lock().await;
        if let Some(score) = scores.get(node_id) {
            score.overall_score >= self.security_policies.min_score_for_consensus
        } else {
            false
        }
    }

    /// Check if a node is allowed to produce blocks
    pub async fn is_allowed_block_producer(&self, node_id: &str) -> bool {
        let scores = self.node_scores.lock().await;
        if let Some(score) = scores.get(node_id) {
            score.overall_score >= self.security_policies.min_score_for_block_production
        } else {
            false
        }
    }

    /// Get security status for all nodes
    pub async fn get_security_status(&self) -> HashMap<String, String> {
        let scores = self.node_scores.lock().await;
        let mut status = HashMap::new();

        for (node_id, score) in scores.iter() {
            let status_str = if score.overall_score < self.security_policies.ban_threshold {
                "banned"
            } else if score.overall_score < self.security_policies.min_score_for_validation {
                "restricted"
            } else if score.overall_score < self.security_policies.min_score_for_consensus {
                "validation_only"
            } else if score.overall_score < self.security_policies.min_score_for_block_production {
                "consensus_only"
            } else {
                "full_access"
            };

            status.insert(node_id.clone(), status_str.to_string());
        }

        status
    }

    /// Update security policies
    pub fn update_policies(&mut self, policies: SecurityPolicies) {
        self.security_policies = policies;
        self.last_update = SystemTime::now();
        info!("Security policies updated");
    }
}
