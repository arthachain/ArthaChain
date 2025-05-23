use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Represents a node's social connections and influence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialNode {
    /// Node ID
    pub id: String,
    /// Reputation score
    pub reputation: f64,
    /// Last active timestamp
    pub last_active: SystemTime,
    /// Direct connections (node_id -> trust_score)
    pub connections: Vec<String>,
    /// Reputation scores from different platforms
    pub metrics: HashMap<String, f64>,
}

impl SocialNode {
    pub fn new(id: String) -> Self {
        Self {
            id,
            reputation: 0.5,
            last_active: SystemTime::now(),
            connections: Vec::new(),
            metrics: HashMap::new(),
        }
    }

    pub fn add_connection(&mut self, node_id: String) {
        if !self.connections.contains(&node_id) {
            self.connections.push(node_id);
        }
    }

    pub fn update_metric(&mut self, key: &str, value: f64) {
        self.metrics.insert(key.to_string(), value);
    }
}

/// Record of interactions between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRecord {
    /// Timestamp of interaction
    pub timestamp: std::time::SystemTime,
    /// Type of interaction
    pub interaction_type: InteractionType,
    /// Target node ID
    pub target_node: String,
    /// Interaction weight/impact
    pub weight: f32,
    /// Outcome score (-1 to 1)
    pub outcome: f32,
}

/// Types of interactions between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    /// Transaction validation
    Validation,
    /// Cross-shard communication
    CrossShard,
    /// Resource sharing
    ResourceSharing,
    /// Governance participation
    Governance,
    /// Community contribution
    Community,
}

/// Community participation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityMetrics {
    /// Governance participation rate
    pub governance_participation: f32,
    /// Resource contribution score
    pub resource_contribution: f32,
    /// Community support score
    pub community_support: f32,
    /// Innovation contribution
    pub innovation_score: f32,
    /// Long-term engagement
    pub engagement_duration: u64,
}

/// Time-based decay parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeDecayParams {
    /// Half-life in seconds
    pub half_life_secs: u64,
    /// Minimum value
    pub min_value: f64,
    /// Maximum age in seconds
    pub max_age_secs: u64,
}

impl Default for TimeDecayParams {
    fn default() -> Self {
        Self {
            half_life_secs: 86400 * 7, // One week
            min_value: 0.1,
            max_age_secs: 86400 * 30, // One month
        }
    }
}

/// Weight adjustment parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightParameters {
    /// Base weight for direct interactions
    pub direct_weight: f64,
    /// Weight decay factor for indirect connections
    pub decay_factor: f64,
    /// Minimum weight threshold
    pub min_weight: f64,
    /// Platform importance weights
    pub platform_weights: HashMap<String, f64>,
    /// Time-based decay parameters
    pub time_decay: TimeDecayParams,
}

impl Default for WeightParameters {
    fn default() -> Self {
        Self {
            direct_weight: 1.0,
            decay_factor: 0.85,
            min_weight: 0.1,
            platform_weights: HashMap::new(),
            time_decay: TimeDecayParams::default(),
        }
    }
}

/// Cache for analysis results
#[derive(Debug, Default)]
struct AnalysisCache {
    /// PageRank scores
    influence_scores: HashMap<String, f64>,
    /// Last update timestamp
    last_update: Option<SystemTime>,
    /// Cache validity duration
    validity_duration: Duration,
}

/// Social graph analysis engine
pub struct SocialGraph {
    /// Social graph structure
    nodes: Arc<RwLock<HashMap<String, SocialNode>>>,
    /// Weight adjustment parameters
    weight_params: Arc<RwLock<WeightParameters>>,
    /// Analysis cache
    cache: Arc<RwLock<AnalysisCache>>,
}

impl SocialGraph {
    /// Create a new social graph analyzer
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            weight_params: Arc::new(RwLock::new(WeightParameters::default())),
            cache: Arc::new(RwLock::new(AnalysisCache::default())),
        }
    }

    /// Add or update a node in the social graph
    pub async fn add_or_update_node(&self, node: SocialNode) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        nodes.insert(node.id.clone(), node);

        // Invalidate cache when graph changes
        let mut cache = self.cache.write().await;
        cache.last_update = None;

        Ok(())
    }

    /// Calculate node influence using PageRank
    pub async fn calculate_influence(&self) -> Result<HashMap<String, f64>> {
        let mut cache = self.cache.write().await;

        // Check if cache is valid
        let should_recalculate = match cache.last_update {
            None => true,
            Some(last_update) => match SystemTime::now().duration_since(last_update) {
                Ok(elapsed) => elapsed > cache.validity_duration,
                Err(_) => true,
            },
        };

        if !should_recalculate {
            return Ok(cache.influence_scores.clone());
        }

        // Simple influence calculation (can be improved later)
        let nodes = self.nodes.read().await;
        let mut scores = HashMap::new();

        for (id, node) in nodes.iter() {
            // Basic score based on connections and reputation
            let connection_score = node.connections.len() as f64 * 0.1;
            let reputation_factor = node.reputation;
            let metric_avg = if !node.metrics.is_empty() {
                node.metrics.values().sum::<f64>() / node.metrics.len() as f64
            } else {
                0.5
            };

            let influence = (connection_score * 0.4 + reputation_factor * 0.4 + metric_avg * 0.2)
                .min(1.0)
                .max(0.0);

            scores.insert(id.clone(), influence);
        }

        // Update cache
        cache.influence_scores = scores.clone();
        cache.last_update = Some(SystemTime::now());
        cache.validity_duration = Duration::from_secs(300); // 5 minutes

        Ok(scores)
    }

    /// Get node's social score
    pub async fn get_social_score(&self, node_id: &str) -> Result<f64> {
        let influence_scores = self.calculate_influence().await?;

        if let Some(node) = self.get_node(node_id).await {
            // Combine different factors
            let influence = influence_scores.get(node_id).unwrap_or(&0.0);
            let metric_avg = if !node.metrics.is_empty() {
                node.metrics.values().sum::<f64>() / node.metrics.len() as f64
            } else {
                0.5
            };

            let community_score = calculate_community_score(&node.metrics);

            return Ok(
                (*influence * 0.4 + metric_avg * 0.3 + community_score * 0.3)
                    .min(1.0)
                    .max(0.0),
            );
        }

        Err(anyhow!("Node not found"))
    }

    /// Update node weights based on recent interactions
    pub async fn update_weights(&self) -> Result<()> {
        let params = self.weight_params.read().await;
        let mut nodes = self.nodes.write().await;
        let now = SystemTime::now();

        for node in nodes.values_mut() {
            // Apply time decay to reputation
            if let Ok(elapsed) = now.duration_since(node.last_active) {
                let decay_factor = calculate_time_decay(
                    elapsed.as_secs(),
                    params.time_decay.half_life_secs,
                    params.time_decay.min_value,
                    params.time_decay.max_age_secs,
                );

                node.reputation *= decay_factor;
                node.reputation = node.reputation.max(params.min_weight);
            }
        }

        Ok(())
    }

    pub async fn get_node(&self, id: &str) -> Option<SocialNode> {
        let nodes = self.nodes.read().await;
        nodes.get(id).cloned()
    }
}

fn calculate_time_decay(elapsed_secs: u64, half_life: u64, min_value: f64, max_age: u64) -> f64 {
    if elapsed_secs >= max_age {
        return min_value;
    }

    let decay = (0.5f64).powf(elapsed_secs as f64 / half_life as f64);
    (decay * (1.0 - min_value) + min_value).max(min_value)
}

/// Calculate community participation score
fn calculate_community_score(metrics: &HashMap<String, f64>) -> f64 {
    let governance_weight = 0.3;
    let resource_weight = 0.2;
    let support_weight = 0.2;
    let innovation_weight = 0.2;
    let engagement_weight = 0.1;

    let governance = metrics
        .get("governance_participation")
        .unwrap_or(&0.5)
        .min(1.0)
        .max(0.0);
    let resource = metrics
        .get("resource_contribution")
        .unwrap_or(&0.5)
        .min(1.0)
        .max(0.0);
    let support = metrics
        .get("community_support")
        .unwrap_or(&0.5)
        .min(1.0)
        .max(0.0);
    let innovation = metrics
        .get("innovation_score")
        .unwrap_or(&0.5)
        .min(1.0)
        .max(0.0);
    let engagement = metrics
        .get("engagement_duration")
        .unwrap_or(&0.5)
        .min(1.0)
        .max(0.0);

    governance * governance_weight
        + resource * resource_weight
        + support * support_weight
        + innovation * innovation_weight
        + engagement * engagement_weight
}
