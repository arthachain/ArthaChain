use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use crate::ai_engine::security::NodeScore;
use crate::consensus::social_graph::SocialGraph;
use std::time::{Duration, SystemTime};

/// Weight adjustment parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightAdjustmentParams {
    /// Base weights for different metrics
    pub base_weights: MetricWeights,
    /// Dynamic adjustment factors
    pub adjustment_factors: AdjustmentFactors,
    /// Time-based parameters
    pub time_params: TimeParameters,
    /// Network health thresholds
    pub health_thresholds: HealthThresholds,
}

/// Base weights for different metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricWeights {
    /// Device health weight
    pub device_health: f32,
    /// Network performance weight
    pub network_perf: f32,
    /// Storage contribution weight
    pub storage: f32,
    /// Social engagement weight
    pub social: f32,
    /// Security score weight
    pub security: f32,
}

/// Dynamic adjustment factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustmentFactors {
    /// Network health multiplier
    pub network_health: f32,
    /// Time-based decay
    pub time_decay: f32,
    /// Performance boost
    pub performance_boost: f32,
    /// Security penalty
    pub security_penalty: f32,
}

/// Time-based parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeParameters {
    /// Weight update interval
    pub update_interval: Duration,
    /// History window
    pub history_window: Duration,
    /// Decay half-life
    pub decay_half_life: Duration,
}

/// Network health thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthThresholds {
    /// Minimum node count
    pub min_nodes: usize,
    /// Minimum network throughput
    pub min_throughput: f64,
    /// Maximum latency
    pub max_latency: Duration,
    /// Minimum uptime
    pub min_uptime: f32,
}

/// Dynamic weight adjuster for social metrics
pub struct DynamicWeightAdjuster {
    /// Current parameters
    params: Arc<RwLock<WeightAdjustmentParams>>,
    /// Node scores
    node_scores: Arc<Mutex<HashMap<String, NodeScore>>>,
    /// Social graph analyzer
    social_analyzer: Arc<SocialGraph>,
    /// Weight history
    weight_history: Arc<Mutex<HashMap<String, Vec<(SystemTime, MetricWeights)>>>>,
    /// Network health metrics
    network_health: Arc<RwLock<NetworkHealth>>,
}

/// Network health metrics
#[derive(Debug, Clone)]
struct NetworkHealth {
    /// Active node count
    node_count: usize,
    /// Average throughput
    avg_throughput: f64,
    /// Average latency
    avg_latency: Duration,
    /// Network uptime
    uptime: f32,
    /// Last update time
    last_update: SystemTime,
}

impl DynamicWeightAdjuster {
    /// Create a new weight adjuster
    pub fn new(
        params: WeightAdjustmentParams,
        social_analyzer: Arc<SocialGraph>,
    ) -> Self {
        Self {
            params: Arc::new(RwLock::new(params)),
            node_scores: Arc::new(Mutex::new(HashMap::new())),
            social_analyzer,
            weight_history: Arc::new(Mutex::new(HashMap::new())),
            network_health: Arc::new(RwLock::new(NetworkHealth {
                node_count: 0,
                avg_throughput: 0.0,
                avg_latency: Duration::from_secs(0),
                uptime: 0.0,
                last_update: SystemTime::now(),
            })),
        }
    }

    /// Start the weight adjustment process
    pub async fn start(&self) -> Result<()> {
        let params = self.params.clone();
        let node_scores = self.node_scores.clone();
        let social_analyzer = self.social_analyzer.clone();
        let weight_history = self.weight_history.clone();
        let network_health = self.network_health.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                params.read().await.time_params.update_interval
            );

            loop {
                interval.tick().await;
                
                // Update network health
                if let Err(err) = Self::update_network_health(&network_health).await {
                    eprintln!("Error updating network health: {}", err);
                    continue;
                }
                
                // Get current scores
                let scores = node_scores.lock().await;
                
                // Process each node
                for (node_id, score) in scores.iter() {
                    // Get social metrics
                    match social_analyzer.get_social_score(node_id).await {
                        Ok(social_score) => {
                            // Calculate new weights
                            match Self::calculate_weights(
                                score,
                                social_score,
                                &(*params.read().await),
                                &(*network_health.read().await),
                            ).await {
                                Ok(new_weights) => {
                                    // Update history
                                    let mut history = weight_history.lock().await;
                                    let node_history = history.entry(node_id.clone())
                                        .or_insert_with(Vec::new);
                                    
                                    node_history.push((SystemTime::now(), new_weights));
                                    
                                    // Trim old history
                                    let cutoff = SystemTime::now() - params.read().await.time_params.history_window;
                                    node_history.retain(|(time, _)| *time > cutoff);
                                },
                                Err(err) => {
                                    eprintln!("Error calculating weights for node {}: {}", node_id, err);
                                }
                            }
                        },
                        Err(err) => {
                            eprintln!("Error getting social score for node {}: {}", node_id, err);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Update network health metrics
    async fn update_network_health(health: &Arc<RwLock<NetworkHealth>>) -> Result<()> {
        let mut current = health.write().await;
        
        // Update metrics (implement actual metric collection)
        current.node_count = 0; // TODO: Get actual count
        current.avg_throughput = 0.0; // TODO: Calculate
        current.avg_latency = Duration::from_secs(0); // TODO: Measure
        current.uptime = 0.0; // TODO: Calculate
        current.last_update = SystemTime::now();
        
        Ok(())
    }

    /// Calculate new weights for a node
    async fn calculate_weights(
        node_score: &NodeScore,
        social_score: f64,
        params: &WeightAdjustmentParams,
        health: &NetworkHealth,
    ) -> Result<MetricWeights> {
        // Start with base weights
        let mut weights = params.base_weights.clone();
        
        // Apply network health multiplier
        let health_factor = if health.node_count < params.health_thresholds.min_nodes
            || health.avg_throughput < params.health_thresholds.min_throughput
            || health.avg_latency > params.health_thresholds.max_latency
            || health.uptime < params.health_thresholds.min_uptime
        {
            params.adjustment_factors.network_health
        } else {
            1.0
        };
        
        // Apply performance boost for high-performing nodes
        let performance_boost = if node_score.overall_score > 0.8 {
            params.adjustment_factors.performance_boost
        } else {
            1.0
        };
        
        // Apply security penalty for low security scores
        let security_penalty = if node_score.ai_behavior_score < 0.5 {
            params.adjustment_factors.security_penalty
        } else {
            1.0
        };
        
        // Apply time-based decay
        let time_factor = (-SystemTime::now().duration_since(node_score.last_updated)?
            .as_secs_f64() / params.time_params.decay_half_life.as_secs_f64())
            .exp();
        
        // Adjust weights
        weights.device_health *= health_factor * performance_boost * time_factor as f32;
        weights.network_perf *= health_factor * performance_boost * time_factor as f32;
        weights.storage *= health_factor * time_factor as f32;
        weights.social *= social_score as f32 * time_factor as f32;
        weights.security *= security_penalty * time_factor as f32;
        
        // Normalize weights
        let sum = weights.device_health + weights.network_perf + weights.storage +
            weights.social + weights.security;
        
        weights.device_health /= sum;
        weights.network_perf /= sum;
        weights.storage /= sum;
        weights.social /= sum;
        weights.security /= sum;
        
        Ok(weights)
    }

    /// Get current weights for a node
    pub async fn get_weights(&self, node_id: &str) -> Result<MetricWeights> {
        let history = self.weight_history.lock().await;
        
        if let Some(node_history) = history.get(node_id) {
            if let Some((_, weights)) = node_history.last() {
                return Ok(weights.clone());
            }
        }
        
        // Return default weights if no history
        Ok(self.params.read().await.base_weights.clone())
    }

    /// Update adjustment parameters
    pub async fn update_params(&self, new_params: WeightAdjustmentParams) -> Result<()> {
        let mut params = self.params.write().await;
        *params = new_params;
        Ok(())
    }
} 