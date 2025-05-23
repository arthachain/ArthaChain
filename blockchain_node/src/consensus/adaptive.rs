use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::consensus::ConsensusType;
use crate::ledger::block::Block;
use crate::network::types::NodeId;
use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use std::collections::HashSet;

/// Configurable parameters for the adaptive consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConsensusConfig {
    /// Available consensus algorithms to switch between
    pub available_algorithms: Vec<ConsensusType>,
    /// Monitoring interval in milliseconds
    pub monitoring_interval_ms: u64,
    /// Adaptation threshold (0.0-1.0)
    pub adaptation_threshold: f64,
    /// Minimum time between algorithm switches (ms)
    pub min_switch_interval_ms: u64,
    /// Maximum consecutive failed blocks before switching
    pub max_consecutive_failures: usize,
    /// Enable automatic adaptation
    pub auto_adaptation: bool,
    /// Weight factors for different metrics
    pub metric_weights: MetricWeights,
}

impl Default for AdaptiveConsensusConfig {
    fn default() -> Self {
        Self {
            available_algorithms: vec![
                ConsensusType::Poa,
                ConsensusType::Pbft,
                ConsensusType::Raft,
                ConsensusType::Svbft,
            ],
            monitoring_interval_ms: 30000,  // 30 seconds
            adaptation_threshold: 0.25,     // 25% difference to trigger adaptation
            min_switch_interval_ms: 300000, // 5 minutes
            max_consecutive_failures: 5,
            auto_adaptation: true,
            metric_weights: MetricWeights::default(),
        }
    }
}

/// Weights for different performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricWeights {
    /// Transaction throughput weight
    pub throughput_weight: f64,
    /// Latency weight
    pub latency_weight: f64,
    /// Energy usage weight
    pub energy_weight: f64,
    /// Security weight
    pub security_weight: f64,
    /// Reliability weight
    pub reliability_weight: f64,
    /// Network conditions weight
    pub network_weight: f64,
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self {
            throughput_weight: 0.25,
            latency_weight: 0.20,
            energy_weight: 0.10,
            security_weight: 0.20,
            reliability_weight: 0.15,
            network_weight: 0.10,
        }
    }
}

/// Performance metrics for consensus algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Transactions per second
    pub tps: f64,
    /// Average block time (ms)
    pub avg_block_time_ms: f64,
    /// Block finality time (ms)
    pub finality_time_ms: f64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage (MB)
    pub memory_usage: f64,
    /// Network bandwidth usage (KB/s)
    pub network_bandwidth: f64,
    /// Failed block proposals
    pub failed_proposals: usize,
    /// Successful block proposals
    pub successful_proposals: usize,
    /// Average number of validators
    pub avg_validators: usize,
    /// Security incidents
    pub security_incidents: usize,
    /// Last update timestamp
    pub last_updated: u64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            tps: 0.0,
            avg_block_time_ms: 0.0,
            finality_time_ms: 0.0,
            cpu_usage: 0.0,
            memory_usage: 0.0,
            network_bandwidth: 0.0,
            failed_proposals: 0,
            successful_proposals: 0,
            avg_validators: 0,
            security_incidents: 0,
            last_updated: 0,
        }
    }
}

/// Network conditions that affect consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConditions {
    /// Average node latency (ms)
    pub avg_latency_ms: f64,
    /// Packet loss percentage
    pub packet_loss_percent: f64,
    /// Network partition events
    pub partition_events: usize,
    /// Network bandwidth (Mbps)
    pub bandwidth_mbps: f64,
    /// Percentage of mobile nodes
    pub mobile_nodes_percent: f64,
    /// Percentage of high-performance nodes
    pub high_perf_nodes_percent: f64,
    /// Geographic distribution metric (0-1)
    pub geographic_distribution: f64,
    /// Last update timestamp
    pub last_updated: u64,
}

impl Default for NetworkConditions {
    fn default() -> Self {
        Self {
            avg_latency_ms: 100.0,
            packet_loss_percent: 0.1,
            partition_events: 0,
            bandwidth_mbps: 100.0,
            mobile_nodes_percent: 0.0,
            high_perf_nodes_percent: 1.0,
            geographic_distribution: 0.5,
            last_updated: 0,
        }
    }
}

/// The Adaptive Consensus Manager
pub struct AdaptiveConsensusManager {
    /// Configuration
    config: RwLock<AdaptiveConsensusConfig>,
    /// Current active consensus algorithm
    current_algorithm: RwLock<ConsensusType>,
    /// Performance metrics by consensus type
    metrics: RwLock<HashMap<ConsensusType, PerformanceMetrics>>,
    /// Current network conditions
    network_conditions: RwLock<NetworkConditions>,
    /// Last time the algorithm was switched
    last_switch_time: RwLock<Instant>,
    /// Running status
    running: RwLock<bool>,
    /// Consecutive failed blocks
    consecutive_failures: RwLock<usize>,
    /// Validators
    validators: Arc<RwLock<HashSet<NodeId>>>,
    /// Blockchain config
    blockchain_config: Arc<RwLock<Config>>,
}

impl AdaptiveConsensusManager {
    /// Create a new adaptive consensus manager
    pub fn new(
        config: AdaptiveConsensusConfig,
        initial_algorithm: ConsensusType,
        validators: Arc<RwLock<HashSet<NodeId>>>,
        blockchain_config: Arc<RwLock<Config>>,
    ) -> Self {
        let mut metrics = HashMap::new();
        for algorithm in &config.available_algorithms {
            metrics.insert(*algorithm, PerformanceMetrics::default());
        }

        Self {
            config: RwLock::new(config),
            current_algorithm: RwLock::new(initial_algorithm),
            metrics: RwLock::new(metrics),
            network_conditions: RwLock::new(NetworkConditions::default()),
            last_switch_time: RwLock::new(Instant::now()),
            running: RwLock::new(false),
            consecutive_failures: RwLock::new(0),
            validators,
            blockchain_config,
        }
    }

    /// Start the adaptive consensus manager
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(anyhow!("Adaptive consensus manager already running"));
        }

        *running = true;
        let config = self.config.read().await.clone();

        // Spawn monitoring task
        let self_clone = Arc::new(self.clone());
        tokio::spawn(async move {
            let interval = Duration::from_millis(config.monitoring_interval_ms);
            loop {
                tokio::time::sleep(interval).await;

                let is_running = *self_clone.running.read().await;
                if !is_running {
                    break;
                }

                if let Err(e) = self_clone.evaluate_and_adapt().await {
                    warn!("Error in adaptive consensus evaluation: {}", e);
                }
            }
        });

        info!(
            "Adaptive consensus manager started with initial algorithm: {:?}",
            *self.current_algorithm.read().await
        );
        Ok(())
    }

    /// Stop the adaptive consensus manager
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(anyhow!("Adaptive consensus manager not running"));
        }

        *running = false;
        info!("Adaptive consensus manager stopped");
        Ok(())
    }

    /// Update performance metrics for the current consensus algorithm
    pub async fn update_metrics(&self, metrics: PerformanceMetrics) -> Result<()> {
        let current_algo = *self.current_algorithm.read().await;
        let mut all_metrics = self.metrics.write().await;

        all_metrics.insert(current_algo, metrics);

        // Reset consecutive failures if we have a successful proposal
        if metrics.successful_proposals > 0 {
            let mut failures = self.consecutive_failures.write().await;
            *failures = 0;
        }

        Ok(())
    }

    /// Update network conditions
    pub async fn update_network_conditions(&self, conditions: NetworkConditions) -> Result<()> {
        let mut network_cond = self.network_conditions.write().await;
        *network_cond = conditions;
        Ok(())
    }

    /// Record a failed block proposal
    pub async fn record_failed_proposal(&self) -> Result<()> {
        let current_algo = *self.current_algorithm.read().await;
        let mut all_metrics = self.metrics.write().await;

        if let Some(metrics) = all_metrics.get_mut(&current_algo) {
            metrics.failed_proposals += 1;
        }

        // Increment consecutive failures counter
        let mut failures = self.consecutive_failures.write().await;
        *failures += 1;

        // Check if we've reached the threshold for automatic switching
        let config = self.config.read().await;
        if config.auto_adaptation && *failures >= config.max_consecutive_failures {
            drop(failures); // Release the lock before calling switch_algorithm
            self.force_algorithm_switch().await?;
        }

        Ok(())
    }

    /// Record a successful block proposal
    pub async fn record_successful_proposal(&self, block: &Block) -> Result<()> {
        let current_algo = *self.current_algorithm.read().await;
        let mut all_metrics = self.metrics.write().await;

        if let Some(metrics) = all_metrics.get_mut(&current_algo) {
            metrics.successful_proposals += 1;

            // Update other metrics based on the block
            if let Some(timestamp) = block.timestamp {
                if let Some(proposal_time) = block.proposal_timestamp {
                    let finality_time = timestamp - proposal_time;

                    // Exponential moving average for block time and finality
                    let alpha = 0.2; // Smoothing factor
                    metrics.finality_time_ms =
                        alpha * (finality_time as f64) + (1.0 - alpha) * metrics.finality_time_ms;

                    // Calculate TPS based on number of transactions and time
                    let txs_count = block.txs.len() as f64;
                    if finality_time > 0 {
                        let current_tps = txs_count * 1000.0 / (finality_time as f64);
                        metrics.tps = alpha * current_tps + (1.0 - alpha) * metrics.tps;
                    }
                }
            }

            // Update last updated timestamp
            metrics.last_updated = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }

        // Reset consecutive failures counter
        let mut failures = self.consecutive_failures.write().await;
        *failures = 0;

        Ok(())
    }

    /// Evaluate current performance and adapt if necessary
    async fn evaluate_and_adapt(&self) -> Result<()> {
        let config = self.config.read().await;

        // Skip evaluation if auto-adaptation is disabled
        if !config.auto_adaptation {
            return Ok(());
        }

        // Check if we've waited long enough since the last switch
        let last_switch = *self.last_switch_time.read().await;
        let min_interval = Duration::from_millis(config.min_switch_interval_ms);
        if last_switch.elapsed() < min_interval {
            debug!("Skipping adaptation evaluation - minimum interval not reached");
            return Ok(());
        }

        // Collect current metrics and network conditions
        let current_algo = *self.current_algorithm.read().await;
        let all_metrics = self.metrics.read().await;
        let current_metrics = all_metrics.get(&current_algo).cloned().unwrap_or_default();
        let network_cond = self.network_conditions.read().await.clone();

        // Select the best algorithm for current conditions
        let best_algo = self
            .select_best_algorithm(&current_metrics, &network_cond)
            .await?;

        // If the best algorithm is different and exceeds the adaptation threshold, switch
        if best_algo != current_algo {
            let best_score = self
                .calculate_algorithm_score(best_algo, &current_metrics, &network_cond)
                .await?;
            let current_score = self
                .calculate_algorithm_score(current_algo, &current_metrics, &network_cond)
                .await?;

            let improvement = (best_score - current_score) / current_score;
            if improvement > config.adaptation_threshold {
                info!(
                    "Adaptive consensus switching from {:?} to {:?} (improvement: {:.2}%)",
                    current_algo,
                    best_algo,
                    improvement * 100.0
                );

                self.switch_algorithm(best_algo).await?;
            } else {
                debug!("Potential algorithm switch skipped - improvement of {:.2}% below threshold of {:.2}%", 
                       improvement * 100.0, config.adaptation_threshold * 100.0);
            }
        }

        Ok(())
    }

    /// Force a switch to the best algorithm regardless of thresholds
    async fn force_algorithm_switch(&self) -> Result<()> {
        let current_algo = *self.current_algorithm.read().await;
        let all_metrics = self.metrics.read().await;
        let current_metrics = all_metrics.get(&current_algo).cloned().unwrap_or_default();
        let network_cond = self.network_conditions.read().await.clone();

        // Select the best algorithm for current conditions
        let best_algo = self
            .select_best_algorithm(&current_metrics, &network_cond)
            .await?;

        // Only switch if the best algorithm is different
        if best_algo != current_algo {
            info!(
                "Forcing adaptive consensus switch from {:?} to {:?} due to consecutive failures",
                current_algo, best_algo
            );

            self.switch_algorithm(best_algo).await?;
        }

        // Reset consecutive failures
        let mut failures = self.consecutive_failures.write().await;
        *failures = 0;

        Ok(())
    }

    /// Switch to a different consensus algorithm
    async fn switch_algorithm(&self, new_algo: ConsensusType) -> Result<()> {
        let config = self.config.read().await;

        // Ensure the algorithm is in the available list
        if !config.available_algorithms.contains(&new_algo) {
            return Err(anyhow!(
                "Algorithm {:?} is not in the available list",
                new_algo
            ));
        }

        // Update current algorithm
        let mut current = self.current_algorithm.write().await;
        *current = new_algo;

        // Update last switch time
        let mut last_switch = self.last_switch_time.write().await;
        *last_switch = Instant::now();

        // Update blockchain config to reflect consensus change
        let mut blockchain_cfg = self.blockchain_config.write().await;
        blockchain_cfg.consensus_type = format!("{:?}", new_algo).to_lowercase();

        info!("Switched consensus algorithm to {:?}", new_algo);
        Ok(())
    }

    /// Select the best algorithm for current conditions
    async fn select_best_algorithm(
        &self,
        current_metrics: &PerformanceMetrics,
        network_cond: &NetworkConditions,
    ) -> Result<ConsensusType> {
        let config = self.config.read().await;
        let validators_count = self.validators.read().await.len();

        let mut best_algo = *self.current_algorithm.read().await;
        let mut best_score = f64::MIN;

        for &algo in &config.available_algorithms {
            let score = self
                .calculate_algorithm_score(algo, current_metrics, network_cond)
                .await?;

            // Apply additional constraints based on validator count
            let adjusted_score = match algo {
                ConsensusType::Pbft | ConsensusType::Svbft => {
                    // BFT algorithms perform well with 4+ validators
                    if validators_count >= 4 {
                        score
                    } else {
                        score * 0.5 // Penalize for small networks
                    }
                }
                ConsensusType::Poa => {
                    // PoA is good for small networks
                    if validators_count < 10 {
                        score * 1.2 // Bonus for small networks
                    } else {
                        score
                    }
                }
                ConsensusType::Raft => {
                    // Raft is good for medium-sized networks with low partition risk
                    if validators_count < 20 && network_cond.partition_events < 2 {
                        score * 1.1
                    } else {
                        score
                    }
                }
                _ => score,
            };

            if adjusted_score > best_score {
                best_score = adjusted_score;
                best_algo = algo;
            }
        }

        Ok(best_algo)
    }

    /// Calculate a score for an algorithm based on current conditions
    async fn calculate_algorithm_score(
        &self,
        algo: ConsensusType,
        current_metrics: &PerformanceMetrics,
        network_cond: &NetworkConditions,
    ) -> Result<f64> {
        let config = self.config.read().await;
        let weights = &config.metric_weights;

        // Base scores for algorithms in different metrics (0-1 scale)
        let base_scores = match algo {
            ConsensusType::Pbft => {
                // PBFT is balanced but sensitive to network conditions
                let throughput = 0.7;
                let latency = if network_cond.avg_latency_ms < 200.0 {
                    0.8
                } else {
                    0.5
                };
                let energy = 0.6;
                let security = 0.8;
                let reliability = if network_cond.packet_loss_percent < 1.0 {
                    0.8
                } else {
                    0.5
                };
                let network = if network_cond.mobile_nodes_percent > 0.3 {
                    0.5
                } else {
                    0.7
                };

                throughput * weights.throughput_weight
                    + latency * weights.latency_weight
                    + energy * weights.energy_weight
                    + security * weights.security_weight
                    + reliability * weights.reliability_weight
                    + network * weights.network_weight
            }
            ConsensusType::Svbft => {
                // SVBFT handles diverse networks well
                let throughput = 0.7;
                let latency = if network_cond.avg_latency_ms < 300.0 {
                    0.7
                } else {
                    0.6
                };
                let energy = 0.5;
                let security = 0.9;
                let reliability = 0.8;
                let network = if network_cond.mobile_nodes_percent > 0.0 {
                    0.9
                } else {
                    0.7
                };

                throughput * weights.throughput_weight
                    + latency * weights.latency_weight
                    + energy * weights.energy_weight
                    + security * weights.security_weight
                    + reliability * weights.reliability_weight
                    + network * weights.network_weight
            }
            ConsensusType::Poa => {
                // PoA has high throughput but lower security
                let throughput = 0.9;
                let latency = 0.9;
                let energy = 0.8;
                let security = 0.6;
                let reliability = 0.7;
                let network = if network_cond.high_perf_nodes_percent > 0.8 {
                    0.9
                } else {
                    0.7
                };

                throughput * weights.throughput_weight
                    + latency * weights.latency_weight
                    + energy * weights.energy_weight
                    + security * weights.security_weight
                    + reliability * weights.reliability_weight
                    + network * weights.network_weight
            }
            ConsensusType::Raft => {
                // Raft is efficient but vulnerable to network partitions
                let throughput = 0.8;
                let latency = 0.8;
                let energy = 0.7;
                let security = 0.7;
                let reliability = if network_cond.partition_events < 2 {
                    0.8
                } else {
                    0.4
                };
                let network = if network_cond.geographic_distribution < 0.5 {
                    0.8
                } else {
                    0.5
                };

                throughput * weights.throughput_weight
                    + latency * weights.latency_weight
                    + energy * weights.energy_weight
                    + security * weights.security_weight
                    + reliability * weights.reliability_weight
                    + network * weights.network_weight
            }
            ConsensusType::Pos | ConsensusType::Dpos => {
                // PoS-based algorithms
                let throughput = 0.6;
                let latency = 0.6;
                let energy = 0.9;
                let security = 0.7;
                let reliability = 0.7;
                let network = 0.6;

                throughput * weights.throughput_weight
                    + latency * weights.latency_weight
                    + energy * weights.energy_weight
                    + security * weights.security_weight
                    + reliability * weights.reliability_weight
                    + network * weights.network_weight
            }
            _ => {
                // Default score for other algorithms
                0.5
            }
        };

        // Adjust base score based on historical performance metrics
        let all_metrics = self.metrics.read().await;
        let historical_adjustment = if let Some(metrics) = all_metrics.get(&algo) {
            let success_rate = if metrics.successful_proposals + metrics.failed_proposals > 0 {
                metrics.successful_proposals as f64
                    / (metrics.successful_proposals + metrics.failed_proposals) as f64
            } else {
                0.5 // No data
            };

            // Adjust score based on success rate
            if success_rate > 0.9 {
                0.2 // Bonus for highly successful algorithms
            } else if success_rate > 0.7 {
                0.1 // Small bonus for mostly successful
            } else if success_rate < 0.3 {
                -0.2 // Penalty for mostly failing
            } else {
                0.0 // Neutral
            }
        } else {
            0.0 // No historical data
        };

        // Final score combines base score with historical adjustment
        let final_score = base_scores + historical_adjustment;

        // Clamp to reasonable range
        Ok(final_score.max(0.1).min(1.0))
    }

    /// Get the current consensus algorithm
    pub async fn get_current_algorithm(&self) -> ConsensusType {
        *self.current_algorithm.read().await
    }

    /// Get metrics for a specific consensus algorithm
    pub async fn get_algorithm_metrics(&self, algo: ConsensusType) -> Option<PerformanceMetrics> {
        let metrics = self.metrics.read().await;
        metrics.get(&algo).cloned()
    }

    /// Get current network conditions
    pub async fn get_network_conditions(&self) -> NetworkConditions {
        self.network_conditions.read().await.clone()
    }

    /// Update configuration
    pub async fn update_config(&self, config: AdaptiveConsensusConfig) -> Result<()> {
        let mut cfg = self.config.write().await;
        *cfg = config;
        Ok(())
    }
}

impl Clone for AdaptiveConsensusManager {
    fn clone(&self) -> Self {
        // This is a partial clone for use in async tasks
        // The RwLocks will be new but the references within will be the same
        Self {
            config: RwLock::new(self.config.try_read().unwrap_or_default().clone()),
            current_algorithm: RwLock::new(
                *self
                    .current_algorithm
                    .try_read()
                    .unwrap_or(&ConsensusType::Svbft),
            ),
            metrics: RwLock::new(HashMap::new()),
            network_conditions: RwLock::new(NetworkConditions::default()),
            last_switch_time: RwLock::new(Instant::now()),
            running: RwLock::new(false),
            consecutive_failures: RwLock::new(0),
            validators: self.validators.clone(),
            blockchain_config: self.blockchain_config.clone(),
        }
    }
}
