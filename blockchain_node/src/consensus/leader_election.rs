use crate::network::types::NodeId;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// LeaderElection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaderElectionStrategy {
    /// Round robin leader election
    RoundRobin,
    /// Random leader election
    Random,
    /// Weighted random leader election
    WeightedRandom,
    /// Performance-based leader election
    PerformanceBased,
    /// Stake-based leader election
    StakeBased,
    /// SVBFT-specific reputation-based leader election
    SocialVerified,
}

/// Configuration for leader election
#[derive(Debug, Clone)]
pub struct LeaderElectionConfig {
    /// Strategy to use for leader election
    pub strategy: LeaderElectionStrategy,
    /// Interval for leader rotation in milliseconds
    pub rotation_interval_ms: u64,
    /// Minimum performance score to be eligible as leader
    pub min_performance_score: f64,
    /// Minimum stake to be eligible as leader
    pub min_stake: u64,
    /// Include mobile nodes as potential leaders
    pub include_mobile_nodes: bool,
    /// Maximum consecutive terms for a leader
    pub max_consecutive_terms: usize,
}

impl Default for LeaderElectionConfig {
    fn default() -> Self {
        Self {
            strategy: LeaderElectionStrategy::SocialVerified,
            rotation_interval_ms: 10000, // 10 seconds
            min_performance_score: 0.7,
            min_stake: 1000,
            include_mobile_nodes: false,
            max_consecutive_terms: 2,
        }
    }
}

/// Performance metrics for leader election
#[derive(Debug, Clone)]
pub struct NodePerformance {
    /// Success rate of previous blocks
    pub success_rate: f64,
    /// Average latency in milliseconds
    pub average_latency: f64,
    /// Number of validators that voted for this node's blocks
    pub validator_support: usize,
    /// Uptime percentage
    pub uptime: f64,
    /// Last update timestamp
    pub last_update: Instant,
}

impl Default for NodePerformance {
    fn default() -> Self {
        Self {
            success_rate: 1.0,
            average_latency: 0.0,
            validator_support: 0,
            uptime: 1.0,
            last_update: Instant::now(),
        }
    }
}

/// Social verification metrics for SVBFT
#[derive(Debug, Clone)]
pub struct SocialMetrics {
    /// Trust score from other validators (0.0-1.0)
    pub trust_score: f64,
    /// Participation score in the network (0.0-1.0)
    pub participation_score: f64,
    /// Reputation score based on historical performance (0.0-1.0)
    pub reputation_score: f64,
    /// Social connectivity (number of social connections)
    pub social_connections: usize,
    /// Last update timestamp
    pub last_update: Instant,
}

impl Default for SocialMetrics {
    fn default() -> Self {
        Self {
            trust_score: 0.5,
            participation_score: 0.5,
            reputation_score: 0.5,
            social_connections: 0,
            last_update: Instant::now(),
        }
    }
}

/// Leader election manager
pub struct LeaderElectionManager {
    /// Configuration
    config: LeaderElectionConfig,
    /// Current leader
    current_leader: RwLock<Option<NodeId>>,
    /// Active validators
    validators: Arc<RwLock<HashSet<NodeId>>>,
    /// Performance metrics for each node
    performance: RwLock<HashMap<NodeId, NodePerformance>>,
    /// Social metrics for each node
    social_metrics: RwLock<HashMap<NodeId, SocialMetrics>>,
    /// Node stakes
    stakes: RwLock<HashMap<NodeId, u64>>,
    /// Last leader change timestamp
    last_leader_change: RwLock<Instant>,
    /// Current term counter
    term_counter: RwLock<HashMap<NodeId, usize>>,
}

impl LeaderElectionManager {
    /// Create a new leader election manager
    pub fn new(config: LeaderElectionConfig, validators: Arc<RwLock<HashSet<NodeId>>>) -> Self {
        Self {
            config,
            current_leader: RwLock::new(None),
            validators,
            performance: RwLock::new(HashMap::new()),
            social_metrics: RwLock::new(HashMap::new()),
            stakes: RwLock::new(HashMap::new()),
            last_leader_change: RwLock::new(Instant::now()),
            term_counter: RwLock::new(HashMap::new()),
        }
    }

    /// Start the leader election process
    pub async fn start(&self) -> Result<()> {
        // Perform initial leader election
        self.elect_leader().await?;

        // Start background task to rotate leaders based on interval
        let config = self.config.clone();
        let last_leader_change = self.last_leader_change.clone();
        let self_clone = Arc::new(self.clone());

        tokio::spawn(async move {
            let rotation_interval = Duration::from_millis(config.rotation_interval_ms);
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;

                let elapsed = {
                    let last_change = last_leader_change.read().await;
                    last_change.elapsed()
                };

                if elapsed >= rotation_interval {
                    if let Err(e) = self_clone.elect_leader().await {
                        warn!("Failed to elect new leader: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Elect a new leader based on the configured strategy
    pub async fn elect_leader(&self) -> Result<NodeId> {
        let validators = self.validators.read().await;
        if validators.is_empty() {
            return Err(anyhow!("No validators available for leader election"));
        }

        let leader = match self.config.strategy {
            LeaderElectionStrategy::RoundRobin => self.round_robin_election(&validators).await?,
            LeaderElectionStrategy::Random => self.random_election(&validators).await?,
            LeaderElectionStrategy::WeightedRandom => {
                self.weighted_random_election(&validators).await?
            }
            LeaderElectionStrategy::PerformanceBased => {
                self.performance_based_election(&validators).await?
            }
            LeaderElectionStrategy::StakeBased => self.stake_based_election(&validators).await?,
            LeaderElectionStrategy::SocialVerified => {
                self.social_verified_election(&validators).await?
            }
        };

        // Update current leader and timestamp
        *self.current_leader.write().await = Some(leader.clone());
        *self.last_leader_change.write().await = Instant::now();

        // Update consecutive terms counter
        let mut term_counter = self.term_counter.write().await;
        for (node, count) in term_counter.iter_mut() {
            if node == &leader {
                *count += 1;
            } else {
                *count = 0;
            }
        }

        // Ensure the leader has an entry in the term counter
        term_counter.entry(leader.clone()).or_insert(1);

        info!("Elected new leader: {}", leader);
        Ok(leader)
    }

    /// Get the current leader
    pub async fn get_current_leader(&self) -> Option<NodeId> {
        self.current_leader.read().await.clone()
    }

    /// Update node performance metrics
    pub async fn update_performance(&self, node_id: NodeId, performance: NodePerformance) {
        let mut performances = self.performance.write().await;
        performances.insert(node_id, performance);
    }

    /// Update social metrics for a node
    pub async fn update_social_metrics(&self, node_id: NodeId, metrics: SocialMetrics) {
        let mut social_metrics = self.social_metrics.write().await;
        social_metrics.insert(node_id, metrics);
    }

    /// Update stake for a node
    pub async fn update_stake(&self, node_id: NodeId, stake: u64) {
        let mut stakes = self.stakes.write().await;
        stakes.insert(node_id, stake);
    }

    // Implementation of different leader election strategies

    /// Round-robin leader election
    async fn round_robin_election(&self, validators: &HashSet<NodeId>) -> Result<NodeId> {
        let current = self.current_leader.read().await;
        let mut validators_vec: Vec<_> = validators.iter().cloned().collect();
        validators_vec.sort(); // Sort for deterministic order

        if let Some(current_leader) = &*current {
            let idx = validators_vec.iter().position(|id| id == current_leader);
            if let Some(pos) = idx {
                let next = (pos + 1) % validators_vec.len();
                return Ok(validators_vec[next].clone());
            }
        }

        // No current leader or not found, start from beginning
        Ok(validators_vec[0].clone())
    }

    /// Random leader election
    async fn random_election(&self, validators: &HashSet<NodeId>) -> Result<NodeId> {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let validators_vec: Vec<_> = validators.iter().cloned().collect();

        // Create deterministic RNG with a time-based seed
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        let idx = rng.gen_range(0..validators_vec.len());
        Ok(validators_vec[idx].clone())
    }

    /// Weighted random leader election
    async fn weighted_random_election(&self, validators: &HashSet<NodeId>) -> Result<NodeId> {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let performances = self.performance.read().await;
        let social_metrics = self.social_metrics.read().await;
        let term_counter = self.term_counter.read().await;

        // Calculate weights for each validator
        let mut weights = Vec::new();
        let mut validators_vec = Vec::new();

        for validator in validators {
            // Skip validators that have reached max consecutive terms
            if let Some(terms) = term_counter.get(validator) {
                if *terms >= self.config.max_consecutive_terms {
                    continue;
                }
            }

            let performance = performances
                .get(validator)
                .unwrap_or(&NodePerformance::default());
            let social = social_metrics
                .get(validator)
                .unwrap_or(&SocialMetrics::default());

            // Calculate combined weight
            let weight = performance.success_rate * social.reputation_score * 100.0;

            if weight > 0.0 {
                weights.push(weight as u32);
                validators_vec.push(validator.clone());
            }
        }

        if validators_vec.is_empty() {
            // Fall back to random election if no weighted candidates
            return self.random_election(validators).await;
        }

        // Create deterministic RNG with a time-based seed
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        // Sum weights and get a random point
        let total_weight: u32 = weights.iter().sum();
        let mut point = rng.gen_range(0..total_weight);

        // Find the validator corresponding to the random point
        for (i, weight) in weights.iter().enumerate() {
            if point < *weight {
                return Ok(validators_vec[i].clone());
            }
            point -= *weight;
        }

        // Should never reach here, but return the last one as fallback
        Ok(validators_vec.last().unwrap().clone())
    }

    /// Performance-based leader election
    async fn performance_based_election(&self, validators: &HashSet<NodeId>) -> Result<NodeId> {
        let performances = self.performance.read().await;
        let term_counter = self.term_counter.read().await;

        let mut best_node = None;
        let mut best_score = 0.0;

        for validator in validators {
            // Skip validators that have reached max consecutive terms
            if let Some(terms) = term_counter.get(validator) {
                if *terms >= self.config.max_consecutive_terms {
                    continue;
                }
            }

            if let Some(perf) = performances.get(validator) {
                // Skip nodes below minimum score
                if perf.success_rate < self.config.min_performance_score {
                    continue;
                }

                // Calculate combined score
                let score = perf.success_rate * (1.0 - perf.average_latency / 1000.0);

                if score > best_score {
                    best_score = score;
                    best_node = Some(validator.clone());
                }
            }
        }

        if let Some(node) = best_node {
            Ok(node)
        } else {
            // Fall back to random election if no suitable performance-based candidate
            self.random_election(validators).await
        }
    }

    /// Stake-based leader election
    async fn stake_based_election(&self, validators: &HashSet<NodeId>) -> Result<NodeId> {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let stakes = self.stakes.read().await;
        let term_counter = self.term_counter.read().await;

        let mut weights = Vec::new();
        let mut validators_vec = Vec::new();

        for validator in validators {
            // Skip validators that have reached max consecutive terms
            if let Some(terms) = term_counter.get(validator) {
                if *terms >= self.config.max_consecutive_terms {
                    continue;
                }
            }

            let stake = stakes.get(validator).unwrap_or(&0);

            // Skip nodes below minimum stake
            if *stake < self.config.min_stake {
                continue;
            }

            weights.push(*stake);
            validators_vec.push(validator.clone());
        }

        if validators_vec.is_empty() {
            // Fall back to random election if no staked candidates
            return self.random_election(validators).await;
        }

        // Create deterministic RNG with a time-based seed
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        // Sum weights and get a random point
        let total_weight: u64 = weights.iter().sum();
        let mut point = rng.gen_range(0..total_weight);

        // Find the validator corresponding to the random point
        for (i, weight) in weights.iter().enumerate() {
            if point < *weight {
                return Ok(validators_vec[i].clone());
            }
            point -= *weight;
        }

        // Should never reach here, but return the last one as fallback
        Ok(validators_vec.last().unwrap().clone())
    }

    /// Social verified leader election for SVBFT
    async fn social_verified_election(&self, validators: &HashSet<NodeId>) -> Result<NodeId> {
        let social_metrics = self.social_metrics.read().await;
        let performances = self.performance.read().await;
        let term_counter = self.term_counter.read().await;

        let mut best_node = None;
        let mut best_score = 0.0;

        for validator in validators {
            // Skip validators that have reached max consecutive terms
            if let Some(terms) = term_counter.get(validator) {
                if *terms >= self.config.max_consecutive_terms {
                    continue;
                }
            }

            let social = social_metrics
                .get(validator)
                .unwrap_or(&SocialMetrics::default());
            let perf = performances
                .get(validator)
                .unwrap_or(&NodePerformance::default());

            // Skip nodes below minimum performance score
            if perf.success_rate < self.config.min_performance_score {
                continue;
            }

            // Calculate combined social score
            let score = social.trust_score * 0.4
                + social.reputation_score * 0.3
                + social.participation_score * 0.2
                + (social.social_connections as f64 / 100.0).min(0.1);

            // Apply performance modifier
            let final_score = score * perf.success_rate;

            if final_score > best_score {
                best_score = final_score;
                best_node = Some(validator.clone());
            }
        }

        if let Some(node) = best_node {
            Ok(node)
        } else {
            // Fall back to random election if no suitable social-verified candidate
            self.random_election(validators).await
        }
    }
}

impl Clone for LeaderElectionManager {
    fn clone(&self) -> Self {
        // This is a partial clone for use in async tasks
        // The RwLocks will be new but the references within will be the same
        Self {
            config: self.config.clone(),
            current_leader: RwLock::new(None),
            validators: self.validators.clone(),
            performance: RwLock::new(HashMap::new()),
            social_metrics: RwLock::new(HashMap::new()),
            stakes: RwLock::new(HashMap::new()),
            last_leader_change: RwLock::new(Instant::now()),
            term_counter: RwLock::new(HashMap::new()),
        }
    }
}
