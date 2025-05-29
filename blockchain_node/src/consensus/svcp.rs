use crate::ai_engine::security::NodeScore;
use crate::config::Config;
use crate::consensus::parallel_processor::ParallelProcessor;
use crate::consensus::parallel_processor::ParallelProcessorConfig;
use crate::ledger::block::{Block, BlockExt};
use crate::ledger::BlockchainState;
use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use rand::{seq::SliceRandom, thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::broadcast;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;

/// Configuration for SVCP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVCPConfig {
    /// Minimum score required to participate in consensus
    pub min_score_threshold: f32,
    /// Maximum number of proposer candidates
    pub max_proposer_candidates: usize,
    /// Minimum number of proposer candidates
    pub min_proposer_candidates: usize,
    /// Target block time in seconds
    pub target_block_time: u64,
    /// Difficulty adjustment window (in blocks)
    pub difficulty_adjustment_window: u64,
    /// Initial POW difficulty
    pub initial_pow_difficulty: u64,
    /// Weight for device score in candidate selection
    pub device_weight: f32,
    /// Weight for network score in candidate selection
    pub network_weight: f32,
    /// Weight for storage score in candidate selection
    pub storage_weight: f32,
    /// Weight for engagement score in candidate selection
    pub engagement_weight: f32,
    /// Weight for AI behavior score in candidate selection
    pub ai_behavior_weight: f32,
    /// Base batch size for transactions per block
    pub base_batch_size: usize,
}

impl Default for SVCPConfig {
    fn default() -> Self {
        Self {
            min_score_threshold: 0.6,
            max_proposer_candidates: 100,
            min_proposer_candidates: 10,
            target_block_time: 5,
            difficulty_adjustment_window: 10,
            initial_pow_difficulty: 4,
            device_weight: 0.2,
            network_weight: 0.3,
            storage_weight: 0.1,
            engagement_weight: 0.2,
            ai_behavior_weight: 0.2,
            base_batch_size: 500,
        }
    }
}

/// Result of mining attempt
#[derive(Debug, Clone)]
pub enum MiningResult {
    /// Block was successfully mined
    Success(Block),
    /// Mining was interrupted
    Interrupted,
    /// Mining failed due to error
    Error(String),
}

/// Candidate proposer entry for BinaryHeap ordering
#[derive(Debug, Clone)]
struct ProposerCandidate {
    /// Node ID
    node_id: String,
    /// Combined score
    score: f32,
    /// Last block proposed timestamp
    last_proposed: SystemTime,
}

impl PartialEq for ProposerCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl Eq for ProposerCandidate {}

impl PartialOrd for ProposerCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProposerCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // First by last_proposed timestamp (older is better)
        // Since binary heap pops max elements first, we need to reverse
        // the comparison to make older timestamps appear first
        match other.last_proposed.cmp(&self.last_proposed) {
            Ordering::Equal => {
                // If timestamps are equal, use score (higher is better)
                match self.score.partial_cmp(&other.score) {
                    Some(ordering) => ordering,
                    None => self.node_id.cmp(&other.node_id), // For stability
                }
            }
            ordering => ordering,
        }
    }
}

/// Calculate target based on difficulty
#[allow(dead_code)]
fn calculate_target(difficulty: u64) -> [u8; 32] {
    let mut target = [0xFF; 32]; // Start with the easiest target (all 1s)

    // Higher difficulty means a smaller target value
    // We'll implement a simple algorithm to decrease the target as difficulty increases
    let difficulty = difficulty.max(1); // Make sure difficulty is at least 1

    // Calculate how many leading zeros we need based on difficulty
    // For each power of 2 in difficulty, we add a leading zero
    let leading_zeros = 32u32.saturating_sub(difficulty.leading_zeros());

    // Adjust target by setting leading bytes to zero
    for i in 0..leading_zeros as usize {
        if i < target.len() {
            target[i] = 0;
        }
    }

    // For fine-tuning, adjust the first non-zero byte
    if (leading_zeros as usize) < target.len() {
        let remainder = difficulty % (1 << leading_zeros);
        if remainder > 0 {
            let divisor = 256 / (1 << (8 - leading_zeros % 8));
            target[leading_zeros as usize] = 0xFF / divisor as u8;
        }
    }

    target
}

/// SVCPMiner implements the Social Verified Consensus Protocol mining
pub struct SVCPMiner {
    /// Configuration (saved for potential future use)
    #[allow(dead_code)]
    config: Config,
    /// SVCP specific configuration
    svcp_config: SVCPConfig,
    /// Current POW difficulty (bits)
    current_difficulty: u64,
    /// Blockchain state
    state: Arc<RwLock<BlockchainState>>,
    /// Node scores by node_id
    node_scores: Arc<Mutex<HashMap<String, NodeScore>>>,
    /// Selected proposers for current round
    current_proposers: Arc<Mutex<Vec<String>>>,
    /// Block times for difficulty adjustment
    block_times: Arc<Mutex<Vec<(SystemTime, Duration)>>>,
    /// Channel for sending mined blocks
    block_sender: mpsc::Sender<Block>,
    /// Channel for receiving shutdown signal
    shutdown_receiver: broadcast::Receiver<()>,
    /// Last update of proposer set
    last_proposer_update: Arc<Mutex<Instant>>,
    /// Running flag
    running: Arc<Mutex<bool>>,
    /// This node's ID
    node_id: String,
    /// Parallel processor for scaling with miner count
    parallel_processor: Option<ParallelProcessor>,

    /// Validator count tracker for scaling
    validator_count: Arc<Mutex<usize>>,

    /// TPS scaling enabled flag
    tps_scaling_enabled: bool,

    /// TPS multiplier per miner (1.5-5.0)
    tps_multiplier: f32,

    /// Flag to enable SIMD verification
    enable_simd_verification: bool,

    /// Dynamic puzzle difficulty adjuster
    dynamic_adjuster: Option<DynamicPuzzleAdjuster>,

    /// Block time monitor for performance tracking
    block_time_monitor: Option<BlockTimeMonitor>,
}

impl SVCPMiner {
    /// Create a new SVCP miner instance
    pub fn new(
        config: Config,
        state: Arc<RwLock<BlockchainState>>,
        block_sender: mpsc::Sender<Block>,
        shutdown_receiver: broadcast::Receiver<()>,
        node_scores: Arc<Mutex<HashMap<String, NodeScore>>>,
        svcp_config: Option<SVCPConfig>,
    ) -> Result<Self> {
        let node_id = config
            .node_id
            .clone()
            .ok_or_else(|| anyhow!("Node ID not set in config"))?;
        let config_clone = svcp_config.clone();

        // Create default TPS multiplier (can be configured via env var or config)
        let tps_multiplier = std::env::var("TPS_MULTIPLIER")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(2.0);

        // Check if TPS scaling is enabled
        let tps_scaling_enabled = std::env::var("ENABLE_TPS_SCALING")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(true); // Enabled by default

        // Create parallel processor if scaling is enabled
        let parallel_processor = if tps_scaling_enabled {
            Some(ParallelProcessor::new(
                state.clone(),
                block_sender.clone(),
                Some(tps_multiplier),
            ))
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            svcp_config: svcp_config.unwrap_or_default(),
            current_difficulty: config_clone.unwrap_or_default().initial_pow_difficulty,
            state,
            node_scores,
            current_proposers: Arc::new(Mutex::new(Vec::new())),
            block_times: Arc::new(Mutex::new(Vec::new())),
            block_sender,
            shutdown_receiver,
            last_proposer_update: Arc::new(Mutex::new(Instant::now())),
            running: Arc::new(Mutex::new(false)),
            node_id,
            parallel_processor,
            validator_count: Arc::new(Mutex::new(1)),
            tps_scaling_enabled,
            tps_multiplier,
            enable_simd_verification: false,
            dynamic_adjuster: None,
            block_time_monitor: None,
        })
    }

    /// Start the SVCP miner
    pub async fn start(&mut self) -> Result<JoinHandle<()>> {
        // Set running flag
        {
            let mut running = self.running.lock().await;
            *running = true;
        }

        // Start parallel processor if enabled
        let parallel_handle = if self.tps_scaling_enabled {
            if let Some(processor) = &self.parallel_processor {
                Some(processor.start().await?)
            } else {
                None
            }
        } else {
            None
        };

        // Clone shared data for the task
        let running = self.running.clone();
        let state = self.state.clone();
        let node_scores = self.node_scores.clone();
        let current_proposers = self.current_proposers.clone();
        let block_times = self.block_times.clone();
        let last_proposer_update = self.last_proposer_update.clone();
        let mut shutdown_receiver =
            std::mem::replace(&mut self.shutdown_receiver, broadcast::channel::<()>(1).1);
        let block_sender = self.block_sender.clone();
        let node_id = self.node_id.clone();
        let svcp_config = self.svcp_config.clone();
        let mut current_difficulty = self.current_difficulty;
        let validator_count = self.validator_count.clone();
        let tps_scaling_enabled = self.tps_scaling_enabled;
        let parallel_processor = self.parallel_processor.clone();

        let handle = tokio::spawn(async move {
            info!("SVCP miner started with difficulty: {current_difficulty}");

            // Update proposers at startup
            if let Err(e) =
                Self::update_proposer_candidates(&node_scores, &current_proposers, &svcp_config)
                    .await
            {
                warn!("Failed to update proposer candidates: {e}");
            }

            let mut mining_interval = tokio::time::interval(Duration::from_secs(1));

            loop {
                tokio::select! {
                    _ = mining_interval.tick() => {
                        // Skip mining if TPS scaling is enabled (parallel processor handles it)
                        if tps_scaling_enabled {
                            // Update validator count for parallel processor
                            if let Some(processor) = &parallel_processor {
                                let validators = {
                                    let proposers = current_proposers.lock().await;
                                    proposers.len().max(1)
                                };

                                // Update validator count in shared state
                                {
                                    let mut count = validator_count.lock().await;
                                    *count = validators;
                                }

                                // Update the processor with current validator count
                                processor.update_miner_count(validators);

                                // Log estimated TPS
                                let estimated_tps = processor.get_estimated_tps();
                                debug!("Estimated TPS with {validators} validators: {estimated_tps:.2}");
                            }

                            // Skip regular mining as parallel processor handles it
                            continue;
                        }

                        // Regular mining flow (if TPS scaling is disabled)

                        // Check if this node is allowed to propose
                        let allowed = {
                            let proposers = current_proposers.lock().await;
                            proposers.contains(&node_id)
                        };

                        if !allowed {
                            // Not in proposer set, skip mining
                            continue;
                        }

                        // Periodically update the proposer candidates
                        let should_update = {
                            let last_update = last_proposer_update.lock().await;
                            last_update.elapsed() > Duration::from_secs(60)
                        };

                        if should_update {
                            if let Err(e) = Self::update_proposer_candidates(&node_scores, &current_proposers, &svcp_config).await {
                                warn!("Failed to update proposer candidates: {e}");
                            }

                            let mut last_update = last_proposer_update.lock().await;
                            *last_update = Instant::now();
                        }

                        // Try to create a new block to mine
                        match Self::create_candidate_block(&state, &node_id).await {
                            Ok(block) => {
                                // Try to mine the block
                                let mining_result = Self::mine_block(block.clone(), current_difficulty, running.clone()).await;

                                match mining_result {
                                    MiningResult::Success(mined_block) => {
                                        info!("Successfully mined block: {}", mined_block.hash());

                                        // Record block time for difficulty adjustment
                                        {
                                            let mut times = block_times.lock().await;
                                            times.push((SystemTime::now(), Duration::from_secs(15)));

                                            // Keep only the last adjustment window
                                            if times.len() > svcp_config.difficulty_adjustment_window as usize {
                                                times.remove(0);
                                            }
                                        }

                                        // Calculate and apply block reward with trust multiplier
                                        let _block_hash = mined_block.hash_bytes();
                                        let proposer_id = mined_block.header.proposer_id.clone();

                                        // Use static method instead of self.calculate_block_reward
                                        let reward = Self::static_calculate_block_reward(
                                            _block_hash.as_bytes(),
                                            &proposer_id,
                                            &node_scores
                                        ).await;

                                        info!("Applied block reward of {reward} tokens to miner {proposer_id}");

                                        // Send mined block
                                        if let Err(e) = block_sender.send(mined_block).await {
                                            warn!("Failed to send mined block: {e}");
                                        }

                                        // Adjust difficulty (static method without 'self' access)
                                        current_difficulty = Self::static_adjust_difficulty(
                                            &block_times,
                                            current_difficulty,
                                            &svcp_config
                                        ).await.unwrap_or(current_difficulty);
                                    },
                                    MiningResult::Interrupted => {
                                        debug!("Mining interrupted");
                                    },
                                    MiningResult::Error(err) => {
                                        warn!("Mining error: {err}");
                                    }
                                }
                            },
                            Err(e) => {
                                warn!("Failed to create candidate block: {e}");
                            }
                        }
                    },
                    _ = shutdown_receiver.recv() => {
                        info!("Received shutdown signal, stopping SVCP miner");
                        break;
                    }
                }
            }

            // If we have a parallel processor handle, wait for it
            if let Some(handle) = parallel_handle {
                if let Err(e) = handle.await {
                    warn!("Parallel processor task failed: {e:?}");
                }
            }

            info!("SVCP miner stopped");
        });

        Ok(handle)
    }

    /// Update the set of proposer candidates based on node scores
    async fn update_proposer_candidates(
        node_scores: &Arc<Mutex<HashMap<String, NodeScore>>>,
        current_proposers: &Arc<Mutex<Vec<String>>>,
        svcp_config: &SVCPConfig,
    ) -> Result<()> {
        let scores = node_scores.lock().await;

        // Create a sorted heap of candidates based on scores
        let mut candidates = BinaryHeap::new();

        for (node_id, score) in scores.iter() {
            if score.overall_score >= svcp_config.min_score_threshold {
                // Calculate weighted score for consensus (may differ from security score)
                let weighted_score = score.device_health_score * svcp_config.device_weight
                    + score.network_score * svcp_config.network_weight
                    + score.storage_score * svcp_config.storage_weight
                    + score.engagement_score * svcp_config.engagement_weight
                    + score.ai_behavior_score * svcp_config.ai_behavior_weight;

                candidates.push(ProposerCandidate {
                    node_id: node_id.clone(),
                    score: weighted_score,
                    last_proposed: score.last_updated,
                });
            }
        }

        // If we don't have enough candidates, warn but continue with what we have
        if candidates.len() < svcp_config.min_proposer_candidates {
            warn!(
                "Not enough proposer candidates: {} (min: {})",
                candidates.len(),
                svcp_config.min_proposer_candidates
            );
        }

        // Select top N candidates
        let mut selected = Vec::new();
        for _ in 0..candidates.len().min(svcp_config.max_proposer_candidates) {
            if let Some(candidate) = candidates.pop() {
                selected.push(candidate.node_id);
            }
        }

        // Update current proposers
        {
            let mut proposers = current_proposers.lock().await;
            *proposers = selected;
        }

        info!(
            "Updated proposer candidates: {} nodes selected",
            current_proposers.lock().await.len()
        );

        Ok(())
    }

    /// Create a candidate block from current state
    async fn create_candidate_block(
        state: &Arc<RwLock<BlockchainState>>,
        node_id: &str,
    ) -> Result<Block> {
        // Get the current blockchain state to build on top of
        let state_guard = state.read().await;

        // Get the last block hash and height from state
        let previous_hash = state_guard.get_latest_block_hash()?;
        let previous_hash = crate::types::Hash::from_hex(&previous_hash)?;
        let height = state_guard.get_height()? + 1;

        // Get pending transactions from the mempool
        // In a real implementation, this would fetch pending transactions from a mempool
        let transactions = state_guard.get_pending_transactions(10); // Limit to 10 transactions for now
                                                                     // transactions is already Vec<crate::ledger::transaction::Transaction>

        // Get the shard ID from state or config
        let shard_id = state_guard.get_shard_id()?; // Use ? operator to propagate the Result

        // Create a block using this node as proposer
        let block = Block::new(
            previous_hash,
            transactions,
            height,
            4, // Initial difficulty, this should be adjusted based on network conditions
            node_id.to_string(),
            shard_id,
        );

        Ok(block)
    }

    /// Mine a block with proof-of-work
    async fn mine_block(
        mut block: Block,
        difficulty: u64,
        running: Arc<Mutex<bool>>,
    ) -> MiningResult {
        // Calculate target threshold based on difficulty
        let target = 1u64 << (64 - difficulty);
        let start_time = Instant::now();

        debug!("Starting mining with difficulty {difficulty}, target: {target}");

        // Try different nonces until we find one that satisfies the difficulty
        for nonce in 0..u64::MAX {
            // Check every 100 iterations if we're still running
            if nonce % 100 == 0 {
                let is_running = *running.lock().await;
                if !is_running {
                    debug!("Mining interrupted after checking {nonce} nonces");
                    return MiningResult::Interrupted;
                }
            }

            // Set nonce using the BlockExt trait
            block.set_nonce(nonce);

            // Calculate block hash using the BlockExt trait
            let hash_bytes = block.hash_pow_bytes();

            // Convert first 8 bytes to u64 for difficulty check
            let hash_value = if hash_bytes.as_bytes().len() >= 8 {
                u64::from_be_bytes([
                    hash_bytes.as_bytes()[0],
                    hash_bytes.as_bytes()[1],
                    hash_bytes.as_bytes()[2],
                    hash_bytes.as_bytes()[3],
                    hash_bytes.as_bytes()[4],
                    hash_bytes.as_bytes()[5],
                    hash_bytes.as_bytes()[6],
                    hash_bytes.as_bytes()[7],
                ])
            } else {
                // Handle case where hash is shorter than 8 bytes (shouldn't happen with our hash functions)
                continue;
            };

            // Check if hash meets difficulty
            if hash_value < target {
                // Found a valid nonce!
                let duration = start_time.elapsed();
                debug!(
                    "Successfully mined block with nonce {nonce} in {duration:?}, hash: {}",
                    block.hash()
                );
                return MiningResult::Success(block);
            }
        }

        MiningResult::Error("Exhausted nonce space".to_string())
    }

    /// Stop the mining process
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.lock().await;
        *running = false;
        Ok(())
    }

    /// Check if a given node is allowed to propose blocks
    pub async fn is_allowed_proposer(&self, node_id: &str) -> bool {
        let proposers = self.current_proposers.lock().await;
        proposers.contains(&node_id.to_string())
    }

    /// Get the current difficulty
    pub async fn get_difficulty(&self) -> u64 {
        self.current_difficulty
    }

    /// Get the current list of proposers
    pub async fn get_proposers(&self) -> Vec<String> {
        self.current_proposers.lock().await.clone()
    }

    /// Select proposer for this round using weighted lottery
    pub async fn select_lottery_winner(&self) -> Option<String> {
        let proposers = self.current_proposers.lock().await;
        let scores = self.node_scores.lock().await;

        if proposers.is_empty() {
            return None;
        }

        // Calculate total weight
        let mut total_weight = 0.0;
        let mut weights = Vec::new();

        for proposer in proposers.iter() {
            let score = scores.get(proposer).map(|s| s.overall_score).unwrap_or(0.0);
            weights.push(score);
            total_weight += score;
        }

        if total_weight <= 0.0 {
            // If no valid weights, choose randomly
            let mut rng = thread_rng();
            return proposers.choose(&mut rng).cloned();
        }

        // Weighted random selection
        let mut rng = thread_rng();

        // Using newer range syntax
        let selection = rng.gen_range(0.0..total_weight);

        let mut cumulative = 0.0;
        for (i, weight) in weights.iter().enumerate() {
            cumulative += weight;
            if cumulative >= selection {
                return Some(proposers[i].clone());
            }
        }

        // Fallback - should not reach here
        proposers.last().cloned()
    }

    /// Get the current estimated TPS based on validator count
    pub fn get_estimated_tps(&self) -> f32 {
        let multiplier = self.tps_multiplier;
        let miner_count = match self.validator_count.try_lock() {
            Ok(guard) => *guard,
            Err(_) => 1,
        };

        // Base TPS is 1000 * multiplier * miner_count
        let base_tps = 1000.0;
        base_tps * multiplier * miner_count as f32
    }

    /// Precompute verification patterns for optimized mining
    pub async fn precompute_verification_patterns(&mut self) -> Result<()> {
        info!("Precomputing verification patterns for optimized mining");

        // In a real implementation, this would prepare lookup tables or other
        // optimizations for hash verification. For now, this is a placeholder.

        // Simulate precomputation work
        let patterns = vec![
            // Example patterns - in real implementation, these would be
            // computed based on current difficulty and network state
            [0u8; 32], [1u8; 32], [2u8; 32],
        ];

        info!("Precomputed {} verification patterns", patterns.len());

        // In a real implementation, we would store these patterns
        // For now, just log that we did the work
        debug!("Verification pattern precomputation complete");

        Ok(())
    }

    /// Calculate block rewards statically
    pub async fn static_calculate_block_reward(
        _block_hash: &[u8],
        proposer_id: &str,
        node_scores: &Arc<Mutex<HashMap<String, NodeScore>>>,
    ) -> u64 {
        // Base reward is 50 tokens
        let base_reward: u64 = 50;

        // Calculate multiplier based on proposer's trust score
        let multiplier =
            Self::static_get_proposer_reward_multiplier(proposer_id, node_scores).await;

        // Apply multiplier to base reward
        let total_reward = (base_reward as f32 * multiplier) as u64;

        // Minimum reward is 10 tokens
        total_reward.max(10)
    }

    /// Static version of get_proposer_reward_multiplier that doesn't require self
    async fn static_get_proposer_reward_multiplier(
        proposer_id: &str,
        node_scores: &Arc<Mutex<HashMap<String, NodeScore>>>,
    ) -> f32 {
        // Default multiplier if no score found
        let default_multiplier = 1.0;

        // Get proposer's score
        let scores = node_scores.lock().await;

        if let Some(score) = scores.get(proposer_id) {
            // Calculate multiplier based on overall score
            // Score range is 0-1, but we want multiplier range of 0.5-2.0
            let range = 0.5..=2.0;
            let multiplier = *range.start() + (*range.end() - *range.start()) * score.overall_score;

            // Apply AI behavior specific bonus/penalty
            let ai_behavior_factor = if score.ai_behavior_score > 0.8 {
                // Bonus for excellent AI behavior
                1.2
            } else if score.ai_behavior_score < 0.3 {
                // Penalty for poor AI behavior
                0.8
            } else {
                // Neutral for average behavior
                1.0
            };

            return multiplier * ai_behavior_factor;
        }

        default_multiplier
    }

    /// Adjust difficulty based on recent block times
    pub async fn adjust_difficulty(&mut self) -> u64 {
        let locked_block_times = self.block_times.lock().await;

        // If we don't have enough blocks, return current difficulty
        if locked_block_times.len() < self.svcp_config.difficulty_adjustment_window as usize {
            return self.current_difficulty;
        }

        // Calculate average block time
        let mut total_time = Duration::from_secs(0);
        for i in 1..locked_block_times.len() {
            let (prev_time, _) = locked_block_times[i - 1];
            let (current_time, _) = locked_block_times[i];

            // Calculate time between blocks
            if let Ok(duration) = current_time.duration_since(prev_time) {
                total_time += duration;
            }
        }

        let avg_block_time = total_time.as_secs_f64() / (locked_block_times.len() - 1) as f64;
        let target_time = self.svcp_config.target_block_time as f64;

        // Adjust difficulty based on ratio
        let ratio = avg_block_time / target_time;

        // If blocks are too slow, decrease difficulty
        // If blocks are too fast, increase difficulty
        let mut new_difficulty = self.current_difficulty as f64;

        if ratio > 1.2 {
            // Blocks are too slow, reduce difficulty (max 50% decrease)
            new_difficulty *= 2.0 - ratio.min(1.5);
        } else if ratio < 0.8 {
            // Blocks are too fast, increase difficulty (max 50% increase)
            new_difficulty *= 2.0 - ratio.max(0.5);
        }

        // Ensure difficulty never goes below 1
        let new_difficulty = new_difficulty.max(1.0) as u64;

        // Update current difficulty
        self.current_difficulty = new_difficulty;

        info!(
            "Adjusted difficulty from {} to {} (avg block time: {:.2}s, target: {}s)",
            self.current_difficulty, new_difficulty, avg_block_time, target_time
        );

        new_difficulty
    }

    /// Adjust difficulty based on recent block times (static method)
    pub async fn static_adjust_difficulty(
        block_times: &Arc<Mutex<Vec<(SystemTime, Duration)>>>,
        current_difficulty: u64,
        svcp_config: &SVCPConfig,
    ) -> Result<u64> {
        let locked_block_times = block_times.lock().await;

        // If we don't have enough blocks, return current difficulty
        if locked_block_times.len() < svcp_config.difficulty_adjustment_window as usize {
            return Ok(current_difficulty);
        }

        // Calculate average block time
        let mut total_time = Duration::from_secs(0);
        for i in 1..locked_block_times.len() {
            let (prev_time, _) = locked_block_times[i - 1];
            let (current_time, _) = locked_block_times[i];

            // Calculate time between blocks
            if let Ok(duration) = current_time.duration_since(prev_time) {
                total_time += duration;
            }
        }

        let avg_block_time = total_time.as_secs_f64() / (locked_block_times.len() - 1) as f64;
        let target_time = svcp_config.target_block_time as f64;

        // Adjust difficulty based on ratio
        let ratio = avg_block_time / target_time;

        // If blocks are too slow, decrease difficulty
        // If blocks are too fast, increase difficulty
        let mut new_difficulty = current_difficulty as f64;

        if ratio > 1.2 {
            // Blocks are too slow, reduce difficulty (max 50% decrease)
            new_difficulty *= 2.0 - ratio.min(1.5);
        } else if ratio < 0.8 {
            // Blocks are too fast, increase difficulty (max 50% increase)
            new_difficulty *= 2.0 - ratio.max(0.5);
        }

        // Ensure difficulty never goes below 1
        let new_difficulty = new_difficulty.max(1.0) as u64;

        Ok(new_difficulty)
    }

    /// Load validators from genesis file
    pub async fn load_validators_from_genesis(
        &mut self,
        genesis_path: &std::path::Path,
        node_scores: &Arc<Mutex<HashMap<String, NodeScore>>>,
    ) -> Result<()> {
        use serde_json::Value;
        use std::fs::File;
        use std::io::BufReader;

        // Read and parse genesis file
        let file =
            File::open(genesis_path).map_err(|e| anyhow!("Failed to open genesis file: {}", e))?;
        let reader = BufReader::new(file);
        let genesis: Value = serde_json::from_reader(reader)
            .map_err(|e| anyhow!("Failed to parse genesis file: {}", e))?;

        // Extract validator set
        if let Some(validator_set) = genesis.get("validator_set").and_then(|v| v.as_array()) {
            let mut proposers = Vec::new();
            let mut scores = node_scores.lock().await;

            for validator in validator_set {
                if let Some(node_id) = validator.get("node_id").and_then(|v| v.as_str()) {
                    let power = validator
                        .get("power")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(100);

                    // Convert power to score (normalized to 0-1 range)
                    let score = (power as f32) / 100.0;

                    // Add to proposers list
                    proposers.push(node_id.to_string());

                    // Create or update node score
                    let node_score = scores.entry(node_id.to_string()).or_insert_with(|| {
                        let mut score = NodeScore {
                            overall_score: 0.7, // Default score
                            device_health_score: 0.7,
                            network_score: 0.7,
                            storage_score: 0.7,
                            engagement_score: 0.7,
                            ai_behavior_score: 0.7,
                            last_updated: SystemTime::now(),
                            history: Vec::new(),
                        };

                        // Add initial history point
                        score.history.push((SystemTime::now(), score.overall_score));
                        score
                    });

                    // Update score based on validator power
                    node_score.overall_score = node_score.overall_score.max(score);

                    info!("Loaded genesis validator: {node_id} with power: {power}");
                }
            }

            // Update current proposers
            let mut current_proposers = self.current_proposers.lock().await;
            *current_proposers = proposers;

            info!("Loaded {} validators from genesis", current_proposers.len());
        }

        Ok(())
    }

    /// Ultra-lightweight consensus optimization for high TPS
    pub async fn optimize_for_high_throughput(&mut self) -> Result<()> {
        // Configure for maximum throughput
        self.tps_multiplier = 50.0; // Massively increase TPS multiplier
        self.tps_scaling_enabled = true;

        // Enable SIMD-accelerated hash verification
        // Enable hardware acceleration if available
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    self.enable_simd_verification = true;
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    self.enable_simd_verification = true;
                }
            }
        }

        // Pre-compute common verification patterns
        self.precompute_verification_patterns().await?;

        // Initialize optimized parallel processor with larger worker pool
        let processor_config = ParallelProcessorConfig {
            worker_threads: num_cpus::get() * 4, // 4x logical cores
            batch_size: 10000,
            use_work_stealing: true,
            prefetch_enabled: true,
            pipeline_verification: true,
        };

        self.parallel_processor = Some(ParallelProcessor::new_with_config(
            self.state.clone(),
            self.block_sender.clone(),
            Some(self.tps_multiplier),
            processor_config,
        ));

        // Start optimized processor
        if let Some(processor) = &self.parallel_processor {
            processor.start_optimized().await?;
        }

        // Enable dynamic puzzle adjustment
        self.enable_dynamic_puzzle_adjustment().await?;

        Ok(())
    }

    /// Enable dynamic puzzle adjustment (mandatory optimization)
    async fn enable_dynamic_puzzle_adjustment(&mut self) -> Result<()> {
        // Create adaptive difficulty adjuster
        let adaptive_adjuster = DynamicPuzzleAdjuster::new(
            self.current_difficulty,
            self.svcp_config.target_block_time,
            AdaptiveConfig {
                min_difficulty: 1,
                max_difficulty: 16,
                responsiveness: 0.8,   // How quickly difficulty adapts (0-1)
                smoothing_factor: 0.3, // Smooths out fluctuations
                target_tps: 500_000.0, // Target 500k TPS
            },
        );

        // Store the adjuster
        self.dynamic_adjuster = Some(adaptive_adjuster);

        // Enable real-time monitoring for block time
        let monitor = BlockTimeMonitor::new(1000); // Keep history of 1000 blocks
        self.block_time_monitor = Some(monitor);

        // Log that dynamic adjustment is enabled
        log::info!("Dynamic puzzle adjustment enabled. Target TPS: 500,000");

        Ok(())
    }

    /// SIMD-accelerated hash verification
    #[cfg(target_arch = "x86_64")]
    fn verify_hash_simd(&self, hash: &[u8], target: &[u8]) -> bool {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            let hash_chunks = hash.chunks_exact(32);
            let target_chunks = target.chunks_exact(32);

            for (h, t) in hash_chunks.zip(target_chunks) {
                let hash_vec = _mm256_loadu_si256(h.as_ptr() as *const __m256i);
                let target_vec = _mm256_loadu_si256(t.as_ptr() as *const __m256i);

                // Compare hash with target (hash must be less than target)
                let cmp = _mm256_cmpgt_epi8(target_vec, hash_vec);
                let mask = _mm256_movemask_epi8(cmp);

                if mask != 0 {
                    return false;
                }
            }

            true
        } else {
            // Fall back to standard comparison
            self.verify_hash_standard(hash, target)
        }
    }

    /// Standard hash verification (fallback)
    #[allow(dead_code)]
    fn verify_hash_standard(&self, hash: &[u8], target: &[u8]) -> bool {
        hash.iter().zip(target.iter()).all(|(h, t)| h <= t)
    }

    /// Mine a block with optimized verification
    #[allow(dead_code)]
    async fn mine_block_optimized(
        &self,
        mut block: Block,
        difficulty: u64,
        running: Arc<Mutex<bool>>,
        _simd_enabled: bool,
    ) -> MiningResult {
        let target = calculate_target(difficulty);
        let start = Instant::now();
        let mut nonce: u64 = thread_rng().gen();
        let batch_size = 10000; // Check many nonces before updating timestamp

        // Pre-allocate buffers
        let mut hash_buffer = [0u8; 32];

        loop {
            // Update timestamp periodically
            if start.elapsed() > Duration::from_secs(1) {
                block.header.timestamp = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
            }

            // Check if mining should continue
            if let Ok(running_guard) = running.try_lock() {
                if !*running_guard {
                    return MiningResult::Interrupted;
                }
            } else {
                // If we couldn't get the lock, check again next time
                continue;
            }

            // Try a batch of nonces
            for _ in 0..batch_size {
                block.header.nonce = nonce;
                nonce = nonce.wrapping_add(1);

                // Calculate hash
                let hash = block.hash();
                hash_buffer.copy_from_slice(hash.as_bytes());

                // Verify hash against target - use the standard verification method
                // instead of the SIMD one which is platform-specific
                let valid = self.verify_hash_standard(&hash_buffer, &target);

                if valid {
                    return MiningResult::Success(block);
                }
            }
        }
    }

    /// Get the node ID
    pub fn get_node_id(&self) -> String {
        self.node_id.clone()
    }
}

/// Dynamic puzzle adjuster for adaptive difficulty
pub struct DynamicPuzzleAdjuster {
    current_difficulty: u64,
    target_block_time: u64,
    adaptive_config: AdaptiveConfig,
    last_adjustment: Instant,
    moving_avg_block_time: f64,
    current_tps_estimate: f64,
}

/// Configuration for adaptive difficulty
pub struct AdaptiveConfig {
    pub min_difficulty: u64,
    pub max_difficulty: u64,
    pub responsiveness: f64,
    pub smoothing_factor: f64,
    pub target_tps: f64,
}

impl DynamicPuzzleAdjuster {
    /// Create a new dynamic puzzle adjuster
    pub fn new(initial_difficulty: u64, target_block_time: u64, config: AdaptiveConfig) -> Self {
        Self {
            current_difficulty: initial_difficulty,
            target_block_time,
            adaptive_config: config,
            last_adjustment: Instant::now(),
            moving_avg_block_time: target_block_time as f64,
            current_tps_estimate: 10000.0, // Initial estimate
        }
    }

    /// Update with a new block time observation
    pub fn update(&mut self, block_time: Duration, transactions: usize) -> u64 {
        // Update moving average
        let block_time_secs = block_time.as_secs_f64();
        self.moving_avg_block_time = self.moving_avg_block_time
            * (1.0 - self.adaptive_config.smoothing_factor)
            + block_time_secs * self.adaptive_config.smoothing_factor;

        // Update TPS estimate
        let tps = transactions as f64 / block_time_secs;
        self.current_tps_estimate = self.current_tps_estimate
            * (1.0 - self.adaptive_config.smoothing_factor)
            + tps * self.adaptive_config.smoothing_factor;

        // Calculate adjustment factor
        let time_ratio = self.target_block_time as f64 / self.moving_avg_block_time;
        let tps_ratio = self.current_tps_estimate / self.adaptive_config.target_tps;

        // Combined adjustment factor (time based and TPS based)
        let adjustment_factor = time_ratio * 0.5 + tps_ratio * 0.5;

        // Apply responsiveness dampening
        let dampened_adjustment =
            1.0 + self.adaptive_config.responsiveness * (adjustment_factor - 1.0);

        // Update difficulty (higher factor = higher difficulty)
        self.current_difficulty =
            ((self.current_difficulty as f64) * dampened_adjustment).round() as u64;

        // Clamp to limits
        self.current_difficulty = self.current_difficulty.clamp(
            self.adaptive_config.min_difficulty,
            self.adaptive_config.max_difficulty,
        );

        self.last_adjustment = Instant::now();
        self.current_difficulty
    }
}

/// Monitor for tracking block times
pub struct BlockTimeMonitor {
    block_times: VecDeque<(Instant, usize)>, // (timestamp, tx_count)
    capacity: usize,
}

impl BlockTimeMonitor {
    pub fn new(capacity: usize) -> Self {
        Self {
            block_times: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add a new block time observation
    pub fn add_observation(&mut self, timestamp: Instant, tx_count: usize) {
        self.block_times.push_back((timestamp, tx_count));

        if self.block_times.len() > self.capacity {
            self.block_times.pop_front();
        }
    }

    /// Calculate average TPS over the last N blocks
    pub fn average_tps(&self, blocks: usize) -> f64 {
        if self.block_times.len() < 2 || blocks < 1 {
            return 0.0;
        }

        let blocks_to_consider = std::cmp::min(blocks, self.block_times.len() - 1);
        let newest_idx = self.block_times.len() - 1;
        let oldest_idx = newest_idx - blocks_to_consider;

        let newest = &self.block_times[newest_idx];
        let oldest = &self.block_times[oldest_idx];

        let time_diff = newest.0.duration_since(oldest.0).as_secs_f64();
        if time_diff <= 0.0 {
            return 0.0;
        }

        // Sum transactions in the window
        let tx_sum: usize = self
            .block_times
            .iter()
            .skip(oldest_idx + 1) // Skip oldest since we're measuring from it
            .map(|(_, tx_count)| tx_count)
            .sum();

        tx_sum as f64 / time_diff
    }
}

/// SVCPConsensus implements the consensus protocol interface
pub struct SVCPConsensus {
    /// Internal miner instance
    pub miner: Arc<RwLock<SVCPMiner>>,
    /// Configuration
    pub config: Config,
    /// Blockchain state
    pub state: Arc<RwLock<BlockchainState>>,
    /// Node scores
    pub node_scores: Arc<Mutex<HashMap<String, NodeScore>>>,
    /// Running flag
    pub running: Arc<Mutex<bool>>,
}

impl SVCPConsensus {
    /// Create a new SVCP consensus instance
    pub fn new(
        config: Config,
        state: Arc<RwLock<BlockchainState>>,
        node_scores: Arc<Mutex<HashMap<String, NodeScore>>>,
    ) -> Result<Self> {
        // Create block channel
        let (block_sender, _) = mpsc::channel(100);

        // Create shutdown channel
        let (_, shutdown_receiver) = broadcast::channel::<()>(1);

        // Create miner
        let miner = SVCPMiner::new(
            config.clone(),
            state.clone(),
            block_sender,
            shutdown_receiver,
            node_scores.clone(),
            None,
        )?;

        Ok(Self {
            miner: Arc::new(RwLock::new(miner)),
            config,
            state,
            node_scores,
            running: Arc::new(Mutex::new(false)),
        })
    }

    /// Start the consensus process
    pub async fn start(&self) -> Result<JoinHandle<()>> {
        let mut miner = self.miner.write().await;
        let handle = miner.start().await?;

        // Set running flag
        {
            let mut running = self.running.lock().await;
            *running = true;
        }

        Ok(handle)
    }

    /// Stop the consensus process
    pub async fn stop(&self) -> Result<()> {
        // Stop miner
        {
            let miner = self.miner.read().await;
            miner.stop().await?;
        }

        // Set running flag
        {
            let mut running = self.running.lock().await;
            *running = false;
        }

        Ok(())
    }

    /// Get the current difficulty
    pub async fn get_difficulty(&self) -> u64 {
        let miner = self.miner.read().await;
        miner.get_difficulty().await
    }

    /// Check if a node is allowed to propose
    pub async fn is_allowed_proposer(&self, node_id: &str) -> bool {
        let miner = self.miner.read().await;
        miner.is_allowed_proposer(node_id).await
    }

    /// Process a new block received from the network
    pub async fn process_block(&self, block: Block) -> Result<bool> {
        // Validate the block
        // Verify POW
        // Verify proposer is valid
        // Verify transactions
        let _block_hash = block.hash();

        // In a real implementation, this would verify and process the block
        // For now, just log and return success
        info!("SVCP processed block: {_block_hash}");

        // Update blockchain state (in a real implementation)

        Ok(true)
    }

    /// Initialize genesis validators
    pub async fn initialize_genesis_validators(&self, path: &std::path::Path) -> Result<()> {
        let mut miner = self.miner.write().await;
        miner
            .load_validators_from_genesis(path, &self.node_scores)
            .await
    }
}

/// Cross-shard consensus interface to avoid being gated by feature flags
pub struct CrossShardConsensus {
    /// Parent consensus instance
    #[allow(dead_code)]
    consensus: Arc<SVCPConsensus>,
}

impl CrossShardConsensus {
    /// Create a new cross-shard consensus instance
    pub fn new(consensus: Arc<SVCPConsensus>) -> Self {
        Self { consensus }
    }

    /// Process a cross-shard transaction
    pub async fn process_cross_shard_tx(
        &self,
        _tx_hash: &str,
        _from_shard: u32,
        _to_shard: u32,
    ) -> Result<()> {
        // In a real implementation, this would coordinate with other shards
        // For now, just log and return success
        info!("Processing cross-shard transaction: {_tx_hash}");
        Ok(())
    }

    /// Verify a cross-shard transaction
    pub fn verify_cross_shard_tx(&self, _tx_hash: &str) -> bool {
        // In a real implementation, this would verify the transaction
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proposer_candidate_ordering() {
        use std::collections::BinaryHeap;
        use std::time::{Duration, SystemTime};

        let now = SystemTime::now();
        let one_hour_ago = now - Duration::from_secs(3600);
        let two_hours_ago = now - Duration::from_secs(7200);

        // Create a few candidates with various scores and last proposed times
        let candidates = vec![
            ProposerCandidate {
                node_id: "node1".to_string(),
                score: 0.8,
                last_proposed: now,
            },
            ProposerCandidate {
                node_id: "node2".to_string(),
                score: 0.9,
                last_proposed: now,
            },
            ProposerCandidate {
                node_id: "node3".to_string(),
                score: 0.8,
                last_proposed: one_hour_ago,
            },
            ProposerCandidate {
                node_id: "node4".to_string(),
                score: 0.7,
                last_proposed: two_hours_ago,
            },
        ];

        // Create a binary heap (max heap)
        let mut heap = BinaryHeap::new();
        for candidate in candidates {
            heap.push(candidate);
        }

        // Extract in order to see actual ordering
        // BinaryHeap pops the "greatest" elements first by Ord implementation
        // So with our implementation prioritizing older timestamps:
        // - node4 has the oldest timestamp (2 hours ago)
        // - then node3 (1 hour ago)
        // - then node2 and node1 (both now, but node2 has higher score)
        let first = heap.pop().unwrap();
        let second = heap.pop().unwrap();
        let third = heap.pop().unwrap();
        let fourth = heap.pop().unwrap();

        // Print the actual order to help debugging
        println!(
            "Actual ordering: {}, {}, {}, {}",
            first.node_id, second.node_id, third.node_id, fourth.node_id
        );

        // Test ordering - older timestamps come first, then higher scores
        assert_eq!(first.node_id, "node4"); // Oldest timestamp
        assert_eq!(second.node_id, "node3"); // Second oldest timestamp
        assert_eq!(third.node_id, "node2"); // Same timestamp as node1, but higher score
        assert_eq!(fourth.node_id, "node1"); // Same timestamp as node2, but lower score
    }
}
