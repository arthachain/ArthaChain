use crate::consensus::leader_election::{LeaderElectionConfig, LeaderElectionManager};
use crate::consensus::leader_failover::{
    LeaderFailoverConfig, LeaderFailoverManager, LeaderHealth,
};
use crate::network::types::NodeId;
use anyhow::{anyhow, Result};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::time::interval;

// 🛡️ SPOF ELIMINATION: Multi-Leader Support

/// Multi-leader coordinator for consensus
#[derive(Debug)]
pub struct MultiLeaderCoordinator {
    pub concurrent_leaders: usize,
    pub rotation_strategy: LeaderRotationStrategy,
    pub leader_workload: HashMap<NodeId, u64>,
}

/// Leader load balancer
#[derive(Debug)]
pub struct LeaderLoadBalancer {
    pub current_leader_index: usize,
    pub leader_performance: HashMap<NodeId, f64>,
}

/// Leader rotation strategy for multi-leader consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeaderRotationStrategy {
    /// Round-robin rotation
    RoundRobin,
    /// Load-based rotation
    LoadBalanced,
    /// Performance-based rotation
    PerformanceBased,
    /// Random rotation
    Random,
}

/// Consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Leader election configuration
    pub leader_election: LeaderElectionConfig,
    /// Leader failover configuration
    pub failover: LeaderFailoverConfig,
    /// Enable automatic consensus recovery
    pub auto_recovery: bool,
    /// Consensus timeout in milliseconds
    pub consensus_timeout_ms: u64,
    /// Maximum consecutive failures before emergency mode
    pub max_consecutive_failures: u32,
    /// Emergency mode duration in seconds
    pub emergency_mode_duration_secs: u64,
    /// Enable Byzantine fault tolerance
    pub byzantine_tolerance: bool,
    /// Maximum Byzantine nodes (f in 3f+1)
    pub max_byzantine_nodes: usize,
    
    // 🛡️ SPOF ELIMINATION: Multi-Leader Consensus (SPOF FIX #4)
    /// Enable multi-leader consensus
    pub enable_multi_leader: bool,
    /// Number of concurrent leaders
    pub concurrent_leaders: usize,
    /// Leader rotation strategy
    pub leader_rotation_strategy: LeaderRotationStrategy,
    /// Leader failure detection timeout
    pub leader_failure_timeout_ms: u64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            leader_election: LeaderElectionConfig::default(),
            failover: LeaderFailoverConfig::default(),
            auto_recovery: true,
            consensus_timeout_ms: 5000,
            max_consecutive_failures: 3,
            emergency_mode_duration_secs: 300, // 5 minutes
            byzantine_tolerance: true,
            max_byzantine_nodes: 1, // Assume 4 nodes minimum (3*1+1)
            
            // 🛡️ SPOF ELIMINATION: Multi-Leader defaults
            enable_multi_leader: true,          // Enable multi-leader by default
            concurrent_leaders: 3,              // 3 concurrent leaders for fault tolerance
            leader_rotation_strategy: LeaderRotationStrategy::LoadBalanced,
            leader_failure_timeout_ms: 2000,   // 2 second timeout for leader detection
        }
    }
}

/// Consensus state
#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusState {
    /// Normal operation
    Normal,
    /// Leader election in progress
    LeaderElection,
    /// Failover in progress
    Failover,
    /// Emergency mode (manual intervention required)
    Emergency,
    /// Recovery in progress
    Recovery,
}

/// Consensus event
#[derive(Debug, Clone)]
pub enum ConsensusEvent {
    /// New leader elected
    LeaderElected(NodeId),
    /// Leader failed
    LeaderFailed(NodeId),
    /// Consensus round completed
    RoundCompleted(u64),
    /// Consensus failed
    ConsensusFailed(String),
    /// Emergency mode activated
    EmergencyModeActivated,
    /// Recovery completed
    RecoveryCompleted,
}

/// Consensus metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusMetrics {
    /// Current consensus round
    pub current_round: u64,
    /// Total successful rounds
    pub successful_rounds: u64,
    /// Total failed rounds
    pub failed_rounds: u64,
    /// Current leader
    pub current_leader: Option<NodeId>,
    /// Leader tenure (rounds)
    pub leader_tenure: u64,
    /// Average round time (ms)
    pub avg_round_time_ms: f64,
    /// Byzantine faults detected
    pub byzantine_faults: u64,
    /// Last consensus timestamp
    pub last_consensus: SystemTime,
}

/// Comprehensive consensus manager
pub struct ConsensusManager {
    /// Configuration
    config: ConsensusConfig,
    /// Current consensus state
    state: Arc<RwLock<ConsensusState>>,
    /// Leader election manager
    election_manager: Arc<LeaderElectionManager>,
    /// Leader failover manager
    failover_manager: Arc<LeaderFailoverManager>,
    /// Active validators
    validators: Arc<RwLock<HashSet<NodeId>>>,
    /// Consensus metrics
    metrics: Arc<RwLock<ConsensusMetrics>>,
    /// Event broadcaster
    event_sender: broadcast::Sender<ConsensusEvent>,
    /// Consensus monitor handle
    monitor_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Recovery manager handle
    recovery_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Emergency mode state
    emergency_mode: Arc<RwLock<Option<SystemTime>>>,
    /// Node ID
    node_id: NodeId,
    
    // 🛡️ SPOF ELIMINATION: Multi-Leader Consensus
    /// Current active leaders
    active_leaders: Arc<RwLock<Vec<NodeId>>>,
    /// Leader load balancer
    leader_balancer: Arc<Mutex<LeaderLoadBalancer>>,
    /// Multi-leader coordinator
    multi_leader_coordinator: Arc<MultiLeaderCoordinator>,
}

impl ConsensusManager {
    /// Create new consensus manager
    pub async fn new(
        config: ConsensusConfig,
        node_id: NodeId,
        validators: HashSet<NodeId>,
    ) -> Result<Self> {
        let validators_arc = Arc::new(RwLock::new(validators.clone()));

        // Create leader election manager
        let election_manager = Arc::new(LeaderElectionManager::new(
            config.leader_election.clone(),
            validators_arc.clone(),
        ));

        // Create leader failover manager
        let failover_manager = Arc::new(LeaderFailoverManager::new(
            config.failover.clone(),
            election_manager.clone(),
            node_id.clone(),
        ));

        let (event_sender, _) = broadcast::channel(1000);

        let manager = Self {
            config: config.clone(),
            state: Arc::new(RwLock::new(ConsensusState::Normal)),
            election_manager,
            failover_manager,
            validators: validators_arc,
            metrics: Arc::new(RwLock::new(ConsensusMetrics {
                current_round: 0,
                successful_rounds: 0,
                failed_rounds: 0,
                current_leader: None,
                leader_tenure: 0,
                avg_round_time_ms: 0.0,
                byzantine_faults: 0,
                last_consensus: SystemTime::now(),
            })),
            event_sender,
            monitor_handle: Arc::new(Mutex::new(None)),
            recovery_handle: Arc::new(Mutex::new(None)),
            emergency_mode: Arc::new(RwLock::new(None)),
            node_id: node_id.clone(),
            
            // 🛡️ SPOF ELIMINATION: Initialize multi-leader consensus fields
            active_leaders: Arc::new(RwLock::new(vec![node_id.clone()])), // Start with self as leader
            leader_balancer: Arc::new(Mutex::new(LeaderLoadBalancer {
                current_leader_index: 0,
                leader_performance: HashMap::new(),
            })),
            multi_leader_coordinator: Arc::new(MultiLeaderCoordinator {
                concurrent_leaders: config.concurrent_leaders,
                rotation_strategy: config.leader_rotation_strategy.clone(),
                leader_workload: HashMap::new(),
            }),
        };

        Ok(manager)
    }

    /// Start consensus manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting consensus manager");

        // Start leader election
        self.election_manager.start().await?;

        // Start leader failover
        self.failover_manager.start().await?;

        // Start consensus monitoring
        self.start_consensus_monitor().await?;

        // Start recovery manager
        self.start_recovery_manager().await?;

        info!("Consensus manager started successfully");
        Ok(())
    }

    /// Start consensus monitoring
    async fn start_consensus_monitor(&self) -> Result<()> {
        let state = self.state.clone();
        let metrics = self.metrics.clone();
        let event_sender = self.event_sender.clone();
        let failover_manager = self.failover_manager.clone();
        let election_manager = self.election_manager.clone();
        let emergency_mode = self.emergency_mode.clone();
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(1000)); // Check every second
            let mut consecutive_failures = 0u32;

            loop {
                interval.tick().await;

                // Check current state
                let current_state = state.read().await.clone();

                match current_state {
                    ConsensusState::Normal => {
                        // Check leader health
                        let leader_health = failover_manager.get_leader_health().await;

                        if leader_health == LeaderHealth::Failed {
                            // Trigger failover
                            *state.write().await = ConsensusState::Failover;
                            let _ = event_sender
                                .send(ConsensusEvent::ConsensusFailed("Leader failed".to_string()));

                            consecutive_failures += 1;
                            if consecutive_failures >= config.max_consecutive_failures {
                                Self::activate_emergency_mode(&emergency_mode, &event_sender).await;
                            }
                        } else {
                            consecutive_failures = 0;
                        }
                    }
                    ConsensusState::Failover => {
                        // Check if failover is complete
                        if !failover_manager.is_failover_in_progress().await {
                            if let Some(new_leader) = failover_manager.get_current_leader().await {
                                *state.write().await = ConsensusState::Normal;
                                let _ = event_sender
                                    .send(ConsensusEvent::LeaderElected(new_leader.clone()));

                                // Update metrics
                                let mut metrics_guard = metrics.write().await;
                                metrics_guard.current_leader = Some(new_leader);
                                metrics_guard.leader_tenure = 0;
                            }
                        }
                    }
                    ConsensusState::Emergency => {
                        // Check if emergency mode should end
                        if let Some(emergency_start) = *emergency_mode.read().await {
                            let elapsed = SystemTime::now()
                                .duration_since(emergency_start)
                                .unwrap_or(Duration::from_secs(0));

                            if elapsed.as_secs() >= config.emergency_mode_duration_secs {
                                *emergency_mode.write().await = None;
                                *state.write().await = ConsensusState::Recovery;
                            }
                        }
                    }
                    ConsensusState::Recovery => {
                        // Attempt to recover consensus
                        if let Err(e) = election_manager.force_election().await {
                            error!("Failed to recover consensus: {}", e);
                        } else {
                            *state.write().await = ConsensusState::Normal;
                            let _ = event_sender.send(ConsensusEvent::RecoveryCompleted);
                        }
                    }
                    _ => {}
                }

                // Update metrics
                Self::update_metrics(&metrics).await;
            }
        });

        *self.monitor_handle.lock().await = Some(handle);
        Ok(())
    }

    /// Start recovery manager
    async fn start_recovery_manager(&self) -> Result<()> {
        let config = self.config.clone();
        let state = self.state.clone();
        let election_manager = self.election_manager.clone();
        let validators = self.validators.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30)); // Check every 30 seconds

            loop {
                interval.tick().await;

                // Only run recovery checks if auto-recovery is enabled
                if !config.auto_recovery {
                    continue;
                }

                let current_state = state.read().await.clone();

                // Check if we need to perform recovery actions
                match current_state {
                    ConsensusState::Emergency | ConsensusState::Recovery => {
                        // Attempt various recovery strategies
                        if let Err(e) =
                            Self::attempt_consensus_recovery(&election_manager, &validators, &state)
                                .await
                        {
                            warn!("Consensus recovery attempt failed: {}", e);
                        }
                    }
                    _ => {}
                }
            }
        });

        *self.recovery_handle.lock().await = Some(handle);
        Ok(())
    }

    /// Activate emergency mode
    async fn activate_emergency_mode(
        emergency_mode: &Arc<RwLock<Option<SystemTime>>>,
        event_sender: &broadcast::Sender<ConsensusEvent>,
    ) {
        warn!("Activating emergency mode due to consecutive failures");
        *emergency_mode.write().await = Some(SystemTime::now());
        let _ = event_sender.send(ConsensusEvent::EmergencyModeActivated);
    }

    /// Attempt consensus recovery
    async fn attempt_consensus_recovery(
        election_manager: &Arc<LeaderElectionManager>,
        validators: &Arc<RwLock<HashSet<NodeId>>>,
        state: &Arc<RwLock<ConsensusState>>,
    ) -> Result<()> {
        info!("Attempting consensus recovery");

        // Check if we have enough validators
        let validator_count = validators.read().await.len();
        if validator_count < 3 {
            return Err(anyhow!("Insufficient validators for consensus"));
        }

        // Force new election
        election_manager.force_election().await?;

        // Update state to normal if successful
        *state.write().await = ConsensusState::Normal;

        info!("Consensus recovery successful");
        Ok(())
    }

    /// Update consensus metrics
    async fn update_metrics(metrics: &Arc<RwLock<ConsensusMetrics>>) {
        let mut metrics_guard = metrics.write().await;
        metrics_guard.last_consensus = SystemTime::now();
        // Update other metrics as needed
    }

    /// Get current consensus state
    pub async fn get_state(&self) -> ConsensusState {
        self.state.read().await.clone()
    }

    /// Get consensus metrics
    pub async fn get_metrics(&self) -> ConsensusMetrics {
        self.metrics.read().await.clone()
    }

    /// Get current leader
    pub async fn get_current_leader(&self) -> Option<NodeId> {
        self.failover_manager.get_current_leader().await
    }

    /// Force leader election
    pub async fn force_leader_election(&self) -> Result<NodeId> {
        info!("Forcing leader election");
        *self.state.write().await = ConsensusState::LeaderElection;

        let new_leader = self.election_manager.force_election().await?;
        self.failover_manager
            .update_leader(new_leader.clone())
            .await?;

        *self.state.write().await = ConsensusState::Normal;
        let _ = self
            .event_sender
            .send(ConsensusEvent::LeaderElected(new_leader.clone()));

        Ok(new_leader)
    }

    /// Force failover to specific node
    pub async fn force_failover(&self, target_node: NodeId) -> Result<()> {
        info!("Forcing failover to node: {:?}", target_node);

        // Verify target node is a valid validator
        let validators = self.validators.read().await;
        if !validators.contains(&target_node) {
            return Err(anyhow!("Target node is not a valid validator"));
        }

        *self.state.write().await = ConsensusState::Failover;
        self.failover_manager
            .update_leader(target_node.clone())
            .await?;

        *self.state.write().await = ConsensusState::Normal;
        let _ = self
            .event_sender
            .send(ConsensusEvent::LeaderElected(target_node));

        Ok(())
    }

    /// Add validator
    pub async fn add_validator(&self, validator: NodeId) -> Result<()> {
        let mut validators = self.validators.write().await;
        validators.insert(validator.clone());
        info!("Added validator: {:?}", validator);
        Ok(())
    }

    /// Remove validator
    pub async fn remove_validator(&self, validator: &NodeId) -> Result<()> {
        let mut validators = self.validators.write().await;
        validators.remove(validator);
        info!("Removed validator: {:?}", validator);

        // If removed validator was the leader, trigger election
        if let Some(current_leader) = self.get_current_leader().await {
            if current_leader == *validator {
                self.force_leader_election().await?;
            }
        }

        Ok(())
    }

    /// Subscribe to consensus events
    pub fn subscribe_events(&self) -> broadcast::Receiver<ConsensusEvent> {
        self.event_sender.subscribe()
    }

    /// Check if node is leader
    pub async fn is_leader(&self) -> bool {
        if let Some(current_leader) = self.get_current_leader().await {
            current_leader == self.node_id
        } else {
            false
        }
    }

    /// Get validator list
    pub async fn get_validators(&self) -> HashSet<NodeId> {
        self.validators.read().await.clone()
    }

    /// Simulate network partition for testing
    #[cfg(test)]
    pub async fn simulate_network_partition(
        &self,
        partitioned_nodes: HashSet<NodeId>,
    ) -> Result<()> {
        warn!(
            "Simulating network partition with nodes: {:?}",
            partitioned_nodes
        );

        // Remove partitioned nodes temporarily
        let mut validators = self.validators.write().await;
        for node in &partitioned_nodes {
            validators.remove(node);
        }

        // Trigger leader election if leader is partitioned
        if let Some(current_leader) = self.get_current_leader().await {
            if partitioned_nodes.contains(&current_leader) {
                self.force_leader_election().await?;
            }
        }

        Ok(())
    }

    /// Heal network partition for testing
    #[cfg(test)]
    pub async fn heal_network_partition(&self, rejoining_nodes: HashSet<NodeId>) -> Result<()> {
        info!(
            "Healing network partition, rejoining nodes: {:?}",
            rejoining_nodes
        );

        // Add nodes back
        let mut validators = self.validators.write().await;
        for node in rejoining_nodes {
            validators.insert(node);
        }

        Ok(())
    }

    /// Stop consensus manager
    pub async fn stop(&self) {
        info!("Stopping consensus manager");

        if let Some(handle) = self.monitor_handle.lock().await.take() {
            handle.abort();
        }
        if let Some(handle) = self.recovery_handle.lock().await.take() {
            handle.abort();
        }

        self.failover_manager.stop().await;

        info!("Consensus manager stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consensus_manager() {
        let config = ConsensusConfig::default();
        let node_id = NodeId::from("test_node");
        let validators = vec![
            NodeId::from("node1"),
            NodeId::from("node2"),
            NodeId::from("node3"),
            NodeId::from("node4"),
        ]
        .into_iter()
        .collect();

        let manager = ConsensusManager::new(config, node_id, validators)
            .await
            .unwrap();

        // Start manager
        manager.start().await.unwrap();

        // Test leader election
        let leader = manager.force_leader_election().await.unwrap();
        assert!(!leader.to_string().is_empty());

        // Test state
        let state = manager.get_state().await;
        assert_eq!(state, ConsensusState::Normal);

        // Stop manager
        manager.stop().await;
    }

    #[tokio::test]
    async fn test_network_partition_recovery() {
        let config = ConsensusConfig::default();
        let node_id = NodeId::from("test_node");
        let validators = vec![
            NodeId::from("node1"),
            NodeId::from("node2"),
            NodeId::from("node3"),
            NodeId::from("node4"),
        ]
        .into_iter()
        .collect();

        let manager = ConsensusManager::new(config, node_id, validators)
            .await
            .unwrap();
        manager.start().await.unwrap();

        // Simulate partition
        let partitioned: HashSet<NodeId> = vec![NodeId::from("node1")].into_iter().collect();
        manager
            .simulate_network_partition(partitioned.clone())
            .await
            .unwrap();

        // Verify system still works
        let remaining_validators = manager.get_validators().await;
        assert_eq!(remaining_validators.len(), 3);

        // Heal partition
        manager.heal_network_partition(partitioned).await.unwrap();

        // Verify all nodes are back
        let all_validators = manager.get_validators().await;
        assert_eq!(all_validators.len(), 4);

        manager.stop().await;
    }
}
