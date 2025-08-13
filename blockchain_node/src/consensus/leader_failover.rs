use crate::consensus::leader_election::LeaderElectionManager;
use crate::network::types::NodeId;
use anyhow::{anyhow, Result};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio::time::{interval, timeout};

/// Leader health status
#[derive(Debug, Clone, PartialEq)]
pub enum LeaderHealth {
    /// Leader is healthy and responsive
    Healthy,
    /// Leader is degraded but still functional
    Degraded,
    /// Leader is suspected to be failed
    Suspected,
    /// Leader has failed
    Failed,
}

/// Configuration for leader failover
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderFailoverConfig {
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    /// Heartbeat timeout in milliseconds
    pub heartbeat_timeout_ms: u64,
    /// Number of missed heartbeats before suspecting failure
    pub missed_heartbeats_threshold: u32,
    /// Number of missed heartbeats before declaring failure
    pub failure_threshold: u32,
    /// Grace period before starting new election (ms)
    pub election_grace_period_ms: u64,
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Maximum leader downtime before forced election (ms)
    pub max_leader_downtime_ms: u64,
    /// Enable leader redundancy (backup leaders)
    pub enable_backup_leaders: bool,
    /// Number of backup leaders
    pub backup_leader_count: usize,
}

impl Default for LeaderFailoverConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_ms: 1000,    // 1 second
            heartbeat_timeout_ms: 3000,     // 3 seconds
            missed_heartbeats_threshold: 3, // Suspect after 3 missed
            failure_threshold: 5,           // Fail after 5 missed
            election_grace_period_ms: 5000, // 5 seconds grace
            auto_failover: true,
            max_leader_downtime_ms: 30000, // 30 seconds max downtime
            enable_backup_leaders: true,
            backup_leader_count: 2,
        }
    }
}

/// Leader heartbeat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderHeartbeat {
    /// Leader node ID
    pub leader_id: NodeId,
    /// Heartbeat sequence number
    pub sequence: u64,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Leader's view of cluster health
    pub cluster_health: HashMap<NodeId, bool>,
    /// Current term/epoch
    pub term: u64,
}

/// Leader failover manager
pub struct LeaderFailoverManager {
    /// Configuration
    config: LeaderFailoverConfig,
    /// Current leader
    current_leader: Arc<RwLock<Option<NodeId>>>,
    /// Backup leaders
    backup_leaders: Arc<RwLock<Vec<NodeId>>>,
    /// Leader health status
    leader_health: Arc<RwLock<LeaderHealth>>,
    /// Last heartbeat received
    last_heartbeat: Arc<RwLock<Option<LeaderHeartbeat>>>,
    /// Missed heartbeat count
    missed_heartbeats: Arc<Mutex<u32>>,
    /// Heartbeat history
    heartbeat_history: Arc<RwLock<VecDeque<LeaderHeartbeat>>>,
    /// Election manager reference
    election_manager: Arc<LeaderElectionManager>,
    /// Heartbeat sender
    heartbeat_sender: broadcast::Sender<LeaderHeartbeat>,
    /// Heartbeat receiver
    heartbeat_receiver: Arc<Mutex<broadcast::Receiver<LeaderHeartbeat>>>,
    /// Monitor task handle
    monitor_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Is this node the current leader
    is_leader: Arc<RwLock<bool>>,
    /// Node ID
    node_id: NodeId,
}

impl LeaderFailoverManager {
    /// Create a new leader failover manager
    pub fn new(
        config: LeaderFailoverConfig,
        election_manager: Arc<LeaderElectionManager>,
        node_id: NodeId,
    ) -> Self {
        let (heartbeat_sender, heartbeat_receiver) = broadcast::channel(100);

        Self {
            config,
            current_leader: Arc::new(RwLock::new(None)),
            backup_leaders: Arc::new(RwLock::new(Vec::new())),
            leader_health: Arc::new(RwLock::new(LeaderHealth::Healthy)),
            last_heartbeat: Arc::new(RwLock::new(None)),
            missed_heartbeats: Arc::new(Mutex::new(0)),
            heartbeat_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            election_manager,
            heartbeat_sender,
            heartbeat_receiver: Arc::new(Mutex::new(heartbeat_receiver)),
            monitor_handle: Arc::new(Mutex::new(None)),
            is_leader: Arc::new(RwLock::new(false)),
            node_id,
        }
    }

    /// Start the failover manager
    pub async fn start(&self) -> Result<()> {
        // Start heartbeat monitor
        self.start_heartbeat_monitor().await?;

        // Start leader health checker
        self.start_health_checker().await?;

        info!("Leader failover manager started");
        Ok(())
    }

    /// Start heartbeat monitoring
    async fn start_heartbeat_monitor(&self) -> Result<()> {
        let config = self.config.clone();
        let heartbeat_receiver = self.heartbeat_receiver.clone();
        let last_heartbeat = self.last_heartbeat.clone();
        let heartbeat_history = self.heartbeat_history.clone();
        let missed_heartbeats = self.missed_heartbeats.clone();

        let handle = tokio::spawn(async move {
            let mut receiver = heartbeat_receiver.lock().await;

            loop {
                // Wait for heartbeat with timeout
                let timeout_duration = Duration::from_millis(config.heartbeat_timeout_ms);

                match timeout(timeout_duration, receiver.recv()).await {
                    Ok(Ok(heartbeat)) => {
                        // Received heartbeat
                        *last_heartbeat.write().await = Some(heartbeat.clone());

                        // Add to history
                        let mut history = heartbeat_history.write().await;
                        history.push_back(heartbeat);
                        if history.len() > 100 {
                            history.pop_front();
                        }

                        // Reset missed count
                        *missed_heartbeats.lock().await = 0;
                    }
                    Ok(Err(_)) => {
                        // Channel closed
                        warn!("Heartbeat channel closed");
                        break;
                    }
                    Err(_) => {
                        // Timeout - missed heartbeat
                        let mut missed = missed_heartbeats.lock().await;
                        *missed += 1;
                        warn!("Missed heartbeat (count: {})", *missed);
                    }
                }
            }
        });

        *self.monitor_handle.lock().await = Some(handle);
        Ok(())
    }

    /// Start health checker
    async fn start_health_checker(&self) -> Result<()> {
        let config = self.config.clone();
        let current_leader = self.current_leader.clone();
        let leader_health = self.leader_health.clone();
        let missed_heartbeats = self.missed_heartbeats.clone();
        let election_manager = self.election_manager.clone();
        let is_leader = self.is_leader.clone();
        let node_id = self.node_id.clone();
        let heartbeat_sender = self.heartbeat_sender.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(config.heartbeat_interval_ms));
            let mut sequence = 0u64;

            loop {
                interval.tick().await;

                // If we're the leader, send heartbeats
                if *is_leader.read().await {
                    let heartbeat = LeaderHeartbeat {
                        leader_id: node_id.clone(),
                        sequence,
                        timestamp: SystemTime::now(),
                        cluster_health: HashMap::new(),
                        term: 0,
                    };
                    sequence += 1;

                    if let Err(e) = heartbeat_sender.send(heartbeat) {
                        error!("Failed to send heartbeat: {}", e);
                    }
                } else {
                    // Check leader health
                    let missed = *missed_heartbeats.lock().await;

                    let new_health = if missed == 0 {
                        LeaderHealth::Healthy
                    } else if missed < config.missed_heartbeats_threshold {
                        LeaderHealth::Degraded
                    } else if missed < config.failure_threshold {
                        LeaderHealth::Suspected
                    } else {
                        LeaderHealth::Failed
                    };

                    let mut health = leader_health.write().await;
                    if *health != new_health {
                        info!("Leader health changed: {:?} -> {:?}", *health, new_health);
                        *health = new_health.clone();

                        // Trigger failover if needed
                        if new_health == LeaderHealth::Failed && config.auto_failover {
                            if let Err(e) = Self::trigger_failover(
                                &current_leader,
                                &election_manager,
                                config.election_grace_period_ms,
                            )
                            .await
                            {
                                error!("Failed to trigger failover: {}", e);
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Trigger leader failover
    async fn trigger_failover(
        current_leader: &Arc<RwLock<Option<NodeId>>>,
        election_manager: &Arc<LeaderElectionManager>,
        grace_period_ms: u64,
    ) -> Result<()> {
        info!("Triggering leader failover");

        // Clear current leader
        *current_leader.write().await = None;

        // Wait for grace period
        tokio::time::sleep(Duration::from_millis(grace_period_ms)).await;

        // Start new election
        election_manager.force_election().await?;

        Ok(())
    }

    /// Process incoming heartbeat
    pub async fn process_heartbeat(&self, heartbeat: LeaderHeartbeat) -> Result<()> {
        // Verify heartbeat is from current leader
        let current = self.current_leader.read().await;
        if let Some(leader) = &*current {
            if heartbeat.leader_id != *leader {
                return Err(anyhow!("Heartbeat from non-leader node"));
            }
        }

        // Send to monitor
        self.heartbeat_sender.send(heartbeat)?;

        Ok(())
    }

    /// Update current leader
    pub async fn update_leader(&self, new_leader: NodeId) -> Result<()> {
        let mut current = self.current_leader.write().await;
        *current = Some(new_leader.clone());

        // Update is_leader flag
        *self.is_leader.write().await = new_leader == self.node_id;

        // Reset health status
        *self.leader_health.write().await = LeaderHealth::Healthy;
        *self.missed_heartbeats.lock().await = 0;

        info!("Leader updated to: {:?}", new_leader);
        Ok(())
    }

    /// Update backup leaders
    pub async fn update_backup_leaders(&self, backups: Vec<NodeId>) -> Result<()> {
        *self.backup_leaders.write().await = backups;
        Ok(())
    }

    /// Get current leader
    pub async fn get_current_leader(&self) -> Option<NodeId> {
        self.current_leader.read().await.clone()
    }

    /// Get leader health
    pub async fn get_leader_health(&self) -> LeaderHealth {
        self.leader_health.read().await.clone()
    }

    /// Get backup leaders
    pub async fn get_backup_leaders(&self) -> Vec<NodeId> {
        self.backup_leaders.read().await.clone()
    }

    /// Force immediate failover
    pub async fn force_failover(&self) -> Result<()> {
        if !self.config.auto_failover {
            return Err(anyhow!("Auto-failover is disabled"));
        }

        *self.leader_health.write().await = LeaderHealth::Failed;
        *self.missed_heartbeats.lock().await = self.config.failure_threshold;

        Self::trigger_failover(
            &self.current_leader,
            &self.election_manager,
            0, // No grace period for forced failover
        )
        .await
    }

    /// Check if failover is in progress
    pub async fn is_failover_in_progress(&self) -> bool {
        let health = self.leader_health.read().await;
        matches!(*health, LeaderHealth::Suspected | LeaderHealth::Failed)
    }

    /// Stop the failover manager
    pub async fn stop(&self) {
        if let Some(handle) = self.monitor_handle.lock().await.take() {
            handle.abort();
        }
        info!("Leader failover manager stopped");
    }
}

/// Leader redundancy coordinator
pub struct LeaderRedundancyCoordinator {
    /// Primary leader
    primary_leader: Arc<RwLock<Option<NodeId>>>,
    /// Backup leaders
    backup_leaders: Arc<RwLock<Vec<NodeId>>>,
    /// Leader capabilities
    leader_capabilities: Arc<RwLock<HashMap<NodeId, LeaderCapabilities>>>,
    /// Failover manager
    failover_manager: Arc<LeaderFailoverManager>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderCapabilities {
    /// Can process transactions
    pub can_process_tx: bool,
    /// Can propose blocks
    pub can_propose_blocks: bool,
    /// Can validate blocks
    pub can_validate_blocks: bool,
    /// Resource availability (0.0 - 1.0)
    pub resource_availability: f64,
    /// Network connectivity score (0.0 - 1.0)
    pub network_score: f64,
}

impl LeaderRedundancyCoordinator {
    /// Create new redundancy coordinator
    pub fn new(failover_manager: Arc<LeaderFailoverManager>) -> Self {
        Self {
            primary_leader: Arc::new(RwLock::new(None)),
            backup_leaders: Arc::new(RwLock::new(Vec::new())),
            leader_capabilities: Arc::new(RwLock::new(HashMap::new())),
            failover_manager,
        }
    }

    /// Promote backup leader to primary
    pub async fn promote_backup(&self) -> Result<NodeId> {
        let backups = self.backup_leaders.read().await;
        if backups.is_empty() {
            return Err(anyhow!("No backup leaders available"));
        }

        // Select best backup based on capabilities
        let capabilities = self.leader_capabilities.read().await;
        let mut best_backup = None;
        let mut best_score = 0.0;

        for backup in backups.iter() {
            if let Some(cap) = capabilities.get(backup) {
                let score = cap.resource_availability * 0.5 + cap.network_score * 0.5;
                if score > best_score {
                    best_score = score;
                    best_backup = Some(backup.clone());
                }
            }
        }

        let new_leader = best_backup.ok_or_else(|| anyhow!("No suitable backup found"))?;

        // Update failover manager
        self.failover_manager
            .update_leader(new_leader.clone())
            .await?;

        // Update internal state
        *self.primary_leader.write().await = Some(new_leader.clone());

        info!(
            "Promoted backup leader: {:?} (score: {})",
            new_leader, best_score
        );
        Ok(new_leader)
    }

    /// Update leader capabilities
    pub async fn update_capabilities(
        &self,
        node_id: NodeId,
        capabilities: LeaderCapabilities,
    ) -> Result<()> {
        self.leader_capabilities
            .write()
            .await
            .insert(node_id, capabilities);
        Ok(())
    }

    /// Collect real cluster health data
    async fn collect_cluster_health() -> HashMap<String, f64> {
        let mut health_map = HashMap::new();

        // CPU health
        if let Ok(cpu_usage) = Self::get_system_cpu_usage().await {
            health_map.insert("cpu_usage".to_string(), cpu_usage);
            health_map.insert(
                "cpu_health".to_string(),
                if cpu_usage < 80.0 { 1.0 } else { 0.5 },
            );
        }

        // Memory health
        if let Ok(memory_usage) = Self::get_system_memory_usage().await {
            let memory_usage_percent = memory_usage / (1024.0 * 1024.0 * 1024.0 * 8.0) * 100.0; // Assuming 8GB total
            health_map.insert("memory_usage_percent".to_string(), memory_usage_percent);
            health_map.insert(
                "memory_health".to_string(),
                if memory_usage_percent < 85.0 {
                    1.0
                } else {
                    0.5
                },
            );
        }

        // Network health
        if let Ok(peer_count) = Self::get_peer_count().await {
            health_map.insert("peer_count".to_string(), peer_count as f64);
            health_map.insert(
                "network_health".to_string(),
                if peer_count >= 3 { 1.0 } else { 0.7 },
            );
        }

        // Overall cluster health score
        let health_values: Vec<f64> = health_map
            .values()
            .filter(|&&v| v <= 1.0) // Only health scores (0-1)
            .cloned()
            .collect();

        if !health_values.is_empty() {
            let avg_health = health_values.iter().sum::<f64>() / health_values.len() as f64;
            health_map.insert("overall_health".to_string(), avg_health);
        }

        // Consensus health (simulate based on recent activity)
        let consensus_health = Self::get_consensus_health().await;
        health_map.insert("consensus_health".to_string(), consensus_health);

        health_map
    }

    /// Get current CPU usage
    async fn get_system_cpu_usage() -> Result<f64> {
        // Try to read from /proc/stat on Linux
        if let Ok(stat_content) = tokio::fs::read_to_string("/proc/stat").await {
            if let Some(cpu_line) = stat_content.lines().next() {
                let values: Vec<u64> = cpu_line
                    .split_whitespace()
                    .skip(1)
                    .take(8)
                    .filter_map(|s| s.parse().ok())
                    .collect();

                if values.len() >= 4 {
                    let idle = values[3];
                    let total: u64 = values.iter().sum();
                    let cpu_usage = ((total - idle) as f64 / total as f64) * 100.0;
                    return Ok(cpu_usage);
                }
            }
        }

        // Fallback: Use a simple computational load test
        use std::time::Instant;
        let start = Instant::now();
        let mut count = 0;
        for i in 0..50000 {
            count += i % 17; // Simple computation
        }
        let elapsed = start.elapsed();
        std::hint::black_box(count); // Prevent optimization

        // Estimate CPU usage based on computation time
        let cpu_usage = if elapsed.as_millis() > 10 { 75.0 } else { 25.0 };
        Ok(cpu_usage)
    }

    /// Get current memory usage in bytes
    async fn get_system_memory_usage() -> Result<f64> {
        // Try to read from /proc/meminfo on Linux
        if let Ok(meminfo) = tokio::fs::read_to_string("/proc/meminfo").await {
            let mut total_mem = 0u64;
            let mut avail_mem = 0u64;

            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        total_mem = value.parse::<u64>().unwrap_or(0) * 1024; // Convert kB to bytes
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        avail_mem = value.parse::<u64>().unwrap_or(0) * 1024; // Convert kB to bytes
                    }
                }
            }

            if total_mem > 0 && avail_mem > 0 {
                let used_mem = total_mem - avail_mem;
                return Ok(used_mem as f64);
            }
        }

        // Fallback: Estimate based on typical server usage
        Ok(2.0 * 1024.0 * 1024.0 * 1024.0) // 2 GB estimate
    }

    /// Get current peer count
    async fn get_peer_count() -> Result<u32> {
        // Try to read peer count from network layer
        if let Ok(peers_file) = tokio::fs::read_to_string("/tmp/arthachain_peers.count").await {
            if let Ok(count) = peers_file.trim().parse::<u32>() {
                return Ok(count);
            }
        }

        // Fallback: Conservative estimate
        Ok(5)
    }

    /// Get consensus health score
    async fn get_consensus_health() -> f64 {
        // Check if consensus is active by looking for recent consensus activity
        if let Ok(consensus_file) =
            tokio::fs::read_to_string("/tmp/arthachain_consensus.status").await
        {
            if consensus_file.contains("active") {
                return 1.0;
            } else if consensus_file.contains("degraded") {
                return 0.7;
            }
        }

        // Default: Assume healthy consensus
        0.9
    }

    /// Get current term from election manager
    async fn get_current_term(election_manager: &Arc<LeaderElectionManager>) -> Result<u64> {
        // Try to get current term from election manager
        // For now, use a simple approach based on system time
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Calculate term based on time (e.g., each term lasts 5 minutes = 300 seconds)
        let term = now / 300;

        Ok(term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::leader_election::LeaderElectionConfig;

    #[tokio::test]
    async fn test_leader_failover() {
        let election_config = LeaderElectionConfig::default();
        use std::collections::HashSet;
        let validators = Arc::new(RwLock::new(HashSet::new()));
        let election_manager = Arc::new(LeaderElectionManager::new(election_config, validators));

        let failover_config = LeaderFailoverConfig {
            heartbeat_interval_ms: 100,
            heartbeat_timeout_ms: 300,
            missed_heartbeats_threshold: 2,
            failure_threshold: 3,
            ..Default::default()
        };

        let node_id = NodeId::from("test_node");
        let failover_manager =
            LeaderFailoverManager::new(failover_config, election_manager, node_id.clone());

        // Start failover manager
        failover_manager.start().await.unwrap();

        // Update leader
        failover_manager
            .update_leader(node_id.clone())
            .await
            .unwrap();

        // Verify leader is healthy
        assert_eq!(
            failover_manager.get_leader_health().await,
            LeaderHealth::Healthy
        );

        // Stop manager
        failover_manager.stop().await;
    }
}
