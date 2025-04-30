use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use anyhow::Result;
use log::warn;
use libp2p::PeerId;

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub max_messages_per_second: usize,
    pub max_bytes_per_second: usize,
    pub max_connections: usize,
    pub ban_duration: Duration,
    pub warning_threshold: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_messages_per_second: 1000,
            max_bytes_per_second: 1024 * 1024 * 10, // 10MB/s
            max_connections: 100,
            ban_duration: Duration::from_secs(300), // 5 minutes
            warning_threshold: 0.8, // 80% of limit
        }
    }
}

/// Peer rate limiting state
#[derive(Debug, Clone)]
pub struct PeerRateState {
    /// Number of messages received
    pub msg_count: u64,
    /// Total message size
    pub total_size: u64,
    /// Rate violations count
    pub violations: u32,
    /// Last message timestamp
    pub last_msg_time: Instant,
    /// First message timestamp (for calculating averages)
    pub first_msg_time: Instant,
    /// Current ban status
    pub banned: bool,
    /// Ban expiration time if banned
    pub ban_until: Option<Instant>,
}

impl Default for PeerRateState {
    fn default() -> Self {
        Self {
            msg_count: 0,
            total_size: 0,
            violations: 0,
            last_msg_time: Instant::now(),
            first_msg_time: Instant::now(),
            banned: false,
            ban_until: None,
        }
    }
}

/// DoS protection manager
pub struct DosProtection {
    config: RateLimitConfig,
    peer_states: Arc<RwLock<HashMap<PeerId, PeerRateState>>>,
    banned_peers: Arc<RwLock<HashSet<PeerId>>>,
}

impl DosProtection {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            peer_states: Arc::new(RwLock::new(HashMap::new())),
            banned_peers: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Check if a peer is allowed to send a message
    pub async fn check_message_rate(&self, peer_id: &PeerId, message_size: usize) -> Result<bool> {
        // Check if peer is banned
        if self.is_banned(peer_id).await {
            return Ok(false);
        }

        let mut states = self.peer_states.write().await;
        let state = states.entry(*peer_id).or_insert_with(PeerRateState::default);

        // Reset counters if needed
        if state.last_msg_time.elapsed() >= Duration::from_secs(1) {
            state.msg_count = 0;
            state.total_size = 0;
            state.last_msg_time = Instant::now();
        }

        // Check message rate
        if state.msg_count >= self.config.max_messages_per_second as u64 {
            self.handle_rate_limit_exceeded(peer_id, state).await;
            return Ok(false);
        }

        // Check byte rate
        if state.total_size + message_size as u64 >= self.config.max_bytes_per_second as u64 {
            self.handle_rate_limit_exceeded(peer_id, state).await;
            return Ok(false);
        }

        // Update counters
        state.msg_count += 1;
        state.total_size += message_size as u64;

        // Check warning threshold
        if state.msg_count as f64 / self.config.max_messages_per_second as f64 >= self.config.warning_threshold {
            warn!("Peer {} approaching message rate limit", peer_id);
        }

        Ok(true)
    }

    /// Check if a peer is banned
    async fn is_banned(&self, peer_id: &PeerId) -> bool {
        let banned = self.banned_peers.read().await;
        if banned.contains(peer_id) {
            // Check if ban has expired
            if let Some(state) = self.peer_states.read().await.get(peer_id) {
                if let Some(ban_until) = state.ban_until {
                    if ban_until <= Instant::now() {
                        // Ban expired, remove from banned list
                        drop(banned);
                        let mut banned = self.banned_peers.write().await;
                        banned.remove(peer_id);
                        return false;
                    }
                }
            }
            return true;
        }
        false
    }

    /// Handle rate limit exceeded
    async fn handle_rate_limit_exceeded(&self, peer_id: &PeerId, state: &mut PeerRateState) {
        state.violations += 1;
        
        if state.violations >= 3 {
            // Ban peer
            state.ban_until = Some(Instant::now() + self.config.ban_duration);
            let mut banned = self.banned_peers.write().await;
            banned.insert(*peer_id);
            warn!("Peer {} banned for {} seconds", peer_id, self.config.ban_duration.as_secs());
        } else {
            warn!("Peer {} rate limit exceeded (warning {}/{})", peer_id, state.violations, 3);
        }
    }

    /// Check connection limit
    pub async fn check_connection_limit(&self, _peer_id: &PeerId) -> bool {
        let states = self.peer_states.read().await;
        states.len() < self.config.max_connections
    }

    /// Get peer statistics
    pub async fn get_peer_stats(&self, peer_id: &PeerId) -> Option<PeerRateState> {
        self.peer_states.read().await.get(peer_id).cloned()
    }

    /// Reset peer state
    pub async fn reset_peer_state(&self, peer_id: &PeerId) {
        let mut states = self.peer_states.write().await;
        states.remove(peer_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_rate_limiting() {
        let config = RateLimitConfig {
            max_messages_per_second: 10,
            max_bytes_per_second: 1000,
            max_connections: 5,
            ban_duration: Duration::from_secs(1),
            warning_threshold: 0.8,
        };

        let protection = DosProtection::new(config);
        let peer_id = PeerId::random();

        // Test message rate limiting
        for _ in 0..10 {
            assert!(protection.check_message_rate(&peer_id, 10).await.unwrap());
        }
        assert!(!protection.check_message_rate(&peer_id, 10).await.unwrap());

        // Test byte rate limiting
        let peer_id = PeerId::random();
        // 100 is well below the limit, so this should succeed
        assert!(protection.check_message_rate(&peer_id, 100).await.unwrap());
        
        // But a large message exceeding the remaining bytes should fail
        assert!(!protection.check_message_rate(&peer_id, 950).await.unwrap());

        // Test banning
        let peer_id = PeerId::random();
        for _ in 0..3 {
            assert!(!protection.check_message_rate(&peer_id, 1000).await.unwrap());
        }
        assert!(protection.is_banned(&peer_id).await);

        // Test ban expiration
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(!protection.is_banned(&peer_id).await);
    }
} 