use dashmap::DashMap;
use std::time::{Duration, Instant};
use teloxide::types::UserId;

/// Rate limiter for faucet requests
pub struct RateLimiter {
    /// Track last request time for each user
    user_requests: DashMap<UserId, Instant>,
    /// Cooldown period between requests
    cooldown_duration: Duration,
}

impl RateLimiter {
    /// Create a new rate limiter with default 24-hour cooldown
    pub fn new() -> Self {
        Self::with_cooldown(Duration::from_secs(24 * 60 * 60)) // 24 hours
    }

    /// Create a new rate limiter with custom cooldown
    pub fn with_cooldown(cooldown_duration: Duration) -> Self {
        Self {
            user_requests: DashMap::new(),
            cooldown_duration,
        }
    }

    /// Check if a user is allowed to make a request
    pub async fn allow_request(&self, user_id: UserId) -> bool {
        let now = Instant::now();
        
        if let Some(last_request) = self.user_requests.get(&user_id) {
            let time_since_last = now.duration_since(*last_request);
            if time_since_last < self.cooldown_duration {
                return false;
            }
        }

        // Update last request time
        self.user_requests.insert(user_id, now);
        true
    }

    /// Get the duration until next allowed request
    pub async fn next_allowed_time(&self, user_id: UserId) -> Duration {
        if let Some(last_request) = self.user_requests.get(&user_id) {
            let time_since_last = Instant::now().duration_since(*last_request);
            if time_since_last < self.cooldown_duration {
                return self.cooldown_duration - time_since_last;
            }
        }
        Duration::from_secs(0)
    }

    /// Get the cooldown duration
    pub fn cooldown_duration(&self) -> Duration {
        self.cooldown_duration
    }

    /// Reset rate limit for a user (admin function)
    pub async fn reset_user(&self, user_id: UserId) {
        self.user_requests.remove(&user_id);
    }

    /// Get number of users in rate limit tracking
    pub fn tracked_users_count(&self) -> usize {
        self.user_requests.len()
    }

    /// Clean up old entries (call periodically)
    pub async fn cleanup_old_entries(&self) {
        let now = Instant::now();
        let cutoff = self.cooldown_duration + Duration::from_secs(3600); // Keep extra hour for safety
        
        self.user_requests.retain(|_, &mut last_request| {
            now.duration_since(last_request) < cutoff
        });
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::with_cooldown(Duration::from_millis(100));
        let user_id = UserId(12345);

        // First request should be allowed
        assert!(limiter.allow_request(user_id).await);

        // Second request immediately should be denied
        assert!(!limiter.allow_request(user_id).await);

        // Wait for cooldown and try again
        sleep(TokioDuration::from_millis(150)).await;
        assert!(limiter.allow_request(user_id).await);
    }

    #[tokio::test]
    async fn test_next_allowed_time() {
        let limiter = RateLimiter::with_cooldown(Duration::from_secs(60));
        let user_id = UserId(67890);

        // No previous request
        assert_eq!(limiter.next_allowed_time(user_id).await, Duration::from_secs(0));

        // Make a request
        assert!(limiter.allow_request(user_id).await);

        // Check remaining time
        let remaining = limiter.next_allowed_time(user_id).await;
        assert!(remaining.as_secs() > 50 && remaining.as_secs() <= 60);
    }
}