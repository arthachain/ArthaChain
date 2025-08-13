use anyhow::{anyhow, Result};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Rate limiting configuration
    pub rate_limiting: RateLimitConfig,
    /// DoS protection configuration
    pub dos_protection: DoSProtectionConfig,
    /// RBAC configuration
    pub rbac: RBACConfig,
    /// Encryption configuration
    pub encryption: EncryptionConfig,
    /// Audit logging configuration
    pub audit: AuditConfig,
    /// Firewall rules
    pub firewall: FirewallConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Requests per second per IP
    pub requests_per_second: u32,
    /// Burst capacity
    pub burst_capacity: u32,
    /// Window size in seconds
    pub window_size_secs: u64,
    /// Rate limit timeout in seconds
    pub timeout_secs: u64,
    /// Whitelist of IPs to exclude from rate limiting
    pub whitelist: Vec<IpAddr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoSProtectionConfig {
    /// Enable DoS protection
    pub enabled: bool,
    /// Connection limit per IP
    pub max_connections_per_ip: u32,
    /// Request size limit in bytes
    pub max_request_size: usize,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Suspicious activity threshold
    pub suspicious_threshold: u32,
    /// Auto-ban duration in seconds
    pub auto_ban_duration_secs: u64,
    /// Enable DDoS detection
    pub ddos_detection: bool,
    /// DDoS threshold (requests per second)
    pub ddos_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RBACConfig {
    /// Enable RBAC
    pub enabled: bool,
    /// Default role for new users
    pub default_role: String,
    /// Admin users
    pub admin_users: Vec<String>,
    /// Role definitions
    pub roles: HashMap<String, Role>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Permissions
    pub permissions: Vec<Permission>,
    /// Inherit from other roles
    pub inherits: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Permission {
    /// Read blockchain data
    ReadBlockchain,
    /// Submit transactions
    SubmitTransactions,
    /// Manage consensus
    ManageConsensus,
    /// System administration
    SystemAdmin,
    /// Recovery operations
    RecoveryOperations,
    /// View monitoring data
    ViewMonitoring,
    /// Custom permission
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Enable encryption at rest
    pub encryption_at_rest: bool,
    /// Encryption algorithm
    pub algorithm: String,
    /// Key rotation interval in hours
    pub key_rotation_hours: u64,
    /// Enable TLS for all communications
    pub tls_enabled: bool,
    /// Minimum TLS version
    pub min_tls_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Log all API calls
    pub log_api_calls: bool,
    /// Log authentication attempts
    pub log_auth_attempts: bool,
    /// Log permission checks
    pub log_permission_checks: bool,
    /// Audit log retention days
    pub retention_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallConfig {
    /// Enable firewall
    pub enabled: bool,
    /// Allowed IP ranges
    pub allowed_ranges: Vec<String>,
    /// Blocked IP addresses
    pub blocked_ips: Vec<IpAddr>,
    /// Allowed ports
    pub allowed_ports: Vec<u16>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        let mut roles = HashMap::new();
        roles.insert("admin".to_string(), Role {
            name: "admin".to_string(),
            permissions: vec![
                Permission::ReadBlockchain,
                Permission::SubmitTransactions,
                Permission::ManageConsensus,
                Permission::SystemAdmin,
                Permission::RecoveryOperations,
                Permission::ViewMonitoring,
            ],
            inherits: vec![],
        });
        roles.insert("user".to_string(), Role {
            name: "user".to_string(),
            permissions: vec![
                Permission::ReadBlockchain,
                Permission::SubmitTransactions,
            ],
            inherits: vec![],
        });
        roles.insert("readonly".to_string(), Role {
            name: "readonly".to_string(),
            permissions: vec![
                Permission::ReadBlockchain,
                Permission::ViewMonitoring,
            ],
            inherits: vec![],
        });

        Self {
            rate_limiting: RateLimitConfig {
                enabled: true,
                requests_per_second: 100,
                burst_capacity: 200,
                window_size_secs: 60,
                timeout_secs: 300,
                whitelist: vec![],
            },
            dos_protection: DoSProtectionConfig {
                enabled: true,
                max_connections_per_ip: 50,
                max_request_size: 1024 * 1024, // 1MB
                request_timeout_secs: 30,
                suspicious_threshold: 1000,
                auto_ban_duration_secs: 3600, // 1 hour
                ddos_detection: true,
                ddos_threshold: 1000,
            },
            rbac: RBACConfig {
                enabled: true,
                default_role: "user".to_string(),
                admin_users: vec!["admin".to_string()],
                roles,
            },
            encryption: EncryptionConfig {
                encryption_at_rest: true,
                algorithm: "AES-256-GCM".to_string(),
                key_rotation_hours: 24,
                tls_enabled: true,
                min_tls_version: "1.3".to_string(),
            },
            audit: AuditConfig {
                enabled: true,
                log_api_calls: true,
                log_auth_attempts: true,
                log_permission_checks: false,
                retention_days: 30,
            },
            firewall: FirewallConfig {
                enabled: true,
                allowed_ranges: vec!["0.0.0.0/0".to_string()], // Allow all by default
                blocked_ips: vec![],
                allowed_ports: vec![8080, 8443, 9090], // API, HTTPS, Metrics
            },
        }
    }
}

/// Rate limiter for IP addresses
#[derive(Debug)]
pub struct RateLimiter {
    /// Request counts per IP
    request_counts: HashMap<IpAddr, VecDeque<Instant>>,
    /// Banned IPs
    banned_ips: HashMap<IpAddr, Instant>,
    /// Configuration
    config: RateLimitConfig,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            request_counts: HashMap::new(),
            banned_ips: HashMap::new(),
            config,
        }
    }

    /// Check if request is allowed
    pub fn is_allowed(&mut self, ip: IpAddr) -> bool {
        if !self.config.enabled {
            return true;
        }

        // Check whitelist
        if self.config.whitelist.contains(&ip) {
            return true;
        }

        // Check if IP is banned
        if let Some(ban_time) = self.banned_ips.get(&ip) {
            if ban_time.elapsed().as_secs() < self.config.timeout_secs {
                return false;
            } else {
                self.banned_ips.remove(&ip);
            }
        }

        let now = Instant::now();
        let window_duration = Duration::from_secs(self.config.window_size_secs);

        // Clean old entries and add new request
        let requests = self.request_counts.entry(ip).or_insert_with(VecDeque::new);
        
        // Remove old requests outside the window
        while let Some(&front_time) = requests.front() {
            if now.duration_since(front_time) > window_duration {
                requests.pop_front();
            } else {
                break;
            }
        }

        // Check if within limit
        if requests.len() >= self.config.requests_per_second as usize {
            // Ban the IP
            self.banned_ips.insert(ip, now);
            warn!("Rate limit exceeded for IP: {}, banning for {} seconds", ip, self.config.timeout_secs);
            return false;
        }

        // Add current request
        requests.push_back(now);
        true
    }

    /// Clean up old entries
    pub fn cleanup(&mut self) {
        let now = Instant::now();
        
        // Clean banned IPs
        self.banned_ips.retain(|_, ban_time| {
            ban_time.elapsed().as_secs() < self.config.timeout_secs
        });

        // Clean old request counts
        let window_duration = Duration::from_secs(self.config.window_size_secs);
        for requests in self.request_counts.values_mut() {
            while let Some(&front_time) = requests.front() {
                if now.duration_since(front_time) > window_duration {
                    requests.pop_front();
                } else {
                    break;
                }
            }
        }

        // Remove empty entries
        self.request_counts.retain(|_, requests| !requests.is_empty());
    }
}

/// DoS protection manager
#[derive(Debug)]
pub struct DoSProtection {
    /// Connection counts per IP
    connection_counts: HashMap<IpAddr, u32>,
    /// Suspicious activity tracker
    suspicious_activity: HashMap<IpAddr, (u32, Instant)>,
    /// Auto-banned IPs
    banned_ips: HashMap<IpAddr, Instant>,
    /// Configuration
    config: DoSProtectionConfig,
}

impl DoSProtection {
    pub fn new(config: DoSProtectionConfig) -> Self {
        Self {
            connection_counts: HashMap::new(),
            suspicious_activity: HashMap::new(),
            banned_ips: HashMap::new(),
            config,
        }
    }

    /// Check if connection is allowed
    pub fn is_connection_allowed(&mut self, ip: IpAddr) -> bool {
        if !self.config.enabled {
            return true;
        }

        // Check if banned
        if let Some(ban_time) = self.banned_ips.get(&ip) {
            if ban_time.elapsed().as_secs() < self.config.auto_ban_duration_secs {
                return false;
            } else {
                self.banned_ips.remove(&ip);
            }
        }

        // Check connection limit
        let current_connections = self.connection_counts.get(&ip).unwrap_or(&0);
        if *current_connections >= self.config.max_connections_per_ip {
            self.record_suspicious_activity(ip);
            return false;
        }

        true
    }

    /// Register new connection
    pub fn register_connection(&mut self, ip: IpAddr) {
        *self.connection_counts.entry(ip).or_insert(0) += 1;
    }

    /// Unregister connection
    pub fn unregister_connection(&mut self, ip: IpAddr) {
        if let Some(count) = self.connection_counts.get_mut(&ip) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.connection_counts.remove(&ip);
            }
        }
    }

    /// Record suspicious activity
    pub fn record_suspicious_activity(&mut self, ip: IpAddr) {
        let (count, _) = self.suspicious_activity
            .entry(ip)
            .or_insert((0, Instant::now()));
        
        *count += 1;
        
        if *count >= self.config.suspicious_threshold {
            self.banned_ips.insert(ip, Instant::now());
            warn!("Auto-banning IP {} due to suspicious activity", ip);
        }
    }

    /// Check request size
    pub fn is_request_size_allowed(&self, size: usize) -> bool {
        size <= self.config.max_request_size
    }

    /// Detect DDoS attack
    pub fn detect_ddos(&self, requests_per_second: u32) -> bool {
        self.config.ddos_detection && requests_per_second > self.config.ddos_threshold
    }
}

/// User session for RBAC
#[derive(Debug, Clone)]
pub struct UserSession {
    /// User ID
    pub user_id: String,
    /// User role
    pub role: String,
    /// Session creation time
    pub created_at: SystemTime,
    /// Last activity
    pub last_activity: SystemTime,
    /// Session token
    pub token: String,
}

/// RBAC manager
#[derive(Debug)]
pub struct RBACManager {
    /// User sessions
    sessions: HashMap<String, UserSession>,
    /// Configuration
    config: RBACConfig,
}

impl RBACManager {
    pub fn new(config: RBACConfig) -> Self {
        Self {
            sessions: HashMap::new(),
            config,
        }
    }

    /// Check if user has permission
    pub fn has_permission(&self, user_id: &str, permission: &Permission) -> bool {
        if !self.config.enabled {
            return true;
        }

        // Check if admin user
        if self.config.admin_users.contains(&user_id.to_string()) {
            return true;
        }

        // Get user role
        let user_role = self.get_user_role(user_id);
        self.role_has_permission(&user_role, permission)
    }

    /// Check if role has permission
    pub fn role_has_permission(&self, role_name: &str, permission: &Permission) -> bool {
        if let Some(role) = self.config.roles.get(role_name) {
            // Check direct permissions
            if role.permissions.contains(permission) {
                return true;
            }

            // Check inherited permissions
            for inherited_role in &role.inherits {
                if self.role_has_permission(inherited_role, permission) {
                    return true;
                }
            }
        }
        false
    }

    /// Get user role
    pub fn get_user_role(&self, user_id: &str) -> String {
        // In a real implementation, this would query a user database
        // For now, return default role
        self.config.default_role.clone()
    }

    /// Create user session
    pub fn create_session(&mut self, user_id: String, role: String) -> String {
        let token = uuid::Uuid::new_v4().to_string();
        let now = SystemTime::now();
        
        let session = UserSession {
            user_id: user_id.clone(),
            role,
            created_at: now,
            last_activity: now,
            token: token.clone(),
        };

        self.sessions.insert(token.clone(), session);
        token
    }

    /// Validate session
    pub fn validate_session(&mut self, token: &str) -> Option<&UserSession> {
        if let Some(session) = self.sessions.get_mut(token) {
            session.last_activity = SystemTime::now();
            Some(session)
        } else {
            None
        }
    }

    /// Remove session
    pub fn remove_session(&mut self, token: &str) {
        self.sessions.remove(token);
    }
}

/// Security manager
pub struct SecurityManager {
    /// Rate limiter
    rate_limiter: Arc<Mutex<RateLimiter>>,
    /// DoS protection
    dos_protection: Arc<Mutex<DoSProtection>>,
    /// RBAC manager
    rbac_manager: Arc<Mutex<RBACManager>>,
    /// Configuration
    config: SecurityConfig,
    /// Cleanup task handle
    cleanup_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl SecurityManager {
    /// Create new security manager
    pub fn new(config: SecurityConfig) -> Self {
        let rate_limiter = Arc::new(Mutex::new(RateLimiter::new(config.rate_limiting.clone())));
        let dos_protection = Arc::new(Mutex::new(DoSProtection::new(config.dos_protection.clone())));
        let rbac_manager = Arc::new(Mutex::new(RBACManager::new(config.rbac.clone())));

        Self {
            rate_limiter,
            dos_protection,
            rbac_manager,
            config,
            cleanup_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Start security manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting security manager");

        // Start cleanup task
        self.start_cleanup_task().await;

        info!("Security manager started");
        Ok(())
    }

    /// Start cleanup task
    async fn start_cleanup_task(&self) {
        let rate_limiter = self.rate_limiter.clone();
        let dos_protection = self.dos_protection.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Cleanup every minute

            loop {
                interval.tick().await;

                // Cleanup rate limiter
                rate_limiter.lock().await.cleanup();

                // Note: DoS protection cleanup would be implemented here
                // For now, we don't have a cleanup method for DoSProtection
            }
        });

        *self.cleanup_handle.lock().await = Some(handle);
    }

    /// Check if request is allowed
    pub async fn is_request_allowed(&self, ip: IpAddr, size: usize) -> bool {
        // Check rate limiting
        if !self.rate_limiter.lock().await.is_allowed(ip) {
            return false;
        }

        // Check DoS protection
        let dos_guard = self.dos_protection.lock().await;
        if !dos_guard.is_request_size_allowed(size) {
            return false;
        }

        true
    }

    /// Check if connection is allowed
    pub async fn is_connection_allowed(&self, ip: IpAddr) -> bool {
        self.dos_protection.lock().await.is_connection_allowed(ip)
    }

    /// Register new connection
    pub async fn register_connection(&self, ip: IpAddr) {
        self.dos_protection.lock().await.register_connection(ip);
    }

    /// Unregister connection
    pub async fn unregister_connection(&self, ip: IpAddr) {
        self.dos_protection.lock().await.unregister_connection(ip);
    }

    /// Check permission
    pub async fn check_permission(&self, user_id: &str, permission: Permission) -> bool {
        self.rbac_manager.lock().await.has_permission(user_id, &permission)
    }

    /// Create user session
    pub async fn create_session(&self, user_id: String, role: String) -> String {
        self.rbac_manager.lock().await.create_session(user_id, role)
    }

    /// Validate session
    pub async fn validate_session(&self, token: &str) -> Option<UserSession> {
        self.rbac_manager.lock().await.validate_session(token).cloned()
    }

    /// Remove session
    pub async fn remove_session(&self, token: &str) {
        self.rbac_manager.lock().await.remove_session(token);
    }

    /// Record suspicious activity
    pub async fn record_suspicious_activity(&self, ip: IpAddr) {
        self.dos_protection.lock().await.record_suspicious_activity(ip);
    }

    /// Get security configuration
    pub fn get_config(&self) -> &SecurityConfig {
        &self.config
    }

    /// Stop security manager
    pub async fn stop(&self) {
        if let Some(handle) = self.cleanup_handle.lock().await.take() {
            handle.abort();
        }
        info!("Security manager stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[tokio::test]
    async fn test_rate_limiter() {
        let config = RateLimitConfig {
            enabled: true,
            requests_per_second: 2,
            burst_capacity: 3,
            window_size_secs: 1,
            timeout_secs: 5,
            whitelist: vec![],
        };

        let mut limiter = RateLimiter::new(config);
        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));

        // First two requests should be allowed
        assert!(limiter.is_allowed(ip));
        assert!(limiter.is_allowed(ip));

        // Third request should be rejected (rate limit exceeded)
        assert!(!limiter.is_allowed(ip));
    }

    #[tokio::test]
    async fn test_rbac() {
        let config = SecurityConfig::default();
        let manager = SecurityManager::new(config);

        // Admin should have all permissions
        assert!(manager.check_permission("admin", Permission::SystemAdmin).await);
        assert!(manager.check_permission("admin", Permission::ReadBlockchain).await);

        // Regular user should not have admin permissions
        assert!(!manager.check_permission("user", Permission::SystemAdmin).await);
        assert!(manager.check_permission("user", Permission::ReadBlockchain).await);
    }

    #[tokio::test]
    async fn test_dos_protection() {
        let config = DoSProtectionConfig {
            enabled: true,
            max_connections_per_ip: 2,
            max_request_size: 1000,
            request_timeout_secs: 30,
            suspicious_threshold: 5,
            auto_ban_duration_secs: 60,
            ddos_detection: true,
            ddos_threshold: 100,
        };

        let mut protection = DoSProtection::new(config);
        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));

        // First connections should be allowed
        assert!(protection.is_connection_allowed(ip));
        protection.register_connection(ip);
        assert!(protection.is_connection_allowed(ip));
        protection.register_connection(ip);

        // Third connection should be rejected
        assert!(!protection.is_connection_allowed(ip));
    }
} 