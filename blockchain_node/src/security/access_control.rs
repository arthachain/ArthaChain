use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Role-based access control system
#[derive(Debug, Clone)]
pub struct AccessControlManager {
    /// User sessions
    sessions: Arc<RwLock<HashMap<String, UserSession>>>,
    /// User roles and permissions
    roles: Arc<RwLock<HashMap<String, Role>>>,
    /// Permission definitions
    permissions: Arc<RwLock<HashMap<String, Permission>>>,
    /// Active API keys
    api_keys: Arc<RwLock<HashMap<String, ApiKey>>>,
    /// Security policies
    policies: SecurityPolicies,
}

/// User session with authentication token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSession {
    /// Session ID
    pub session_id: String,
    /// User ID
    pub user_id: String,
    /// Authentication token
    pub token: String,
    /// User roles
    pub roles: HashSet<String>,
    /// Session creation time
    pub created_at: u64,
    /// Session expiry time
    pub expires_at: u64,
    /// Last activity timestamp
    pub last_activity: u64,
    /// Source IP address
    pub source_ip: String,
    /// Multi-factor authentication status
    pub mfa_verified: bool,
}

/// Role definition with permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Role description
    pub description: String,
    /// Associated permissions
    pub permissions: HashSet<String>,
    /// Role hierarchy level (higher = more privileged)
    pub level: u32,
    /// Role creation time
    pub created_at: u64,
    /// Role is active
    pub active: bool,
}

/// Permission definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    /// Permission name
    pub name: String,
    /// Permission description
    pub description: String,
    /// Resource type this permission applies to
    pub resource_type: ResourceType,
    /// Allowed operations
    pub operations: HashSet<Operation>,
    /// Permission scope
    pub scope: PermissionScope,
}

/// API key for programmatic access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// API key ID
    pub key_id: String,
    /// Hashed key value
    pub key_hash: String,
    /// Associated user ID
    pub user_id: String,
    /// Key name/description
    pub name: String,
    /// Associated roles
    pub roles: HashSet<String>,
    /// Key creation time
    pub created_at: u64,
    /// Key expiry time (optional)
    pub expires_at: Option<u64>,
    /// Last used timestamp
    pub last_used: Option<u64>,
    /// Rate limit per hour
    pub rate_limit: u32,
    /// Current usage count
    pub usage_count: u32,
    /// Key is active
    pub active: bool,
}

/// Resource types in the blockchain system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// Storage operations
    Storage,
    /// Transaction operations
    Transaction,
    /// Block operations
    Block,
    /// Network operations
    Network,
    /// Configuration operations
    Config,
    /// AI/ML operations
    AI,
    /// Consensus operations
    Consensus,
    /// Monitoring/metrics
    Metrics,
    /// User management
    Users,
    /// API access
    API,
    /// BCI neural data
    BCI,
    /// Shard operations
    Shard,
}

/// Available operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Read operation
    Read,
    /// Write operation
    Write,
    /// Delete operation
    Delete,
    /// Execute operation
    Execute,
    /// Admin operation
    Admin,
    /// Create operation
    Create,
    /// Update operation
    Update,
    /// Query operation
    Query,
}

/// Permission scope definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PermissionScope {
    /// Global access
    Global,
    /// User-specific access
    User(String),
    /// Shard-specific access
    Shard(u64),
    /// Resource-specific access
    Resource(String),
    /// Custom scope
    Custom(HashMap<String, String>),
}

/// Security policies configuration
#[derive(Debug, Clone)]
pub struct SecurityPolicies {
    /// Session timeout in seconds
    pub session_timeout: u64,
    /// Maximum concurrent sessions per user
    pub max_sessions_per_user: u32,
    /// Require MFA for admin operations
    pub require_mfa_admin: bool,
    /// API key rate limit per hour
    pub api_key_rate_limit: u32,
    /// Password complexity requirements
    pub password_policy: PasswordPolicy,
    /// IP allowlist for admin operations
    pub admin_ip_allowlist: Option<HashSet<String>>,
}

/// Password policy configuration
#[derive(Debug, Clone)]
pub struct PasswordPolicy {
    /// Minimum password length
    pub min_length: usize,
    /// Require uppercase letters
    pub require_uppercase: bool,
    /// Require lowercase letters
    pub require_lowercase: bool,
    /// Require numbers
    pub require_numbers: bool,
    /// Require special characters
    pub require_special: bool,
    /// Prevent password reuse (last N passwords)
    pub prevent_reuse: usize,
}

/// Authentication request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthRequest {
    /// Authentication token or API key
    pub token: String,
    /// Requested resource
    pub resource: ResourceType,
    /// Requested operation
    pub operation: Operation,
    /// Resource identifier (optional)
    pub resource_id: Option<String>,
    /// Request source IP
    pub source_ip: String,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Authentication result
#[derive(Debug, Clone)]
pub struct AuthResult {
    /// Authentication successful
    pub authorized: bool,
    /// User session (if authenticated)
    pub session: Option<UserSession>,
    /// Reason for denial (if not authorized)
    pub reason: Option<String>,
    /// Required actions (e.g., MFA)
    pub required_actions: Vec<String>,
}

impl AccessControlManager {
    /// Create new access control manager
    pub fn new() -> Self {
        let mut manager = Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            roles: Arc::new(RwLock::new(HashMap::new())),
            permissions: Arc::new(RwLock::new(HashMap::new())),
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            policies: SecurityPolicies::default(),
        };

        // Initialize default roles and permissions in the background
        let manager_clone = AccessControlManager {
            sessions: manager.sessions.clone(),
            roles: manager.roles.clone(),
            permissions: manager.permissions.clone(),
            api_keys: manager.api_keys.clone(),
            policies: manager.policies.clone(),
        };

        tokio::spawn(async move {
            if let Err(e) = manager_clone.initialize_default_roles().await {
                eprintln!("Failed to initialize default roles: {}", e);
            }
        });

        manager
    }

    /// Initialize default roles and permissions
    async fn initialize_default_roles(&self) -> Result<()> {
        // Define default permissions
        let permissions = vec![
            Permission {
                name: "storage.read".to_string(),
                description: "Read from storage".to_string(),
                resource_type: ResourceType::Storage,
                operations: HashSet::from([Operation::Read]),
                scope: PermissionScope::Global,
            },
            Permission {
                name: "storage.write".to_string(),
                description: "Write to storage".to_string(),
                resource_type: ResourceType::Storage,
                operations: HashSet::from([Operation::Write, Operation::Create, Operation::Update]),
                scope: PermissionScope::Global,
            },
            Permission {
                name: "storage.admin".to_string(),
                description: "Full storage administration".to_string(),
                resource_type: ResourceType::Storage,
                operations: HashSet::from([
                    Operation::Read,
                    Operation::Write,
                    Operation::Delete,
                    Operation::Admin,
                ]),
                scope: PermissionScope::Global,
            },
            Permission {
                name: "transaction.read".to_string(),
                description: "Read transactions".to_string(),
                resource_type: ResourceType::Transaction,
                operations: HashSet::from([Operation::Read, Operation::Query]),
                scope: PermissionScope::Global,
            },
            Permission {
                name: "transaction.create".to_string(),
                description: "Create transactions".to_string(),
                resource_type: ResourceType::Transaction,
                operations: HashSet::from([Operation::Create, Operation::Write]),
                scope: PermissionScope::Global,
            },
            Permission {
                name: "bci.read".to_string(),
                description: "Read BCI neural data".to_string(),
                resource_type: ResourceType::BCI,
                operations: HashSet::from([Operation::Read]),
                scope: PermissionScope::User("self".to_string()),
            },
            Permission {
                name: "bci.process".to_string(),
                description: "Process BCI neural data".to_string(),
                resource_type: ResourceType::BCI,
                operations: HashSet::from([Operation::Execute, Operation::Write]),
                scope: PermissionScope::User("self".to_string()),
            },
            Permission {
                name: "consensus.participate".to_string(),
                description: "Participate in consensus".to_string(),
                resource_type: ResourceType::Consensus,
                operations: HashSet::from([Operation::Execute, Operation::Write]),
                scope: PermissionScope::Global,
            },
            Permission {
                name: "admin.all".to_string(),
                description: "Full system administration".to_string(),
                resource_type: ResourceType::Config,
                operations: HashSet::from([
                    Operation::Admin,
                    Operation::Read,
                    Operation::Write,
                    Operation::Delete,
                ]),
                scope: PermissionScope::Global,
            },
        ];

        // Store permissions
        let mut perms = self.permissions.write().await;
        for permission in permissions {
            perms.insert(permission.name.clone(), permission);
        }
        drop(perms);

        // Define default roles
        let roles = vec![
            Role {
                name: "user".to_string(),
                description: "Basic user role".to_string(),
                permissions: HashSet::from([
                    "storage.read".to_string(),
                    "transaction.read".to_string(),
                    "transaction.create".to_string(),
                    "bci.read".to_string(),
                    "bci.process".to_string(),
                ]),
                level: 1,
                created_at: current_timestamp(),
                active: true,
            },
            Role {
                name: "validator".to_string(),
                description: "Blockchain validator role".to_string(),
                permissions: HashSet::from([
                    "storage.read".to_string(),
                    "storage.write".to_string(),
                    "transaction.read".to_string(),
                    "transaction.create".to_string(),
                    "consensus.participate".to_string(),
                ]),
                level: 5,
                created_at: current_timestamp(),
                active: true,
            },
            Role {
                name: "admin".to_string(),
                description: "System administrator role".to_string(),
                permissions: HashSet::from([
                    "storage.admin".to_string(),
                    "transaction.read".to_string(),
                    "transaction.create".to_string(),
                    "consensus.participate".to_string(),
                    "admin.all".to_string(),
                ]),
                level: 10,
                created_at: current_timestamp(),
                active: true,
            },
        ];

        // Store roles
        let mut role_map = self.roles.write().await;
        for role in roles {
            role_map.insert(role.name.clone(), role);
        }

        Ok(())
    }

    /// Authenticate a request
    pub async fn authenticate(&self, request: &AuthRequest) -> Result<AuthResult> {
        // Try session-based authentication first
        if let Some(session) = self.validate_session(&request.token).await? {
            return self.authorize_session(&session, request).await;
        }

        // Try API key authentication
        if let Some(api_key) = self.validate_api_key(&request.token).await? {
            return self.authorize_api_key(&api_key, request).await;
        }

        Ok(AuthResult {
            authorized: false,
            session: None,
            reason: Some("Invalid authentication token".to_string()),
            required_actions: vec![],
        })
    }

    /// Validate session token
    async fn validate_session(&self, token: &str) -> Result<Option<UserSession>> {
        let sessions = self.sessions.read().await;

        // Find matching session and clone it
        let mut found_session = None;
        for session in sessions.values() {
            if session.token == token {
                let now = current_timestamp();

                // Check expiry
                if session.expires_at < now {
                    return Ok(None);
                }

                // Check session timeout
                if now - session.last_activity > self.policies.session_timeout {
                    return Ok(None);
                }

                found_session = Some((session.session_id.clone(), session.clone()));
                break;
            }
        }

        // Update last activity if session found
        if let Some((session_id, mut session)) = found_session {
            drop(sessions);
            let mut sessions_mut = self.sessions.write().await;
            if let Some(mut_session) = sessions_mut.get_mut(&session_id) {
                mut_session.last_activity = current_timestamp();
                session.last_activity = current_timestamp();
            }
            return Ok(Some(session));
        }

        Ok(None)
    }

    /// Validate API key
    async fn validate_api_key(&self, key: &str) -> Result<Option<ApiKey>> {
        let key_hash = hash_string(key);
        let api_keys = self.api_keys.read().await;

        // Find matching API key and clone it
        let mut found_key = None;
        for api_key in api_keys.values() {
            if api_key.key_hash == key_hash && api_key.active {
                let now = current_timestamp();

                // Check expiry
                if let Some(expires_at) = api_key.expires_at {
                    if expires_at < now {
                        continue;
                    }
                }

                // Check rate limit
                if api_key.usage_count >= api_key.rate_limit {
                    return Ok(None);
                }

                found_key = Some((api_key.key_id.clone(), api_key.clone()));
                break;
            }
        }

        // Update usage if key found
        if let Some((key_id, api_key)) = found_key {
            drop(api_keys);
            let mut api_keys_mut = self.api_keys.write().await;
            if let Some(mut_key) = api_keys_mut.get_mut(&key_id) {
                mut_key.last_used = Some(current_timestamp());
                mut_key.usage_count += 1;
            }
            return Ok(Some(api_key));
        }

        Ok(None)
    }

    /// Authorize session-based request
    async fn authorize_session(
        &self,
        session: &UserSession,
        request: &AuthRequest,
    ) -> Result<AuthResult> {
        // Check IP allowlist for admin operations
        if matches!(request.operation, Operation::Admin) {
            if let Some(allowlist) = &self.policies.admin_ip_allowlist {
                if !allowlist.contains(&request.source_ip) {
                    return Ok(AuthResult {
                        authorized: false,
                        session: Some(session.clone()),
                        reason: Some("IP not in admin allowlist".to_string()),
                        required_actions: vec![],
                    });
                }
            }
        }

        // Check MFA for admin operations
        if matches!(request.operation, Operation::Admin)
            && self.policies.require_mfa_admin
            && !session.mfa_verified
        {
            return Ok(AuthResult {
                authorized: false,
                session: Some(session.clone()),
                reason: Some("MFA required for admin operations".to_string()),
                required_actions: vec!["mfa_verification".to_string()],
            });
        }

        // Check permissions
        let authorized = self.check_permissions(&session.roles, request).await?;

        Ok(AuthResult {
            authorized,
            session: Some(session.clone()),
            reason: if authorized {
                None
            } else {
                Some("Insufficient permissions".to_string())
            },
            required_actions: vec![],
        })
    }

    /// Authorize API key-based request
    async fn authorize_api_key(
        &self,
        api_key: &ApiKey,
        request: &AuthRequest,
    ) -> Result<AuthResult> {
        // API keys cannot perform admin operations
        if matches!(request.operation, Operation::Admin) {
            return Ok(AuthResult {
                authorized: false,
                session: None,
                reason: Some("API keys cannot perform admin operations".to_string()),
                required_actions: vec![],
            });
        }

        // Check permissions
        let authorized = self.check_permissions(&api_key.roles, request).await?;

        Ok(AuthResult {
            authorized,
            session: None,
            reason: if authorized {
                None
            } else {
                Some("Insufficient permissions".to_string())
            },
            required_actions: vec![],
        })
    }

    /// Check if roles have required permissions
    async fn check_permissions(
        &self,
        user_roles: &HashSet<String>,
        request: &AuthRequest,
    ) -> Result<bool> {
        let roles = self.roles.read().await;
        let permissions = self.permissions.read().await;

        // Collect all permissions from user roles
        let mut user_permissions = HashSet::new();
        for role_name in user_roles {
            if let Some(role) = roles.get(role_name) {
                if role.active {
                    user_permissions.extend(role.permissions.iter().cloned());
                }
            }
        }

        // Check if any permission allows the requested operation
        for perm_name in user_permissions {
            if let Some(permission) = permissions.get(&perm_name) {
                if permission.resource_type == request.resource
                    && permission.operations.contains(&request.operation)
                {
                    // Check scope
                    match &permission.scope {
                        PermissionScope::Global => return Ok(true),
                        PermissionScope::User(user) => {
                            if user == "self" || request.context.get("user_id") == Some(user) {
                                return Ok(true);
                            }
                        }
                        PermissionScope::Shard(shard_id) => {
                            if request
                                .context
                                .get("shard_id")
                                .and_then(|s| s.parse::<u64>().ok())
                                == Some(*shard_id)
                            {
                                return Ok(true);
                            }
                        }
                        PermissionScope::Resource(resource_id) => {
                            if request.resource_id.as_ref() == Some(resource_id) {
                                return Ok(true);
                            }
                        }
                        PermissionScope::Custom(_) => {
                            // Custom scope logic would go here
                            return Ok(true);
                        }
                    }
                }
            }
        }

        Ok(false)
    }

    /// Create new user session
    pub async fn create_session(
        &self,
        user_id: String,
        roles: HashSet<String>,
        source_ip: String,
    ) -> Result<UserSession> {
        let session_id = Uuid::new_v4().to_string();
        let token = self.generate_secure_token();
        let now = current_timestamp();

        let session = UserSession {
            session_id: session_id.clone(),
            user_id,
            token: token.clone(),
            roles,
            created_at: now,
            expires_at: now + self.policies.session_timeout,
            last_activity: now,
            source_ip,
            mfa_verified: false,
        };

        // Check session limits
        let sessions = self.sessions.read().await;
        let user_session_count = sessions
            .values()
            .filter(|s| s.user_id == session.user_id)
            .count();

        if user_session_count >= self.policies.max_sessions_per_user as usize {
            return Err(anyhow!("Maximum sessions per user exceeded"));
        }
        drop(sessions);

        // Store session
        let mut sessions_mut = self.sessions.write().await;
        sessions_mut.insert(session_id, session.clone());

        Ok(session)
    }

    /// Create new API key
    pub async fn create_api_key(
        &self,
        user_id: String,
        name: String,
        roles: HashSet<String>,
        expires_in_days: Option<u32>,
    ) -> Result<(String, ApiKey)> {
        let key_id = Uuid::new_v4().to_string();
        let raw_key = self.generate_secure_token();
        let key_hash = hash_string(&raw_key);
        let now = current_timestamp();

        let api_key = ApiKey {
            key_id: key_id.clone(),
            key_hash,
            user_id,
            name,
            roles,
            created_at: now,
            expires_at: expires_in_days.map(|days| now + (days as u64 * 24 * 60 * 60)),
            last_used: None,
            rate_limit: self.policies.api_key_rate_limit,
            usage_count: 0,
            active: true,
        };

        // Store API key
        let mut api_keys = self.api_keys.write().await;
        api_keys.insert(key_id, api_key.clone());

        Ok((raw_key, api_key))
    }

    /// Revoke session
    pub async fn revoke_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(session_id);
        Ok(())
    }

    /// Revoke API key
    pub async fn revoke_api_key(&self, key_id: &str) -> Result<()> {
        let mut api_keys = self.api_keys.write().await;
        if let Some(api_key) = api_keys.get_mut(key_id) {
            api_key.active = false;
        }
        Ok(())
    }

    /// Generate secure token
    fn generate_secure_token(&self) -> String {
        Uuid::new_v4().to_string() + &Uuid::new_v4().to_string()
    }

    /// Clean up expired sessions and API keys
    pub async fn cleanup_expired(&self) -> Result<()> {
        let now = current_timestamp();

        // Clean up sessions
        let mut sessions = self.sessions.write().await;
        sessions.retain(|_, session| {
            session.expires_at > now
                && (now - session.last_activity) < self.policies.session_timeout
        });
        drop(sessions);

        // Clean up API keys
        let mut api_keys = self.api_keys.write().await;
        api_keys.retain(|_, api_key| {
            api_key.active && api_key.expires_at.map_or(true, |exp| exp > now)
        });

        Ok(())
    }
}

impl Default for SecurityPolicies {
    fn default() -> Self {
        Self {
            session_timeout: 24 * 60 * 60, // 24 hours
            max_sessions_per_user: 5,
            require_mfa_admin: true,
            api_key_rate_limit: 1000,
            password_policy: PasswordPolicy::default(),
            admin_ip_allowlist: None,
        }
    }
}

impl Default for PasswordPolicy {
    fn default() -> Self {
        Self {
            min_length: 12,
            require_uppercase: true,
            require_lowercase: true,
            require_numbers: true,
            require_special: true,
            prevent_reuse: 5,
        }
    }
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Hash a string using SHA3-256
fn hash_string(input: &str) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_access_control() {
        let acm = AccessControlManager::new();

        // Wait for initialization
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Create session
        let session = acm
            .create_session(
                "test_user".to_string(),
                HashSet::from(["user".to_string()]),
                "127.0.0.1".to_string(),
            )
            .await
            .unwrap();

        // Test authorization
        let request = AuthRequest {
            token: session.token.clone(),
            resource: ResourceType::Storage,
            operation: Operation::Read,
            resource_id: None,
            source_ip: "127.0.0.1".to_string(),
            context: HashMap::new(),
        };

        let result = acm.authenticate(&request).await.unwrap();
        assert!(result.authorized);
    }
}
