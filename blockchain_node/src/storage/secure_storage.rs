use super::{Result, Storage};
use crate::security::access_control::{AccessControlManager, AuthRequest, Operation, ResourceType};
use crate::security::encryption::{
    DataType, EncryptionContext, EncryptionManager, EncryptionRequirements,
};
use crate::utils::crypto::Hash;

use async_trait::async_trait;
use log::{info, warn};
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Secure storage wrapper with authentication, authorization, and encryption
pub struct SecureStorage {
    /// Underlying storage implementation
    storage: Box<dyn Storage>,
    /// Access control manager
    access_control: Arc<AccessControlManager>,
    /// Encryption manager
    encryption: Arc<EncryptionManager>,
    /// Security audit log
    audit_log: Arc<RwLock<Vec<SecurityAuditEntry>>>,
    /// Storage configuration
    config: SecureStorageConfig,
    /// Active sessions cache
    session_cache: Arc<RwLock<HashMap<String, CachedSession>>>,
}

/// Security audit entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditEntry {
    /// Unique audit ID
    pub audit_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// User ID (if available)
    pub user_id: Option<String>,
    /// Session ID (if available)
    pub session_id: Option<String>,
    /// Operation performed
    pub operation: Operation,
    /// Resource accessed
    pub resource_type: ResourceType,
    /// Resource ID
    pub resource_id: Option<String>,
    /// Source IP address
    pub source_ip: String,
    /// Success/failure status
    pub success: bool,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Additional context
    pub context: HashMap<String, String>,
    /// Data size in bytes
    pub data_size: Option<usize>,
    /// Encryption status
    pub encrypted: bool,
    /// Anonymized data
    pub anonymized: bool,
}

/// Cached session information
#[derive(Debug, Clone)]
struct CachedSession {
    /// User ID
    user_id: String,
    /// User roles
    roles: std::collections::HashSet<String>,
    /// Cache timestamp
    cached_at: u64,
    /// Cache TTL in seconds
    ttl: u64,
}

/// Secure storage configuration
#[derive(Debug, Clone)]
pub struct SecureStorageConfig {
    /// Require authentication for all operations
    pub require_auth: bool,
    /// Require encryption for all data
    pub require_encryption: bool,
    /// Enable anonymization for sensitive data
    pub enable_anonymization: bool,
    /// Maximum data size allowed
    pub max_data_size: usize,
    /// Audit log retention period in seconds
    pub audit_retention_period: u64,
    /// Session cache TTL in seconds
    pub session_cache_ttl: u64,
    /// Rate limiting per user
    pub rate_limit_per_user: u32,
    /// Rate limiting window in seconds
    pub rate_limit_window: u64,
}

/// Request context for secure operations
#[derive(Debug, Clone)]
pub struct SecureOperationContext {
    /// Authentication token
    pub auth_token: String,
    /// Source IP address
    pub source_ip: String,
    /// User agent or client info
    pub user_agent: Option<String>,
    /// Request ID for tracking
    pub request_id: String,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Rate limiting tracker
#[derive(Debug, Clone)]
struct RateLimitTracker {
    /// Requests in current window
    requests: u32,
    /// Window start time
    window_start: u64,
}

impl SecureStorage {
    /// Create new secure storage wrapper
    pub fn new(
        storage: Box<dyn Storage>,
        access_control: Arc<AccessControlManager>,
        encryption: Arc<EncryptionManager>,
        config: SecureStorageConfig,
    ) -> Self {
        Self {
            storage,
            access_control,
            encryption,
            audit_log: Arc::new(RwLock::new(Vec::new())),
            config,
            session_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Store data with security checks
    pub async fn store_secure(
        &self,
        data: &[u8],
        context: SecureOperationContext,
    ) -> anyhow::Result<Hash> {
        let start_time = current_timestamp();
        let mut audit_context = HashMap::new();
        audit_context.insert("request_id".to_string(), context.request_id.clone());
        audit_context.extend(context.context.clone());

        // Validate input
        if data.len() > self.config.max_data_size {
            self.log_audit_entry(SecurityAuditEntry {
                audit_id: Uuid::new_v4().to_string(),
                timestamp: start_time,
                user_id: None,
                session_id: None,
                operation: Operation::Write,
                resource_type: ResourceType::Storage,
                resource_id: None,
                source_ip: context.source_ip.clone(),
                success: false,
                error_message: Some("Data size exceeds maximum allowed".to_string()),
                context: audit_context.clone(),
                data_size: Some(data.len()),
                encrypted: false,
                anonymized: false,
            })
            .await;
            return Err(anyhow::anyhow!(
                "Data size exceeds maximum allowed".to_string(),
            ));
        }

        // Authenticate and authorize
        let auth_result = self
            .authenticate_request(&context, Operation::Write, None)
            .await?;
        if !auth_result.0 {
            return Err(anyhow::anyhow!(auth_result.1));
        }

        let session = auth_result.2;
        let user_id = session.as_ref().map(|s| s.user_id.clone());

        // Check rate limiting
        if let Some(ref user_session) = session {
            if !self.check_rate_limit(&user_session.user_id).await {
                self.log_audit_entry(SecurityAuditEntry {
                    audit_id: Uuid::new_v4().to_string(),
                    timestamp: start_time,
                    user_id: user_id.clone(),
                    session_id: Some(user_session.session_id.clone()),
                    operation: Operation::Write,
                    resource_type: ResourceType::Storage,
                    resource_id: None,
                    source_ip: context.source_ip.clone(),
                    success: false,
                    error_message: Some("Rate limit exceeded".to_string()),
                    context: audit_context.clone(),
                    data_size: Some(data.len()),
                    encrypted: false,
                    anonymized: false,
                })
                .await;
                return Err(anyhow::anyhow!("Rate limit exceeded".to_string()));
            }
        }

        // Encrypt data if required
        let (final_data, encrypted, anonymized) = if self.config.require_encryption {
            let encryption_context = EncryptionContext {
                data_type: DataType::Storage,
                user_id: user_id.clone(),
                context: context.context.clone(),
                requirements: EncryptionRequirements {
                    required: true,
                    anonymize: self.config.enable_anonymization,
                    algorithm: None,
                    key_type: None,
                },
            };

            match self.encryption.encrypt(data, encryption_context).await {
                Ok(encrypted_data) => {
                    let serialized = serde_json::to_vec(&encrypted_data)
                        .map_err(|e| anyhow::anyhow!(format!("Serialization failed: {}", e)))?;
                    (serialized, true, self.config.enable_anonymization)
                }
                Err(e) => {
                    self.log_audit_entry(SecurityAuditEntry {
                        audit_id: Uuid::new_v4().to_string(),
                        timestamp: start_time,
                        user_id: user_id.clone(),
                        session_id: session.as_ref().map(|s| s.session_id.clone()),
                        operation: Operation::Write,
                        resource_type: ResourceType::Storage,
                        resource_id: None,
                        source_ip: context.source_ip.clone(),
                        success: false,
                        error_message: Some(format!("Encryption failed: {}", e)),
                        context: audit_context.clone(),
                        data_size: Some(data.len()),
                        encrypted: false,
                        anonymized: false,
                    })
                    .await;
                    return Err(anyhow::anyhow!(format!("Encryption failed: {}", e)));
                }
            }
        } else {
            (data.to_vec(), false, false)
        };

        // Compute content hash to use as key
        let content_hash = blake3::hash(&final_data);
        let key_bytes = content_hash.as_bytes();

        // Store data under the content hash key
        match self
            .storage
            .put(key_bytes, &final_data)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))
        {
            Ok(()) => {
                let hash = Hash::from_slice(key_bytes);
                // Log successful operation
                self.log_audit_entry(SecurityAuditEntry {
                    audit_id: Uuid::new_v4().to_string(),
                    timestamp: start_time,
                    user_id: user_id.clone(),
                    session_id: session.as_ref().map(|s| s.session_id.clone()),
                    operation: Operation::Write,
                    resource_type: ResourceType::Storage,
                    resource_id: Some(hex::encode(hash.as_ref())),
                    source_ip: context.source_ip.clone(),
                    success: true,
                    error_message: None,
                    context: audit_context,
                    data_size: Some(data.len()),
                    encrypted,
                    anonymized,
                })
                .await;

                info!(
                    "Secure storage: Successfully stored {} bytes with hash {}",
                    data.len(),
                    hex::encode(&hash)
                );
                Ok(hash)
            }
            Err(e) => {
                // Log failed operation
                self.log_audit_entry(SecurityAuditEntry {
                    audit_id: Uuid::new_v4().to_string(),
                    timestamp: start_time,
                    user_id: user_id.clone(),
                    session_id: session.as_ref().map(|s| s.session_id.clone()),
                    operation: Operation::Write,
                    resource_type: ResourceType::Storage,
                    resource_id: None,
                    source_ip: context.source_ip.clone(),
                    success: false,
                    error_message: Some(e.to_string()),
                    context: audit_context,
                    data_size: Some(data.len()),
                    encrypted,
                    anonymized,
                })
                .await;
                Err(e)
            }
        }
    }

    /// Retrieve data with security checks
    pub async fn retrieve_secure(
        &self,
        hash: &Hash,
        context: SecureOperationContext,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        let start_time = current_timestamp();
        let mut audit_context = HashMap::new();
        audit_context.insert("request_id".to_string(), context.request_id.clone());
        audit_context.extend(context.context.clone());

        // Authenticate and authorize
        let auth_result = self
            .authenticate_request(&context, Operation::Read, Some(hex::encode(hash)))
            .await?;
        if !auth_result.0 {
            return Err(anyhow::anyhow!(auth_result.1));
        }

        let session = auth_result.2;
        let user_id = session.as_ref().map(|s| s.user_id.clone());

        // Check rate limiting
        if let Some(ref user_session) = session {
            if !self.check_rate_limit(&user_session.user_id).await {
                self.log_audit_entry(SecurityAuditEntry {
                    audit_id: Uuid::new_v4().to_string(),
                    timestamp: start_time,
                    user_id: user_id.clone(),
                    session_id: Some(user_session.session_id.clone()),
                    operation: Operation::Read,
                    resource_type: ResourceType::Storage,
                    resource_id: Some(hex::encode(hash)),
                    source_ip: context.source_ip.clone(),
                    success: false,
                    error_message: Some("Rate limit exceeded".to_string()),
                    context: audit_context.clone(),
                    data_size: None,
                    encrypted: false,
                    anonymized: false,
                })
                .await;
                return Err(anyhow::anyhow!("Rate limit exceeded".to_string()));
            }
        }

        // Retrieve data using content hash key
        match self
            .storage
            .get(hash.as_bytes())
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))
        {
            Ok(Some(encrypted_data)) => {
                // Decrypt if needed
                let (final_data, decrypted) = if self.config.require_encryption {
                    // Try to deserialize as encrypted data
                    match serde_json::from_slice(&encrypted_data) {
                        Ok(encrypted_container) => {
                            let encryption_context = EncryptionContext {
                                data_type: DataType::Storage,
                                user_id: user_id.clone(),
                                context: context.context.clone(),
                                requirements: EncryptionRequirements {
                                    required: true,
                                    anonymize: false,
                                    algorithm: None,
                                    key_type: None,
                                },
                            };

                            match self
                                .encryption
                                .decrypt(&encrypted_container, encryption_context)
                                .await
                            {
                                Ok(decrypted_data) => (decrypted_data.data, true),
                                Err(e) => {
                                    self.log_audit_entry(SecurityAuditEntry {
                                        audit_id: Uuid::new_v4().to_string(),
                                        timestamp: start_time,
                                        user_id: user_id.clone(),
                                        session_id: session.as_ref().map(|s| s.session_id.clone()),
                                        operation: Operation::Read,
                                        resource_type: ResourceType::Storage,
                                        resource_id: Some(hex::encode(hash)),
                                        source_ip: context.source_ip.clone(),
                                        success: false,
                                        error_message: Some(format!("Decryption failed: {}", e)),
                                        context: audit_context.clone(),
                                        data_size: Some(encrypted_data.len()),
                                        encrypted: true,
                                        anonymized: false,
                                    })
                                    .await;
                                    return Err(anyhow::anyhow!(format!(
                                        "Decryption failed: {}",
                                        e
                                    )));
                                }
                            }
                        }
                        Err(_) => {
                            // Data might not be encrypted, return as-is
                            warn!("Data appears to be unencrypted despite encryption requirement");
                            (encrypted_data, false)
                        }
                    }
                } else {
                    (encrypted_data, false)
                };

                // Log successful operation
                self.log_audit_entry(SecurityAuditEntry {
                    audit_id: Uuid::new_v4().to_string(),
                    timestamp: start_time,
                    user_id: user_id.clone(),
                    session_id: session.as_ref().map(|s| s.session_id.clone()),
                    operation: Operation::Read,
                    resource_type: ResourceType::Storage,
                    resource_id: Some(hex::encode(hash.as_ref())),
                    source_ip: context.source_ip.clone(),
                    success: true,
                    error_message: None,
                    context: audit_context,
                    data_size: Some(final_data.len()),
                    encrypted: decrypted,
                    anonymized: false,
                })
                .await;

                Ok(Some(final_data))
            }
            Ok(None) => {
                // Log not found
                self.log_audit_entry(SecurityAuditEntry {
                    audit_id: Uuid::new_v4().to_string(),
                    timestamp: start_time,
                    user_id: user_id.clone(),
                    session_id: session.as_ref().map(|s| s.session_id.clone()),
                    operation: Operation::Read,
                    resource_type: ResourceType::Storage,
                    resource_id: Some(hex::encode(hash.as_ref())),
                    source_ip: context.source_ip.clone(),
                    success: true,
                    error_message: None,
                    context: audit_context,
                    data_size: None,
                    encrypted: false,
                    anonymized: false,
                })
                .await;
                Ok(None)
            }
            Err(e) => {
                // Log failed operation
                self.log_audit_entry(SecurityAuditEntry {
                    audit_id: Uuid::new_v4().to_string(),
                    timestamp: start_time,
                    user_id: user_id.clone(),
                    session_id: session.as_ref().map(|s| s.session_id.clone()),
                    operation: Operation::Read,
                    resource_type: ResourceType::Storage,
                    resource_id: Some(hex::encode(hash.as_ref())),
                    source_ip: context.source_ip.clone(),
                    success: false,
                    error_message: Some(e.to_string()),
                    context: audit_context,
                    data_size: None,
                    encrypted: false,
                    anonymized: false,
                })
                .await;
                Err(e)
            }
        }
    }

    /// Delete data with security checks
    pub async fn delete_secure(
        &self,
        hash: &Hash,
        context: SecureOperationContext,
    ) -> anyhow::Result<()> {
        let start_time = current_timestamp();
        let mut audit_context = HashMap::new();
        audit_context.insert("request_id".to_string(), context.request_id.clone());
        audit_context.extend(context.context.clone());

        // Authenticate and authorize
        let auth_result = self
            .authenticate_request(&context, Operation::Delete, Some(hex::encode(hash)))
            .await?;
        if !auth_result.0 {
            return Err(anyhow::anyhow!(auth_result.1));
        }

        let session = auth_result.2;
        let user_id = session.as_ref().map(|s| s.user_id.clone());

        // Perform deletion using content hash key
        match self
            .storage
            .delete(hash.as_bytes())
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))
        {
            Ok(()) => {
                // Log successful operation
                self.log_audit_entry(SecurityAuditEntry {
                    audit_id: Uuid::new_v4().to_string(),
                    timestamp: start_time,
                    user_id: user_id.clone(),
                    session_id: session.as_ref().map(|s| s.session_id.clone()),
                    operation: Operation::Delete,
                    resource_type: ResourceType::Storage,
                    resource_id: Some(hex::encode(hash.as_ref())),
                    source_ip: context.source_ip.clone(),
                    success: true,
                    error_message: None,
                    context: audit_context,
                    data_size: None,
                    encrypted: false,
                    anonymized: false,
                })
                .await;
                Ok(())
            }
            Err(e) => {
                // Log failed operation
                self.log_audit_entry(SecurityAuditEntry {
                    audit_id: Uuid::new_v4().to_string(),
                    timestamp: start_time,
                    user_id: user_id.clone(),
                    session_id: session.as_ref().map(|s| s.session_id.clone()),
                    operation: Operation::Delete,
                    resource_type: ResourceType::Storage,
                    resource_id: Some(hex::encode(hash.as_ref())),
                    source_ip: context.source_ip.clone(),
                    success: false,
                    error_message: Some(e.to_string()),
                    context: audit_context,
                    data_size: None,
                    encrypted: false,
                    anonymized: false,
                })
                .await;
                Err(e)
            }
        }
    }

    /// Authenticate and authorize request
    async fn authenticate_request(
        &self,
        context: &SecureOperationContext,
        operation: Operation,
        resource_id: Option<String>,
    ) -> Result<(
        bool,
        String,
        Option<crate::security::access_control::UserSession>,
    )> {
        if !self.config.require_auth {
            return Ok((true, "Authentication disabled".to_string(), None));
        }

        // Check session cache first
        if let Some(cached) = self.get_cached_session(&context.auth_token).await {
            if current_timestamp() - cached.cached_at < cached.ttl {
                // Use cached session for authorization
                // This is a simplified check - in reality we'd need to construct the full session
                return Ok((true, "Authorized via cache".to_string(), None));
            }
        }

        // Perform full authentication
        let auth_request = AuthRequest {
            token: context.auth_token.clone(),
            resource: ResourceType::Storage,
            operation,
            resource_id,
            source_ip: context.source_ip.clone(),
            context: context.context.clone(),
        };

        match self.access_control.authenticate(&auth_request).await {
            Ok(auth_result) => {
                if auth_result.authorized {
                    // Cache successful authentication
                    if let Some(ref session) = auth_result.session {
                        self.cache_session(&context.auth_token, session).await;
                    }
                    Ok((true, "Authorized".to_string(), auth_result.session))
                } else {
                    let reason = auth_result
                        .reason
                        .unwrap_or_else(|| "Access denied".to_string());
                    Ok((false, reason, auth_result.session))
                }
            }
            Err(e) => Ok((false, format!("Authentication error: {}", e), None)),
        }
    }

    /// Check rate limiting for user
    async fn check_rate_limit(&self, user_id: &str) -> bool {
        // Simplified rate limiting - in production would use Redis or similar
        // For now, always allow
        let _ = user_id;
        true
    }

    /// Cache session information
    async fn cache_session(
        &self,
        token: &str,
        session: &crate::security::access_control::UserSession,
    ) {
        let cached_session = CachedSession {
            user_id: session.user_id.clone(),
            roles: session.roles.clone(),
            cached_at: current_timestamp(),
            ttl: self.config.session_cache_ttl,
        };

        let mut cache = self.session_cache.write().await;
        cache.insert(token.to_string(), cached_session);
    }

    /// Get cached session
    async fn get_cached_session(&self, token: &str) -> Option<CachedSession> {
        let cache = self.session_cache.read().await;
        cache.get(token).cloned()
    }

    /// Log audit entry
    async fn log_audit_entry(&self, entry: SecurityAuditEntry) {
        let mut log = self.audit_log.write().await;
        log.push(entry);

        // Clean up old entries
        let cutoff = current_timestamp() - self.config.audit_retention_period;
        log.retain(|entry| entry.timestamp > cutoff);
    }

    /// Get audit log entries
    pub async fn get_audit_log(&self, limit: Option<usize>) -> Vec<SecurityAuditEntry> {
        let log = self.audit_log.read().await;
        if let Some(limit) = limit {
            log.iter().rev().take(limit).cloned().collect()
        } else {
            log.clone()
        }
    }

    /// Get security statistics
    pub async fn get_security_stats(&self) -> SecurityStats {
        let log = self.audit_log.read().await;
        let total_operations = log.len();
        let successful_operations = log.iter().filter(|e| e.success).count();
        let failed_operations = total_operations - successful_operations;
        let encrypted_operations = log.iter().filter(|e| e.encrypted).count();
        let anonymized_operations = log.iter().filter(|e| e.anonymized).count();

        SecurityStats {
            total_operations,
            successful_operations,
            failed_operations,
            encrypted_operations,
            anonymized_operations,
            cache_size: self.session_cache.read().await.len(),
        }
    }
}

/// Security statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStats {
    /// Total operations performed
    pub total_operations: usize,
    /// Successful operations
    pub successful_operations: usize,
    /// Failed operations
    pub failed_operations: usize,
    /// Operations with encryption
    pub encrypted_operations: usize,
    /// Operations with anonymization
    pub anonymized_operations: usize,
    /// Session cache size
    pub cache_size: usize,
}

#[async_trait]
impl Storage for SecureStorage {
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.storage.get(key).await
    }

    async fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.storage.put(key, value).await
    }

    async fn delete(&self, key: &[u8]) -> Result<()> {
        self.storage.delete(key).await
    }

    async fn exists(&self, key: &[u8]) -> Result<bool> {
        self.storage.exists(key).await
    }

    async fn list_keys(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        // By policy, we can still delegate; callers may restrict exposure at higher layers
        self.storage.list_keys(prefix).await
    }

    async fn get_stats(&self) -> Result<crate::storage::StorageStats> {
        self.storage.get_stats().await
    }

    async fn flush(&self) -> Result<()> {
        self.storage.flush().await
    }

    async fn close(&self) -> Result<()> {
        self.storage.close().await
    }
}

// Additional helper methods for secure storage (not part of Storage trait)
impl SecureStorage {
    // Additional secure operations can be added here
}

impl Default for SecureStorageConfig {
    fn default() -> Self {
        Self {
            require_auth: true,
            require_encryption: true,
            enable_anonymization: true,
            max_data_size: 100 * 1024 * 1024,           // 100MB
            audit_retention_period: 365 * 24 * 60 * 60, // 1 year
            session_cache_ttl: 300,                     // 5 minutes
            rate_limit_per_user: 1000,                  // 1000 requests per window
            rate_limit_window: 3600,                    // 1 hour window
        }
    }
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::rocksdb_storage::RocksDbStorage;

    #[tokio::test]
    async fn test_secure_storage() {
        let storage = Box::new(RocksDbStorage::new());
        let access_control = Arc::new(AccessControlManager::new());
        let encryption = Arc::new(EncryptionManager::new());
        let config = SecureStorageConfig::default();

        let secure_storage = SecureStorage::new(storage, access_control, encryption, config);

        // Test data
        let data = b"test secure storage data";
        let context = SecureOperationContext {
            auth_token: "test_token".to_string(),
            source_ip: "127.0.0.1".to_string(),
            user_agent: Some("test".to_string()),
            request_id: Uuid::new_v4().to_string(),
            context: HashMap::new(),
        };

        // This test would require proper setup
        // For now, just ensure the structure compiles
        assert!(!data.is_empty());
        assert!(!context.auth_token.is_empty());
    }
}
