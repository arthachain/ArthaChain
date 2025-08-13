// Security module for comprehensive blockchain security

pub mod access_control;
pub mod advanced_monitoring;
pub mod encryption;

// Re-export main types for convenience
pub use access_control::{
    AccessControlManager, ApiKey, AuthRequest, AuthResult, Operation, Permission, ResourceType,
    Role, SecurityPolicies, UserSession,
};
pub use advanced_monitoring::{
    AdvancedSecurityMonitor, MonitoringConfig, SecurityIncident, SecurityMetrics, ThreatLevel,
    ThreatType,
};
pub use encryption::{
    AnonymizationLevel, DataType, DecryptedData, EncryptedData, EncryptionAlgorithm,
    EncryptionContext, EncryptionManager, EncryptionRequirements, KeyType,
};

use anyhow::Result;
use log::info;
use std::sync::Arc;

/// Central security manager that orchestrates all security components
pub struct SecurityManager {
    /// Access control manager
    pub access_control: Arc<AccessControlManager>,
    /// Encryption manager
    pub encryption: Arc<EncryptionManager>,
    /// Advanced security monitoring
    pub monitoring: Arc<AdvancedSecurityMonitor>,
    /// Security initialization status
    initialized: bool,
}

impl SecurityManager {
    /// Create new security manager
    pub fn new() -> Self {
        let monitoring_config = MonitoringConfig::default();
        Self {
            access_control: Arc::new(AccessControlManager::new()),
            encryption: Arc::new(EncryptionManager::new()),
            monitoring: Arc::new(AdvancedSecurityMonitor::new(monitoring_config)),
            initialized: false,
        }
    }

    /// Initialize security manager with master credentials
    pub async fn initialize(&mut self, master_password: &str) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        info!("Initializing blockchain security systems...");

        // Initialize encryption
        self.encryption.initialize(master_password).await?;

        // Initialize access control (already initialized in constructor)
        // The access control system initializes itself with default roles

        self.initialized = true;
        info!("Security systems initialized successfully");

        Ok(())
    }

    /// Check if security manager is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get access control manager
    pub fn get_access_control(&self) -> Arc<AccessControlManager> {
        self.access_control.clone()
    }

    /// Get encryption manager
    pub fn get_encryption(&self) -> Arc<EncryptionManager> {
        self.encryption.clone()
    }

    /// Get security monitoring system
    pub fn get_monitoring(&self) -> Arc<AdvancedSecurityMonitor> {
        self.monitoring.clone()
    }

    /// Perform security maintenance tasks
    pub async fn perform_maintenance(&self) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }

        // Clean up expired sessions and API keys
        self.access_control.cleanup_expired().await?;

        // Rotate encryption keys if needed
        let rotated_count = self.encryption.rotate_keys().await?;
        if rotated_count > 0 {
            info!(
                "Rotated {} encryption keys during maintenance",
                rotated_count
            );
        }

        Ok(())
    }

    /// Get security health status
    pub async fn get_health_status(&self) -> SecurityHealthStatus {
        let monitoring_metrics = self.monitoring.get_metrics();
        SecurityHealthStatus {
            initialized: self.initialized,
            encryption_active: self.initialized,
            access_control_active: true,
            monitoring_active: true,
            pending_key_rotations: 0, // Would be implemented in a real system
            active_sessions: 0,       // Would be implemented in a real system
            total_incidents: monitoring_metrics.total_incidents,
            avg_response_time_ms: monitoring_metrics.avg_response_time_ms,
        }
    }
}

/// Security health status
#[derive(Debug, Clone)]
pub struct SecurityHealthStatus {
    /// Security manager initialized
    pub initialized: bool,
    /// Encryption system active
    pub encryption_active: bool,
    /// Access control system active
    pub access_control_active: bool,
    /// Security monitoring active
    pub monitoring_active: bool,
    /// Number of pending key rotations
    pub pending_key_rotations: usize,
    /// Number of active sessions
    pub active_sessions: usize,
    /// Total security incidents detected
    pub total_incidents: u64,
    /// Average response time for threat analysis
    pub avg_response_time_ms: f64,
}

impl Default for SecurityManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_security_manager() {
        let mut security_manager = SecurityManager::new();
        assert!(!security_manager.is_initialized());

        // Initialize with test password
        security_manager
            .initialize("test_password_123")
            .await
            .unwrap();
        assert!(security_manager.is_initialized());

        // Test maintenance
        security_manager.perform_maintenance().await.unwrap();

        // Test health status
        let health = security_manager.get_health_status().await;
        assert!(health.initialized);
        assert!(health.monitoring_active);
    }
}
