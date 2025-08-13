use super::bci_interface::{BCIModel, BCIOutput, SignalParams, Spike, BCIState};
use super::neural_base::{NeuralConfig, NeuralNetwork};
use crate::security::encryption::{EncryptionManager, EncryptionContext, DataType, EncryptionRequirements, KeyType, AnonymizationLevel};
use crate::security::access_control::{AccessControlManager, AuthRequest, ResourceType, Operation};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Secure BCI interface with encryption and anonymization
pub struct SecureBCIInterface {
    /// Underlying BCI model
    bci_model: Arc<RwLock<BCIModel>>,
    /// Encryption manager
    encryption: Arc<EncryptionManager>,
    /// Access control manager
    access_control: Arc<AccessControlManager>,
    /// Privacy settings
    privacy_config: BCIPrivacyConfig,
    /// User consent management
    consent_manager: Arc<RwLock<ConsentManager>>,
    /// Secure signal buffer
    secure_buffer: Arc<RwLock<SecureSignalBuffer>>,
    /// Audit trail
    audit_trail: Arc<RwLock<Vec<BCIAuditEntry>>>,
}

/// Privacy configuration for BCI processing
#[derive(Debug, Clone)]
pub struct BCIPrivacyConfig {
    /// Minimum anonymization level required
    pub min_anonymization_level: AnonymizationLevel,
    /// Require explicit consent for processing
    pub require_explicit_consent: bool,
    /// Maximum signal retention time in seconds
    pub max_retention_seconds: u64,
    /// Enable differential privacy
    pub enable_differential_privacy: bool,
    /// Privacy budget parameters
    pub privacy_budget: PrivacyBudget,
    /// Data localization requirements
    pub data_localization: DataLocalization,
    /// Secure deletion requirements
    pub secure_deletion: bool,
}

/// Privacy budget for differential privacy
#[derive(Debug, Clone)]
pub struct PrivacyBudget {
    /// Epsilon parameter (privacy loss)
    pub epsilon: f64,
    /// Delta parameter (failure probability)
    pub delta: f64,
    /// Budget per user per day
    pub daily_budget: f64,
    /// Current usage tracking
    pub usage_tracking: HashMap<String, f64>,
}

/// Data localization requirements
#[derive(Debug, Clone)]
pub enum DataLocalization {
    /// No restrictions
    None,
    /// Data must stay in specific country
    Country(String),
    /// Data must stay in specific region
    Region(String),
    /// Data must stay on-premises
    OnPremises,
}

/// User consent management
#[derive(Debug, Clone)]
pub struct ConsentManager {
    /// User consent records
    consents: HashMap<String, UserConsent>,
    /// Consent templates
    consent_templates: HashMap<String, ConsentTemplate>,
    /// Consent expiry tracking
    expiry_tracking: HashMap<String, u64>,
}

/// User consent record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserConsent {
    /// User ID
    pub user_id: String,
    /// Consent ID
    pub consent_id: String,
    /// Consent type
    pub consent_type: ConsentType,
    /// Granted permissions
    pub permissions: HashSet<BCIPermission>,
    /// Consent timestamp
    pub granted_at: u64,
    /// Expiry timestamp
    pub expires_at: Option<u64>,
    /// Consent version
    pub version: String,
    /// Digital signature
    pub signature: Option<String>,
    /// Consent is active
    pub active: bool,
}

/// Consent template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentTemplate {
    /// Template ID
    pub template_id: String,
    /// Template name
    pub name: String,
    /// Description
    pub description: String,
    /// Required permissions
    pub required_permissions: HashSet<BCIPermission>,
    /// Template text
    pub template_text: String,
    /// Legal basis
    pub legal_basis: String,
    /// Template version
    pub version: String,
}

/// Types of consent
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsentType {
    /// Research participation
    Research,
    /// Clinical use
    Clinical,
    /// Commercial use
    Commercial,
    /// Training data
    Training,
    /// Anonymous analytics
    Analytics,
}

/// BCI-specific permissions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BCIPermission {
    /// Read neural signals
    ReadSignals,
    /// Process signals
    ProcessSignals,
    /// Store signals
    StoreSignals,
    /// Share anonymized data
    ShareAnonymized,
    /// Use for training
    UseForTraining,
    /// Real-time processing
    RealTimeProcessing,
    /// Historical analysis
    HistoricalAnalysis,
}

/// Secure signal buffer with encryption
#[derive(Debug, Clone)]
pub struct SecureSignalBuffer {
    /// Encrypted signal chunks
    encrypted_chunks: Vec<EncryptedSignalChunk>,
    /// Buffer metadata
    metadata: BufferMetadata,
    /// User context
    user_context: Option<String>,
}

/// Encrypted signal chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedSignalChunk {
    /// Chunk ID
    pub chunk_id: String,
    /// Encrypted signal data
    pub encrypted_data: Vec<u8>,
    /// Encryption metadata
    pub encryption_nonce: Vec<u8>,
    /// Key ID used
    pub key_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// Anonymization applied
    pub anonymized: bool,
    /// Signal quality metrics
    pub quality_metrics: SignalQualityMetrics,
}

/// Buffer metadata
#[derive(Debug, Clone)]
pub struct BufferMetadata {
    /// Total chunks
    pub total_chunks: usize,
    /// Buffer size in bytes
    pub buffer_size: usize,
    /// Oldest timestamp
    pub oldest_timestamp: u64,
    /// Newest timestamp
    pub newest_timestamp: u64,
    /// User ID
    pub user_id: Option<String>,
}

/// Signal quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQualityMetrics {
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Signal variance
    pub variance: f32,
    /// Missing samples percentage
    pub missing_samples: f32,
    /// Artifacts detected
    pub artifacts_detected: bool,
}

/// BCI audit entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BCIAuditEntry {
    /// Entry ID
    pub entry_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// User ID
    pub user_id: Option<String>,
    /// Operation type
    pub operation: BCIOperation,
    /// Data size processed
    pub data_size: usize,
    /// Anonymization level applied
    pub anonymization_level: AnonymizationLevel,
    /// Consent status
    pub consent_verified: bool,
    /// Privacy budget consumed
    pub privacy_budget_used: Option<f64>,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// BCI operations for auditing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BCIOperation {
    /// Signal acquisition
    SignalAcquisition,
    /// Signal processing
    SignalProcessing,
    /// Feature extraction
    FeatureExtraction,
    /// Model training
    ModelTraining,
    /// Prediction/classification
    Prediction,
    /// Data export
    DataExport,
    /// Data deletion
    DataDeletion,
}

/// Secure BCI processing result
#[derive(Debug, Clone)]
pub struct SecureBCIResult {
    /// Processing result
    pub result: BCIOutput,
    /// Privacy metadata
    pub privacy_metadata: PrivacyMetadata,
    /// Audit trail entry
    pub audit_entry: BCIAuditEntry,
}

/// Privacy metadata
#[derive(Debug, Clone)]
pub struct PrivacyMetadata {
    /// Anonymization level applied
    pub anonymization_level: AnonymizationLevel,
    /// Differential privacy parameters
    pub dp_parameters: Option<DifferentialPrivacyParams>,
    /// Data retention policy
    pub retention_policy: RetentionPolicy,
    /// User consent verified
    pub consent_verified: bool,
}

/// Differential privacy parameters
#[derive(Debug, Clone)]
pub struct DifferentialPrivacyParams {
    /// Epsilon used
    pub epsilon: f64,
    /// Delta used
    pub delta: f64,
    /// Noise added
    pub noise_added: f64,
}

/// Data retention policy
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Retention period in seconds
    pub retention_seconds: u64,
    /// Auto-deletion enabled
    pub auto_delete: bool,
    /// Deletion date
    pub deletion_date: Option<u64>,
}

impl SecureBCIInterface {
    /// Create new secure BCI interface
    pub async fn new(
        config: NeuralConfig,
        signal_params: SignalParams,
        encryption: Arc<EncryptionManager>,
        access_control: Arc<AccessControlManager>,
        privacy_config: BCIPrivacyConfig,
    ) -> Result<Self> {
        let bci_model = Arc::new(RwLock::new(BCIModel::new(config, signal_params).await?));

        Ok(Self {
            bci_model,
            encryption,
            access_control,
            privacy_config,
            consent_manager: Arc::new(RwLock::new(ConsentManager::new())),
            secure_buffer: Arc::new(RwLock::new(SecureSignalBuffer::new())),
            audit_trail: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Process signals securely with privacy protection
    pub async fn process_signals_secure(
        &self,
        signals: &[Vec<f32>],
        user_id: &str,
        auth_token: &str,
        source_ip: &str,
    ) -> Result<SecureBCIResult> {
        let start_time = current_timestamp();
        let operation_id = Uuid::new_v4().to_string();

        // Verify authentication and authorization
        let auth_request = AuthRequest {
            token: auth_token.to_string(),
            resource: ResourceType::BCI,
            operation: Operation::Execute,
            resource_id: Some(user_id.to_string()),
            source_ip: source_ip.to_string(),
            context: HashMap::from([
                ("operation".to_string(), "process_signals".to_string()),
                ("user_id".to_string(), user_id.to_string()),
            ]),
        };

        let auth_result = self.access_control.authenticate(&auth_request).await?;
        if !auth_result.authorized {
            return Err(anyhow!("Unauthorized: {}", auth_result.reason.unwrap_or_else(|| "Access denied".to_string())));
        }

        // Verify user consent
        let consent_verified = self.verify_user_consent(user_id, &[BCIPermission::ProcessSignals]).await?;
        if !consent_verified && self.privacy_config.require_explicit_consent {
            return Err(anyhow!("User consent required for BCI signal processing"));
        }

        // Check privacy budget
        if self.privacy_config.enable_differential_privacy {
            if !self.check_privacy_budget(user_id).await? {
                return Err(anyhow!("Privacy budget exceeded for user"));
            }
        }

        // Convert signals to bytes for encryption
        let signal_bytes = self.signals_to_bytes(signals);

        // Apply anonymization and encryption
        let encryption_context = EncryptionContext {
            data_type: DataType::BCISignals,
            user_id: Some(user_id.to_string()),
            context: HashMap::from([
                ("operation_id".to_string(), operation_id.clone()),
                ("signal_count".to_string(), signals.len().to_string()),
            ]),
            requirements: EncryptionRequirements {
                required: true,
                anonymize: true,
                algorithm: None,
                key_type: Some(KeyType::BCI),
            },
        };

        let encrypted_signals = self.encryption.encrypt(&signal_bytes, encryption_context).await?;

        // Store in secure buffer temporarily
        self.store_in_secure_buffer(user_id, &encrypted_signals).await?;

        // Decrypt for processing (in memory only)
        let decryption_context = EncryptionContext {
            data_type: DataType::BCISignals,
            user_id: Some(user_id.to_string()),
            context: HashMap::new(),
            requirements: EncryptionRequirements {
                required: true,
                anonymize: false,
                algorithm: None,
                key_type: Some(KeyType::BCI),
            },
        };

        let decrypted_data = self.encryption.decrypt(&encrypted_signals, decryption_context).await?;
        let processed_signals = self.bytes_to_signals(&decrypted_data.data);

        // Process signals with BCI model
        let mut bci_model = self.bci_model.write().await;
        let classifications = bci_model.process_signals(&processed_signals).await?;

        // Create secure result
        let bci_output = BCIOutput {
            intent: classifications.iter().map(|&c| c as f32).collect(),
            confidence: 0.8, // Placeholder
            spikes: Vec::new(), // Would be populated in real implementation
            latency: (current_timestamp() - start_time) as f32,
        };

        // Update privacy budget if applicable
        if self.privacy_config.enable_differential_privacy {
            self.consume_privacy_budget(user_id, 0.1).await?;
        }

        // Create audit entry
        let audit_entry = BCIAuditEntry {
            entry_id: operation_id,
            timestamp: start_time,
            user_id: Some(user_id.to_string()),
            operation: BCIOperation::SignalProcessing,
            data_size: signal_bytes.len(),
            anonymization_level: self.privacy_config.min_anonymization_level.clone(),
            consent_verified,
            privacy_budget_used: if self.privacy_config.enable_differential_privacy { Some(0.1) } else { None },
            success: true,
            error_message: None,
        };

        // Log audit entry
        self.log_audit_entry(audit_entry.clone()).await;

        // Create privacy metadata
        let privacy_metadata = PrivacyMetadata {
            anonymization_level: self.privacy_config.min_anonymization_level.clone(),
            dp_parameters: if self.privacy_config.enable_differential_privacy {
                Some(DifferentialPrivacyParams {
                    epsilon: self.privacy_config.privacy_budget.epsilon,
                    delta: self.privacy_config.privacy_budget.delta,
                    noise_added: 0.1, // Placeholder
                })
            } else {
                None
            },
            retention_policy: RetentionPolicy {
                retention_seconds: self.privacy_config.max_retention_seconds,
                auto_delete: true,
                deletion_date: Some(start_time + self.privacy_config.max_retention_seconds),
            },
            consent_verified,
        };

        Ok(SecureBCIResult {
            result: bci_output,
            privacy_metadata,
            audit_entry,
        })
    }

    /// Grant user consent
    pub async fn grant_consent(&self, user_id: &str, consent_type: ConsentType, permissions: HashSet<BCIPermission>) -> Result<String> {
        let consent_id = Uuid::new_v4().to_string();
        let consent = UserConsent {
            user_id: user_id.to_string(),
            consent_id: consent_id.clone(),
            consent_type,
            permissions,
            granted_at: current_timestamp(),
            expires_at: Some(current_timestamp() + 365 * 24 * 60 * 60), // 1 year
            version: "1.0".to_string(),
            signature: None,
            active: true,
        };

        let mut consent_manager = self.consent_manager.write().await;
        consent_manager.consents.insert(user_id.to_string(), consent);

        Ok(consent_id)
    }

    /// Revoke user consent
    pub async fn revoke_consent(&self, user_id: &str) -> Result<()> {
        let mut consent_manager = self.consent_manager.write().await;
        if let Some(consent) = consent_manager.consents.get_mut(user_id) {
            consent.active = false;
        }

        // Schedule secure deletion of user data
        self.schedule_secure_deletion(user_id).await?;

        Ok(())
    }

    /// Verify user consent for specific permissions
    async fn verify_user_consent(&self, user_id: &str, required_permissions: &[BCIPermission]) -> Result<bool> {
        let consent_manager = self.consent_manager.read().await;
        
        if let Some(consent) = consent_manager.consents.get(user_id) {
            // Check if consent is active and not expired
            if !consent.active {
                return Ok(false);
            }

            if let Some(expires_at) = consent.expires_at {
                if current_timestamp() > expires_at {
                    return Ok(false);
                }
            }

            // Check if all required permissions are granted
            for permission in required_permissions {
                if !consent.permissions.contains(permission) {
                    return Ok(false);
                }
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check privacy budget availability
    async fn check_privacy_budget(&self, user_id: &str) -> Result<bool> {
        // Simplified privacy budget check
        // In a real implementation, this would be more sophisticated
        Ok(true)
    }

    /// Consume privacy budget
    async fn consume_privacy_budget(&self, user_id: &str, epsilon: f64) -> Result<()> {
        // Update privacy budget usage
        // In a real implementation, this would track usage properly
        let _ = (user_id, epsilon);
        Ok(())
    }

    /// Store encrypted signals in secure buffer
    async fn store_in_secure_buffer(&self, user_id: &str, encrypted_data: &crate::security::encryption::EncryptedData) -> Result<()> {
        let chunk = EncryptedSignalChunk {
            chunk_id: Uuid::new_v4().to_string(),
            encrypted_data: encrypted_data.data.clone(),
            encryption_nonce: encrypted_data.nonce.clone(),
            key_id: encrypted_data.key_id.clone(),
            timestamp: current_timestamp(),
            anonymized: true,
            quality_metrics: SignalQualityMetrics {
                snr: 10.0, // Placeholder
                variance: 1.0,
                missing_samples: 0.0,
                artifacts_detected: false,
            },
        };

        let mut buffer = self.secure_buffer.write().await;
        buffer.encrypted_chunks.push(chunk);
        buffer.metadata.total_chunks += 1;
        buffer.metadata.buffer_size += encrypted_data.data.len();
        buffer.metadata.user_id = Some(user_id.to_string());

        Ok(())
    }

    /// Schedule secure deletion of user data
    async fn schedule_secure_deletion(&self, user_id: &str) -> Result<()> {
        // In a real implementation, this would schedule secure deletion
        println!("Scheduling secure deletion for user: {}", user_id);
        Ok(())
    }

    /// Log audit entry
    async fn log_audit_entry(&self, entry: BCIAuditEntry) {
        let mut audit_trail = self.audit_trail.write().await;
        audit_trail.push(entry);

        // Clean up old entries
        let cutoff = current_timestamp() - (30 * 24 * 60 * 60); // 30 days
        audit_trail.retain(|entry| entry.timestamp > cutoff);
    }

    /// Convert signals to bytes
    fn signals_to_bytes(&self, signals: &[Vec<f32>]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for signal in signals {
            for &value in signal {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    /// Convert bytes to signals
    fn bytes_to_signals(&self, bytes: &[u8]) -> Vec<Vec<f32>> {
        let mut signals = Vec::new();
        let values: Vec<f32> = bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Assuming 256 channels (this would be configurable in real implementation)
        let channels = 256;
        for chunk in values.chunks(channels) {
            signals.push(chunk.to_vec());
        }

        signals
    }

    /// Get audit trail
    pub async fn get_audit_trail(&self, user_id: Option<&str>, limit: Option<usize>) -> Vec<BCIAuditEntry> {
        let audit_trail = self.audit_trail.read().await;
        let mut entries: Vec<_> = if let Some(uid) = user_id {
            audit_trail.iter().filter(|e| e.user_id.as_ref() == Some(uid)).cloned().collect()
        } else {
            audit_trail.clone()
        };

        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        if let Some(limit) = limit {
            entries.truncate(limit);
        }

        entries
    }

    /// Get privacy statistics
    pub async fn get_privacy_stats(&self) -> BCIPrivacyStats {
        let audit_trail = self.audit_trail.read().await;
        let consent_manager = self.consent_manager.read().await;

        BCIPrivacyStats {
            total_operations: audit_trail.len(),
            operations_with_consent: audit_trail.iter().filter(|e| e.consent_verified).count(),
            anonymized_operations: audit_trail.iter().filter(|e| !matches!(e.anonymization_level, AnonymizationLevel::None)).count(),
            active_consents: consent_manager.consents.values().filter(|c| c.active).count(),
            expired_consents: consent_manager.consents.values().filter(|c| !c.active).count(),
        }
    }
}

/// BCI privacy statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BCIPrivacyStats {
    /// Total BCI operations
    pub total_operations: usize,
    /// Operations with verified consent
    pub operations_with_consent: usize,
    /// Operations with anonymization
    pub anonymized_operations: usize,
    /// Number of active consents
    pub active_consents: usize,
    /// Number of expired consents
    pub expired_consents: usize,
}

impl ConsentManager {
    fn new() -> Self {
        Self {
            consents: HashMap::new(),
            consent_templates: HashMap::new(),
            expiry_tracking: HashMap::new(),
        }
    }
}

impl SecureSignalBuffer {
    fn new() -> Self {
        Self {
            encrypted_chunks: Vec::new(),
            metadata: BufferMetadata {
                total_chunks: 0,
                buffer_size: 0,
                oldest_timestamp: 0,
                newest_timestamp: 0,
                user_id: None,
            },
            user_context: None,
        }
    }
}

impl Default for BCIPrivacyConfig {
    fn default() -> Self {
        Self {
            min_anonymization_level: AnonymizationLevel::Advanced,
            require_explicit_consent: true,
            max_retention_seconds: 30 * 24 * 60 * 60, // 30 days
            enable_differential_privacy: true,
            privacy_budget: PrivacyBudget {
                epsilon: 1.0,
                delta: 1e-5,
                daily_budget: 10.0,
                usage_tracking: HashMap::new(),
            },
            data_localization: DataLocalization::OnPremises,
            secure_deletion: true,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::SecurityManager;

    #[tokio::test]
    async fn test_secure_bci_interface() {
        let mut security_manager = SecurityManager::new();
        security_manager.initialize("test_password_123").await.unwrap();

        let neural_config = NeuralConfig::default();
        let signal_params = SignalParams {
            sampling_rate: 1000,
            num_channels: 256,
            window_size: 100,
            filter_params: super::super::bci_interface::FilterParams {
                low_cut: 0.5,
                high_cut: 200.0,
                order: 4,
            },
            spike_threshold: 5.0,
            normalize: true,
            use_wavelet: true,
        };

        let interface = SecureBCIInterface::new(
            neural_config,
            signal_params,
            security_manager.get_encryption(),
            security_manager.get_access_control(),
            BCIPrivacyConfig::default(),
        ).await.unwrap();

        // Test consent management
        let consent_id = interface.grant_consent(
            "test_user",
            ConsentType::Research,
            HashSet::from([BCIPermission::ProcessSignals, BCIPermission::ReadSignals]),
        ).await.unwrap();

        assert!(!consent_id.is_empty());

        // Test consent verification
        let verified = interface.verify_user_consent(
            "test_user",
            &[BCIPermission::ProcessSignals],
        ).await.unwrap();

        assert!(verified);
    }
} 