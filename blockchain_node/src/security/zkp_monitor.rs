use crate::utils::security_logger::{SecurityLogger, SecurityLevel, SecurityCategory};
use crate::crypto::zkp::{ZKProof, VerificationResult};
use anyhow::{Result, Context};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use log::{warn, error, debug, info};

/// Types of ZKP verification issues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ZKPVerificationIssue {
    /// Invalid proof structure
    InvalidProofStructure,
    /// Failed verification with specific error
    VerificationFailure(String),
    /// Unusual proof patterns (potential attack vector)
    UnusualPattern,
    /// Replay attack detected
    ReplayAttack,
    /// Performance attack (proof takes too long to verify)
    PerformanceAttack,
}

/// Statistics for ZKP verification
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ZKPStats {
    /// Number of proofs verified
    pub total_verified: usize,
    /// Number of successful verifications
    pub successful: usize,
    /// Number of failed verifications
    pub failed: usize,
    /// Average verification time in milliseconds
    pub avg_verification_time_ms: f64,
    /// Maximum verification time in milliseconds
    pub max_verification_time_ms: f64,
    /// Issues encountered by type
    pub issues_by_type: HashMap<String, usize>,
}

/// Tracks ZKP verification for security monitoring
pub struct ZKPSecurityMonitor {
    /// Security logger
    security_logger: Arc<SecurityLogger>,
    /// Statistics
    stats: RwLock<HashMap<String, ZKPStats>>,
    /// Seen proof nonces (for replay protection)
    seen_proofs: RwLock<HashMap<String, u64>>,
    /// Performance thresholds
    slow_threshold_ms: u64,
    critical_threshold_ms: u64,
}

impl ZKPSecurityMonitor {
    /// Create a new ZKP security monitor
    pub fn new(security_logger: Arc<SecurityLogger>) -> Self {
        Self {
            security_logger,
            stats: RwLock::new(HashMap::new()),
            seen_proofs: RwLock::new(HashMap::new()),
            slow_threshold_ms: 500, // 500ms is slow
            critical_threshold_ms: 2000, // 2 seconds is critical
        }
    }
    
    /// Monitor and log a ZKP verification
    pub async fn monitor_verification(
        &self,
        proof_type: &str,
        proof: &ZKProof,
        result: &VerificationResult,
        verification_time_ms: u64,
    ) -> Result<Vec<ZKPVerificationIssue>> {
        let mut issues = Vec::new();
        
        // Update statistics
        let mut stats = self.stats.write().await;
        let stat_entry = stats.entry(proof_type.to_string()).or_insert_with(ZKPStats::default);
        stat_entry.total_verified += 1;
        
        // Update verification time statistics
        let prev_avg = stat_entry.avg_verification_time_ms;
        let prev_count = stat_entry.total_verified as f64 - 1.0;
        stat_entry.avg_verification_time_ms = 
            (prev_avg * prev_count + verification_time_ms as f64) / stat_entry.total_verified as f64;
        
        // Track max verification time
        if verification_time_ms as f64 > stat_entry.max_verification_time_ms {
            stat_entry.max_verification_time_ms = verification_time_ms as f64;
        }
        
        // Check result
        match result {
            VerificationResult::Valid => {
                stat_entry.successful += 1;
                
                // Check for performance issues
                if verification_time_ms > self.critical_threshold_ms {
                    let issue = ZKPVerificationIssue::PerformanceAttack;
                    issues.push(issue.clone());
                    self.log_issue(proof_type, &issue, proof.nonce(), verification_time_ms).await?;
                    
                    // Increment issue count
                    let issue_str = format!("{:?}", issue);
                    *stat_entry.issues_by_type.entry(issue_str).or_insert(0) += 1;
                } else if verification_time_ms > self.slow_threshold_ms {
                    debug!("Slow ZKP verification for {}: {}ms", proof_type, verification_time_ms);
                }
                
                // Check for replay attacks
                let mut seen = self.seen_proofs.write().await;
                if let Some(previous_time) = seen.get(&proof.nonce().to_string()) {
                    let issue = ZKPVerificationIssue::ReplayAttack;
                    issues.push(issue.clone());
                    self.log_issue(proof_type, &issue, proof.nonce(), verification_time_ms).await?;
                    
                    // Increment issue count
                    let issue_str = format!("{:?}", issue);
                    *stat_entry.issues_by_type.entry(issue_str).or_insert(0) += 1;
                } else {
                    // Store nonce with timestamp
                    seen.insert(proof.nonce().to_string(), chrono::Utc::now().timestamp() as u64);
                }
            }
            VerificationResult::Invalid(error) => {
                stat_entry.failed += 1;
                
                let issue = ZKPVerificationIssue::VerificationFailure(error.clone());
                issues.push(issue.clone());
                self.log_issue(proof_type, &issue, proof.nonce(), verification_time_ms).await?;
                
                // Increment issue count
                let issue_str = format!("VerificationFailure");
                *stat_entry.issues_by_type.entry(issue_str).or_insert(0) += 1;
            }
        }
        
        // Prune old nonces periodically (keep nonces for up to 24 hours)
        if stat_entry.total_verified % 1000 == 0 {
            self.prune_old_nonces().await;
        }
        
        Ok(issues)
    }
    
    /// Log a ZKP verification issue
    async fn log_issue(
        &self,
        proof_type: &str,
        issue: &ZKPVerificationIssue,
        nonce: u64,
        verification_time_ms: u64,
    ) -> Result<()> {
        // Determine severity based on issue type
        let level = match issue {
            ZKPVerificationIssue::ReplayAttack | 
            ZKPVerificationIssue::PerformanceAttack => SecurityLevel::Critical,
            ZKPVerificationIssue::VerificationFailure(_) => SecurityLevel::Warning,
            _ => SecurityLevel::Info,
        };
        
        // Log the issue
        self.security_logger.log_event(
            level,
            SecurityCategory::ZeroKnowledgeProof,
            None, // No specific node ID in this case
            &format!("ZKP verification issue: {:?}", issue),
            serde_json::json!({
                "proof_type": proof_type,
                "nonce": nonce,
                "verification_time_ms": verification_time_ms,
                "issue": format!("{:?}", issue),
            }),
        ).await.context("Failed to log ZKP verification issue")?;
        
        Ok(())
    }
    
    /// Get statistics for a specific proof type
    pub async fn get_stats(&self, proof_type: &str) -> Option<ZKPStats> {
        self.stats.read().await.get(proof_type).cloned()
    }
    
    /// Get all statistics
    pub async fn get_all_stats(&self) -> HashMap<String, ZKPStats> {
        self.stats.read().await.clone()
    }
    
    /// Prune old nonces (older than 24 hours)
    async fn prune_old_nonces(&self) {
        let now = chrono::Utc::now().timestamp() as u64;
        let cutoff = now - 24 * 60 * 60; // 24 hours in seconds
        
        let mut seen = self.seen_proofs.write().await;
        seen.retain(|_, timestamp| *timestamp >= cutoff);
        
        debug!("Pruned old ZKP nonces, {} remain", seen.len());
    }
    
    /// Set performance thresholds
    pub fn set_thresholds(&mut self, slow_threshold_ms: u64, critical_threshold_ms: u64) {
        self.slow_threshold_ms = slow_threshold_ms;
        self.critical_threshold_ms = critical_threshold_ms;
    }
    
    /// Reset all statistics
    pub async fn reset_stats(&self) {
        self.stats.write().await.clear();
        info!("ZKP security statistics reset");
    }
    
    /// Generate a security report for ZKP verifications
    pub async fn generate_report(&self) -> String {
        let stats = self.stats.read().await;
        let mut report = String::from("# ZKP Security Monitoring Report\n\n");
        
        if stats.is_empty() {
            report.push_str("No ZKP verification statistics available.\n");
            return report;
        }
        
        report.push_str("## Summary\n\n");
        report.push_str("| Proof Type | Total | Success | Failed | Avg Time (ms) | Max Time (ms) |\n");
        report.push_str("|------------|-------|---------|--------|---------------|---------------|\n");
        
        let mut total_issues = 0;
        
        for (proof_type, stat) in stats.iter() {
            report.push_str(&format!(
                "| {} | {} | {} | {} | {:.2} | {:.2} |\n",
                proof_type,
                stat.total_verified,
                stat.successful,
                stat.failed,
                stat.avg_verification_time_ms,
                stat.max_verification_time_ms
            ));
            
            total_issues += stat.issues_by_type.values().sum::<usize>();
        }
        
        if total_issues > 0 {
            report.push_str("\n## Issues Detected\n\n");
            
            for (proof_type, stat) in stats.iter() {
                if !stat.issues_by_type.is_empty() {
                    report.push_str(&format!("### {}\n\n", proof_type));
                    report.push_str("| Issue Type | Count |\n");
                    report.push_str("|------------|-------|\n");
                    
                    for (issue_type, count) in stat.issues_by_type.iter() {
                        report.push_str(&format!("| {} | {} |\n", issue_type, count));
                    }
                    
                    report.push_str("\n");
                }
            }
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[tokio::test]
    async fn test_zkp_monitor() -> Result<()> {
        // Create a temporary file for testing
        let temp_dir = tempfile::tempdir()?;
        let log_path = temp_dir.path().join("test_security.log");
        
        // Create a security logger
        let security_logger = Arc::new(SecurityLogger::new(
            log_path.to_str().unwrap(),
            100
        )?);
        
        // Create ZKP monitor
        let monitor = ZKPSecurityMonitor::new(security_logger);
        
        // Create a mock proof and result
        let proof = ZKProof::mock(1234); // Assuming a mock implementation exists
        let valid_result = VerificationResult::Valid;
        let invalid_result = VerificationResult::Invalid("Test failure".to_string());
        
        // Test successful verification
        let issues = monitor.monitor_verification(
            "test_proof",
            &proof,
            &valid_result,
            100 // 100ms
        ).await?;
        
        assert!(issues.is_empty());
        
        // Test failed verification
        let issues = monitor.monitor_verification(
            "test_proof",
            &proof,
            &invalid_result,
            50 // 50ms
        ).await?;
        
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0], ZKPVerificationIssue::VerificationFailure(_)));
        
        // Test replay attack
        let issues = monitor.monitor_verification(
            "test_proof",
            &proof,
            &valid_result,
            30 // 30ms
        ).await?;
        
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0], ZKPVerificationIssue::ReplayAttack));
        
        // Test performance attack
        let proof2 = ZKProof::mock(5678);
        let issues = monitor.monitor_verification(
            "test_proof",
            &proof2,
            &valid_result,
            3000 // 3000ms - over critical threshold
        ).await?;
        
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0], ZKPVerificationIssue::PerformanceAttack));
        
        // Get stats
        let stats = monitor.get_stats("test_proof").await.unwrap();
        assert_eq!(stats.total_verified, 4);
        assert_eq!(stats.successful, 3);
        assert_eq!(stats.failed, 1);
        
        Ok(())
    }
}

/// Mock implementation of ZKProof for testing
#[cfg(test)]
impl ZKProof {
    pub fn mock(nonce: u64) -> Self {
        Self { nonce }
    }
    
    pub fn nonce(&self) -> u64 {
        self.nonce
    }
}

/// Mock implementation of ZKProof
#[cfg(not(test))]
impl ZKProof {
    pub fn nonce(&self) -> u64 {
        self.nonce
    }
} 