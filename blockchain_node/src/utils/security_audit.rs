use anyhow::Result;
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Audit priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuditPriority {
    /// Critical priority - must be audited before production
    Critical,
    /// High priority - should be audited before production
    High,
    /// Medium priority - audit recommended
    Medium,
    /// Low priority - audit optional
    Low,
}

/// Code component types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ComponentType {
    /// Consensus algorithm code
    Consensus,
    /// Smart contract execution environment
    SmartContract,
    /// P2P networking code
    Network,
    /// Data storage code
    Storage,
    /// Cryptographic primitives and operations
    Cryptography,
    /// Authentication and authorization
    Auth,
    /// State management and validation
    State,
    /// API endpoints and interfaces
    API,
    /// AI security modules
    AI,
}

/// Audit target information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTarget {
    /// Component type
    pub component_type: ComponentType,
    /// Path to the module
    pub path: String,
    /// Function or section name
    pub name: String,
    /// Brief description of what this code does
    pub description: String,
    /// Security implications
    pub security_implications: String,
    /// Audit priority
    pub priority: AuditPriority,
    /// Whether this has been audited
    pub audited: bool,
    /// Audit report reference (if audited)
    pub audit_report: Option<String>,
    /// Audit date (if audited)
    pub audit_date: Option<String>,
}

/// Security audit registry that tracks what needs to be audited
pub struct SecurityAuditRegistry {
    /// Audit targets by component type
    targets: Arc<RwLock<HashMap<ComponentType, Vec<AuditTarget>>>>,
}

impl SecurityAuditRegistry {
    /// Create a new security audit registry
    pub fn new() -> Self {
        Self {
            targets: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a component for audit
    pub async fn register_target(&self, target: AuditTarget) -> Result<()> {
        let mut targets = self.targets.write().await;

        let component_targets = targets
            .entry(target.component_type)
            .or_insert_with(Vec::new);

        component_targets.push(target);

        Ok(())
    }

    /// Mark a target as audited
    pub async fn mark_audited(
        &self,
        component_type: ComponentType,
        path: &str,
        name: &str,
        report: &str,
        date: &str,
    ) -> Result<bool> {
        let mut targets = self.targets.write().await;

        if let Some(component_targets) = targets.get_mut(&component_type) {
            for target in component_targets.iter_mut() {
                if target.path == path && target.name == name {
                    target.audited = true;
                    target.audit_report = Some(report.to_string());
                    target.audit_date = Some(date.to_string());
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get all targets for a component type
    pub async fn get_targets_by_type(&self, component_type: ComponentType) -> Vec<AuditTarget> {
        let targets = self.targets.read().await;

        targets.get(&component_type).cloned().unwrap_or_default()
    }

    /// Get all targets
    pub async fn get_all_targets(&self) -> Vec<AuditTarget> {
        let targets = self.targets.read().await;

        targets.values().flatten().cloned().collect()
    }

    /// Get unaudited targets by priority
    pub async fn get_unaudited_by_priority(&self, priority: AuditPriority) -> Vec<AuditTarget> {
        let targets = self.targets.read().await;

        targets
            .values()
            .flatten()
            .filter(|t| !t.audited && t.priority <= priority)
            .cloned()
            .collect()
    }

    /// Generate audit report in Markdown format
    pub async fn generate_audit_report(&self) -> String {
        let targets = self.targets.read().await;
        let mut report = String::new();

        report.push_str("# Security Audit Status Report\n\n");
        report.push_str(&format!(
            "Generated on: {}\n\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        ));

        let all_targets: Vec<_> = targets.values().flatten().collect();
        let total = all_targets.len();
        let audited = all_targets.iter().filter(|t| t.audited).count();
        let percentage = if total > 0 {
            (audited as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        report.push_str(&format!("## Summary\n\n"));
        report.push_str(&format!("- Total components: {}\n", total));
        report.push_str(&format!("- Audited: {} ({:.1}%)\n", audited, percentage));
        report.push_str(&format!("- Unaudited: {}\n\n", total - audited));

        // Critical unaudited components
        let critical_unaudited: Vec<_> = all_targets
            .iter()
            .filter(|t| !t.audited && t.priority == AuditPriority::Critical)
            .collect();

        if !critical_unaudited.is_empty() {
            report.push_str("## ⚠️ Critical Unaudited Components\n\n");

            for target in critical_unaudited {
                report.push_str(&format!(
                    "- **{}**: {} ({})\n",
                    target.name, target.description, target.path
                ));
                report.push_str(&format!(
                    "  - Security implications: {}\n\n",
                    target.security_implications
                ));
            }
        }

        // Component breakdown
        report.push_str("## Component Status\n\n");

        for (component_type, component_targets) in targets.iter() {
            report.push_str(&format!("### {:?}\n\n", component_type));

            if component_targets.is_empty() {
                report.push_str("No components registered.\n\n");
                continue;
            }

            report.push_str("| Component | Path | Priority | Status | Audit Date |\n");
            report.push_str("|-----------|------|----------|--------|------------|\n");

            for target in component_targets {
                let status = if target.audited {
                    "✅ Audited"
                } else {
                    "❌ Unaudited"
                };

                report.push_str(&format!(
                    "| {} | {} | {:?} | {} | {} |\n",
                    target.name,
                    target.path,
                    target.priority,
                    status,
                    target.audit_date.as_deref().unwrap_or("N/A")
                ));
            }

            report.push_str("\n");
        }

        report
    }

    /// Export audit targets to JSON
    pub async fn export_to_json(&self) -> Result<String> {
        let targets = self.get_all_targets().await;
        let json = serde_json::to_string_pretty(&targets)?;
        Ok(json)
    }

    /// Import audit targets from JSON
    pub async fn import_from_json(&self, json: &str) -> Result<()> {
        let targets: Vec<AuditTarget> = serde_json::from_str(json)?;

        for target in targets {
            self.register_target(target).await?;
        }

        Ok(())
    }
}

// Pre-populate with critical consensus and contract components that need auditing
pub async fn initialize_audit_registry() -> Result<SecurityAuditRegistry> {
    let registry = SecurityAuditRegistry::new();

    // Register SVCP components
    registry
        .register_target(AuditTarget {
            component_type: ComponentType::Consensus,
            path: "blockchain_node/src/consensus/svcp.rs".to_string(),
            name: "SVCPMiner::mine_block".to_string(),
            description: "Block mining and PoW verification".to_string(),
            security_implications: "Critical for consensus security, vulnerable to timing attacks"
                .to_string(),
            priority: AuditPriority::Critical,
            audited: false,
            audit_report: None,
            audit_date: None,
        })
        .await?;

    registry
        .register_target(AuditTarget {
            component_type: ComponentType::Consensus,
            path: "blockchain_node/src/consensus/svcp.rs".to_string(),
            name: "SVCPMiner::update_proposer_candidates".to_string(),
            description: "Selection of block proposers based on node scores".to_string(),
            security_implications:
                "Critical for consensus fairness, potential for manipulation of proposer selection"
                    .to_string(),
            priority: AuditPriority::Critical,
            audited: false,
            audit_report: None,
            audit_date: None,
        })
        .await?;

    // Register Smart Contract components
    registry
        .register_target(AuditTarget {
            component_type: ComponentType::SmartContract,
            path: "blockchain_node/src/wasm/executor.rs".to_string(),
            name: "WasmExecutor::execute".to_string(),
            description: "Execution of WASM smart contracts".to_string(),
            security_implications:
                "Contract execution sandbox, potential for resource exhaustion attacks".to_string(),
            priority: AuditPriority::Critical,
            audited: false,
            audit_report: None,
            audit_date: None,
        })
        .await?;

    registry
        .register_target(AuditTarget {
            component_type: ComponentType::SmartContract,
            path: "blockchain_node/src/evm/executor.rs".to_string(),
            name: "EvmExecutor::execute_transaction".to_string(),
            description: "Execution of EVM transactions".to_string(),
            security_implications: "Gas metering, reentrancy protection, contract state isolation"
                .to_string(),
            priority: AuditPriority::Critical,
            audited: false,
            audit_report: None,
            audit_date: None,
        })
        .await?;

    // Register Cryptography components
    registry
        .register_target(AuditTarget {
            component_type: ComponentType::Cryptography,
            path: "blockchain_node/src/utils/crypto.rs".to_string(),
            name: "verify_signature".to_string(),
            description: "Signature verification for transactions and blocks".to_string(),
            security_implications:
                "Transaction and block authenticity, potential for signature forgery".to_string(),
            priority: AuditPriority::Critical,
            audited: false,
            audit_report: None,
            audit_date: None,
        })
        .await?;

    // Register AI components
    registry.register_target(AuditTarget {
        component_type: ComponentType::AI,
        path: "blockchain_node/src/ai_engine/security.rs".to_string(),
        name: "SecurityAI::evaluate_node".to_string(),
        description: "AI-based evaluation of node trustworthiness".to_string(),
        security_implications: "Node scoring affects consensus participation, potential for gaming the score system".to_string(),
        priority: AuditPriority::High,
        audited: false,
        audit_report: None,
        audit_date: None,
    }).await?;

    info!("Security audit registry initialized with critical components");

    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_registry() {
        let registry = initialize_audit_registry().await.unwrap();

        // Verify initial registry state
        let all_targets = registry.get_all_targets().await;
        assert!(!all_targets.is_empty());

        // Get unaudited critical targets
        let critical_targets = registry
            .get_unaudited_by_priority(AuditPriority::Critical)
            .await;
        assert!(!critical_targets.is_empty());

        // Mark a target as audited
        let result = registry
            .mark_audited(
                ComponentType::Consensus,
                "blockchain_node/src/consensus/svcp.rs",
                "SVCPMiner::mine_block",
                "Audit report #123",
                "2023-09-15",
            )
            .await
            .unwrap();

        assert!(result);

        // Verify target was marked as audited
        let consensus_targets = registry.get_targets_by_type(ComponentType::Consensus).await;
        let audited_target = consensus_targets
            .iter()
            .find(|t| t.name == "SVCPMiner::mine_block")
            .unwrap();

        assert!(audited_target.audited);
        assert_eq!(
            audited_target.audit_report,
            Some("Audit report #123".to_string())
        );

        // Generate report
        let report = registry.generate_audit_report().await;
        assert!(report.contains("Security Audit Status Report"));
    }
}
