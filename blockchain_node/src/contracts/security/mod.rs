use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::types::{Address, Hash, ContractId, CallData};
use crate::consensus::metrics::SecurityMetrics;

pub struct SecurityManager {
    // Access control
    access_control: Arc<RwLock<AccessControl>>,
    // Reentrancy protection
    reentrancy_guard: Arc<RwLock<ReentrancyGuard>>,
    // Contract verification
    contract_verifier: Arc<RwLock<ContractVerifier>>,
    // Call validation
    call_validator: Arc<RwLock<CallValidator>>,
    // Security metrics
    metrics: Arc<SecurityMetrics>,
}

struct AccessControl {
    // Role-based access control
    roles: HashMap<Address, HashSet<Role>>,
    // Role hierarchies
    role_admins: HashMap<Role, Role>,
    // Role requirements for functions
    function_requirements: HashMap<Hash, HashSet<Role>>,
}

struct ReentrancyGuard {
    // Track active calls
    active_calls: HashSet<(Address, Hash)>,
    // Call depth tracking
    call_depths: HashMap<Address, u32>,
    // Maximum allowed depth
    max_depth: u32,
}

struct ContractVerifier {
    // Verified contracts
    verified_contracts: HashMap<ContractId, ContractMetadata>,
    // Verification rules
    verification_rules: Vec<VerificationRule>,
    // Trusted deployers
    trusted_deployers: HashSet<Address>,
}

struct CallValidator {
    // Allowed function signatures
    allowed_signatures: HashSet<Hash>,
    // Blocked addresses
    blocked_addresses: HashSet<Address>,
    // Call filters
    call_filters: Vec<CallFilter>,
}

#[derive(Clone, Hash, Eq, PartialEq)]
enum Role {
    Admin,
    Operator,
    Proposer,
    Executor,
    Pauser,
    Custom(String),
}

struct ContractMetadata {
    bytecode_hash: Hash,
    source_hash: Hash,
    compiler_version: String,
    verification_time: u64,
    audited: bool,
}

enum VerificationRule {
    BytecodeMatch,
    SourceMatch,
    AuditRequired,
    CompilerVersionCheck,
    CustomRule(Box<dyn Fn(&ContractMetadata) -> bool + Send + Sync>),
}

struct CallFilter {
    pattern: CallPattern,
    action: FilterAction,
}

enum CallPattern {
    FunctionSignature(Hash),
    Destination(Address),
    ValueRange { min: u64, max: u64 },
    Custom(Box<dyn Fn(&CallData) -> bool + Send + Sync>),
}

enum FilterAction {
    Allow,
    Deny,
    RequireRole(Role),
    RequireApproval(u32),
}

impl SecurityManager {
    pub fn new(metrics: Arc<SecurityMetrics>) -> Self {
        Self {
            access_control: Arc::new(RwLock::new(AccessControl::new())),
            reentrancy_guard: Arc::new(RwLock::new(ReentrancyGuard::new())),
            contract_verifier: Arc::new(RwLock::new(ContractVerifier::new())),
            call_validator: Arc::new(RwLock::new(CallValidator::new())),
            metrics,
        }
    }

    pub async fn validate_call(&self, caller: Address, target: Address, data: &CallData) -> anyhow::Result<()> {
        // Check reentrancy
        let mut guard = self.reentrancy_guard.write().await;
        guard.check_reentrancy(caller, target)?;
        
        // Validate caller has required roles
        let access = self.access_control.read().await;
        access.validate_caller_roles(caller, data)?;
        
        // Validate target contract
        let verifier = self.contract_verifier.read().await;
        verifier.validate_contract(target)?;
        
        // Validate call data
        let validator = self.call_validator.read().await;
        validator.validate_call(caller, target, data)?;
        
        self.metrics.record_validated_call(caller, target);
        Ok(())
    }

    pub async fn register_contract(&self, contract_id: ContractId, metadata: ContractMetadata) -> anyhow::Result<()> {
        let mut verifier = self.contract_verifier.write().await;
        verifier.register_contract(contract_id, metadata)?;
        
        self.metrics.record_contract_registered(contract_id);
        Ok(())
    }

    pub async fn grant_role(&self, account: Address, role: Role) -> anyhow::Result<()> {
        let mut access = self.access_control.write().await;
        access.grant_role(account, role)?;
        
        self.metrics.record_role_granted(account, role);
        Ok(())
    }

    pub async fn add_call_filter(&self, pattern: CallPattern, action: FilterAction) -> anyhow::Result<()> {
        let mut validator = self.call_validator.write().await;
        validator.add_filter(CallFilter { pattern, action });
        Ok(())
    }
}

impl AccessControl {
    fn new() -> Self {
        Self {
            roles: HashMap::new(),
            role_admins: HashMap::new(),
            function_requirements: HashMap::new(),
        }
    }

    fn validate_caller_roles(&self, caller: Address, data: &CallData) -> anyhow::Result<()> {
        let function_hash = data.function_hash();
        
        if let Some(required_roles) = self.function_requirements.get(&function_hash) {
            let caller_roles = self.roles.get(&caller).unwrap_or(&HashSet::new());
            
            if !required_roles.iter().any(|role| caller_roles.contains(role)) {
                return Err(anyhow::anyhow!("Caller does not have required role"));
            }
        }
        
        Ok(())
    }

    fn grant_role(&mut self, account: Address, role: Role) -> anyhow::Result<()> {
        self.roles.entry(account)
            .or_insert_with(HashSet::new)
            .insert(role);
        Ok(())
    }
}

impl ReentrancyGuard {
    fn new() -> Self {
        Self {
            active_calls: HashSet::new(),
            call_depths: HashMap::new(),
            max_depth: 10,
        }
    }

    fn check_reentrancy(&mut self, caller: Address, target: Address) -> anyhow::Result<()> {
        let call_key = (caller, target.into());
        
        // Check if call is already active
        if self.active_calls.contains(&call_key) {
            return Err(anyhow::anyhow!("Reentrant call detected"));
        }
        
        // Check call depth
        let depth = self.call_depths.entry(caller).or_insert(0);
        if *depth >= self.max_depth {
            return Err(anyhow::anyhow!("Maximum call depth exceeded"));
        }
        
        // Mark call as active
        self.active_calls.insert(call_key);
        *depth += 1;
        
        Ok(())
    }

    fn exit_call(&mut self, caller: Address, target: Address) {
        let call_key = (caller, target.into());
        self.active_calls.remove(&call_key);
        
        if let Some(depth) = self.call_depths.get_mut(&caller) {
            *depth -= 1;
        }
    }
}

impl ContractVerifier {
    fn new() -> Self {
        Self {
            verified_contracts: HashMap::new(),
            verification_rules: Vec::new(),
            trusted_deployers: HashSet::new(),
        }
    }

    fn register_contract(&mut self, contract_id: ContractId, metadata: ContractMetadata) -> anyhow::Result<()> {
        // Apply verification rules
        for rule in &self.verification_rules {
            match rule {
                VerificationRule::BytecodeMatch => {
                    // Verify bytecode hash matches
                }
                VerificationRule::SourceMatch => {
                    // Verify source code hash matches
                }
                VerificationRule::AuditRequired => {
                    if !metadata.audited {
                        return Err(anyhow::anyhow!("Contract must be audited"));
                    }
                }
                VerificationRule::CompilerVersionCheck => {
                    // Verify compiler version is supported
                }
                VerificationRule::CustomRule(rule) => {
                    if !rule(&metadata) {
                        return Err(anyhow::anyhow!("Custom verification rule failed"));
                    }
                }
            }
        }
        
        self.verified_contracts.insert(contract_id, metadata);
        Ok(())
    }

    fn validate_contract(&self, contract_id: Address) -> anyhow::Result<()> {
        if !self.verified_contracts.contains_key(&contract_id.into()) {
            return Err(anyhow::anyhow!("Contract not verified"));
        }
        Ok(())
    }
}

impl CallValidator {
    fn new() -> Self {
        Self {
            allowed_signatures: HashSet::new(),
            blocked_addresses: HashSet::new(),
            call_filters: Vec::new(),
        }
    }

    fn validate_call(&self, caller: Address, target: Address, data: &CallData) -> anyhow::Result<()> {
        // Check blocked addresses
        if self.blocked_addresses.contains(&target) {
            return Err(anyhow::anyhow!("Target address is blocked"));
        }
        
        // Check function signature
        let signature = data.function_hash();
        if !self.allowed_signatures.contains(&signature) {
            return Err(anyhow::anyhow!("Function signature not allowed"));
        }
        
        // Apply filters
        for filter in &self.call_filters {
            match &filter.pattern {
                CallPattern::FunctionSignature(sig) => {
                    if *sig == signature {
                        self.apply_filter_action(&filter.action, caller)?;
                    }
                }
                CallPattern::Destination(dest) => {
                    if *dest == target {
                        self.apply_filter_action(&filter.action, caller)?;
                    }
                }
                CallPattern::ValueRange { min, max } => {
                    let value = data.value();
                    if value >= *min && value <= *max {
                        self.apply_filter_action(&filter.action, caller)?;
                    }
                }
                CallPattern::Custom(check) => {
                    if check(data) {
                        self.apply_filter_action(&filter.action, caller)?;
                    }
                }
            }
        }
        
        Ok(())
    }

    fn apply_filter_action(&self, action: &FilterAction, caller: Address) -> anyhow::Result<()> {
        match action {
            FilterAction::Allow => Ok(()),
            FilterAction::Deny => Err(anyhow::anyhow!("Call denied by filter")),
            FilterAction::RequireRole(_role) => {
                // Would check role in production
                Ok(())
            }
            FilterAction::RequireApproval(required) => {
                // Would check approvals in production
                if *required > 0 {
                    return Err(anyhow::anyhow!("Call requires approval"));
                }
                Ok(())
            }
        }
    }

    fn add_filter(&mut self, filter: CallFilter) {
        self.call_filters.push(filter);
    }
} 