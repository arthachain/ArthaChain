use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// Gas-Free Application Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasFreeApp {
    /// Unique identifier for the gas-free application
    pub app_id: String,
    /// Company/Project name
    pub company_name: String,
    /// Application type
    pub app_type: GasFreeAppType,
    /// Gas-free duration (in seconds, 0 = permanent)
    pub duration: u64,
    /// Start timestamp
    pub start_time: u64,
    /// End timestamp (0 = permanent)
    pub end_time: u64,
    /// Maximum gas-free transactions per day
    pub max_tx_per_day: u64,
    /// Current daily transaction count
    pub daily_tx_count: u64,
    /// Last reset timestamp
    pub last_reset: u64,
    /// Gas-free limit per transaction (in wei)
    pub gas_limit_per_tx: u64,
    /// Allowed transaction types
    pub allowed_tx_types: Vec<String>,
    /// Company signature for verification
    pub company_signature: Vec<u8>,
    /// Status
    pub is_active: bool,
    /// Created timestamp
    pub created_at: u64,
}

/// Types of Gas-Free Applications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GasFreeAppType {
    /// Complete gas-free for the duration
    CompletelyFree,
    /// Discounted gas (percentage)
    Discounted { percentage: u8 },
    /// Gas-free up to a limit
    LimitedFree { max_gas: u64 },
    /// Gas-free for specific operations
    SelectiveFree { operations: Vec<String> },
}

/// Gas-Free Transaction Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasFreeTxRequest {
    pub app_id: String,
    pub from_address: Vec<u8>,
    pub to_address: Vec<u8>,
    pub data: Vec<u8>,
    pub value: u64,
    pub gas_limit: u64,
    pub tx_type: String,
    pub timestamp: u64,
}

/// Gas-Free Manager
pub struct GasFreeManager {
    /// Active gas-free applications
    apps: Arc<RwLock<HashMap<String, GasFreeApp>>>,
    /// Daily transaction counters
    daily_counters: Arc<RwLock<HashMap<String, u64>>>,
    /// Company whitelist (hardcoded for security)
    company_whitelist: Vec<String>,
}

impl GasFreeManager {
    /// Create new gas-free manager
    pub fn new() -> Self {
        let company_whitelist = vec![
            "ArthaChain".to_string(),
            "ArthaCorp".to_string(),
            "ArthaLabs".to_string(),
            "ArthaVentures".to_string(),
        ];

        Self {
            apps: Arc::new(RwLock::new(HashMap::new())),
            daily_counters: Arc::new(RwLock::new(HashMap::new())),
            company_whitelist,
        }
    }

    /// Register a new gas-free application (company only)
    pub async fn register_app(&self, app: GasFreeApp) -> Result<bool> {
        // Verify company is whitelisted
        if !self.company_whitelist.contains(&app.company_name) {
            return Ok(false);
        }

        // Verify signature (simplified for demo)
        if !self.verify_company_signature(&app) {
            return Ok(false);
        }

        let mut apps = self.apps.write().await;
        apps.insert(app.app_id.clone(), app);
        
        Ok(true)
    }

    /// Check if transaction is eligible for gas-free
    pub async fn is_gas_free_eligible(&self, request: &GasFreeTxRequest) -> Result<Option<GasFreeApp>> {
        let apps = self.apps.read().await;
        
        if let Some(app) = apps.get(&request.app_id) {
            if !app.is_active {
                return Ok(None);
            }

            // Check if within time bounds
            let current_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            if app.end_time > 0 && current_time > app.end_time {
                return Ok(None);
            }

            // Check daily limit
            if !self.check_daily_limit(&request.app_id).await? {
                return Ok(None);
            }

            // Check transaction type
            if !app.allowed_tx_types.contains(&request.tx_type) {
                return Ok(None);
            }

            // Check gas limit
            if request.gas_limit > app.gas_limit_per_tx {
                return Ok(None);
            }

            Ok(Some(app.clone()))
        } else {
            Ok(None)
        }
    }

    /// Process gas-free transaction
    pub async fn process_gas_free_tx(&self, request: &GasFreeTxRequest) -> Result<bool> {
        if let Some(app) = self.is_gas_free_eligible(request).await? {
            // Increment daily counter
            self.increment_daily_counter(&request.app_id).await?;
            
            // Log the gas-free transaction
            println!("🚀 Gas-Free Transaction Processed:");
            println!("   App ID: {}", app.app_id);
            println!("   Company: {}", app.company_name);
            println!("   Type: {:?}", app.app_type);
            println!("   From: {:?}", request.from_address);
            println!("   To: {:?}", request.to_address);
            println!("   Value: {} wei", request.value);
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get all active gas-free applications
    pub async fn get_active_apps(&self) -> Result<Vec<GasFreeApp>> {
        let apps = self.apps.read().await;
        let active_apps: Vec<GasFreeApp> = apps
            .values()
            .filter(|app| app.is_active)
            .cloned()
            .collect();
        
        Ok(active_apps)
    }

    /// Update application status
    pub async fn update_app_status(&self, app_id: &str, is_active: bool) -> Result<bool> {
        let mut apps = self.apps.write().await;
        
        if let Some(app) = apps.get_mut(app_id) {
            app.is_active = is_active;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Reset daily counters (called daily)
    pub async fn reset_daily_counters(&self) -> Result<()> {
        let mut counters = self.daily_counters.write().await;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Reset counters older than 24 hours
        counters.retain(|_, last_reset| {
            current_time - *last_reset < 86400 // 24 hours
        });

        Ok(())
    }

    /// Check daily transaction limit
    async fn check_daily_limit(&self, app_id: &str) -> Result<bool> {
        let apps = self.apps.read().await;
        let counters = self.daily_counters.read().await;
        
        if let Some(app) = apps.get(app_id) {
            let current_count = counters.get(app_id).unwrap_or(&0);
            Ok(*current_count < app.max_tx_per_day)
        } else {
            Ok(false)
        }
    }

    /// Increment daily counter
    async fn increment_daily_counter(&self, app_id: &str) -> Result<()> {
        let mut counters = self.daily_counters.write().await;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let count = counters.entry(app_id.to_string()).or_insert(0);
        *count += 1;

        // Update last reset time
        if let Some(app) = self.apps.read().await.get(app_id) {
            if current_time - app.last_reset >= 86400 {
                *count = 1; // Reset counter
            }
        }

        Ok(())
    }

    /// Verify company signature (simplified for demo)
    fn verify_company_signature(&self, app: &GasFreeApp) -> bool {
        // In production, this would verify cryptographic signatures
        // For demo, we'll just check if signature is not empty
        !app.company_signature.is_empty()
    }

    /// Create demo gas-free applications
    pub async fn create_demo_apps(&self) -> Result<()> {
        let demo_apps = vec![
            GasFreeApp {
                app_id: "demo_product_1".to_string(),
                company_name: "ArthaChain".to_string(),
                app_type: GasFreeAppType::CompletelyFree,
                duration: 86400 * 30, // 30 days
                start_time: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                end_time: 0, // Permanent for demo
                max_tx_per_day: 1000,
                daily_tx_count: 0,
                last_reset: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                gas_limit_per_tx: 21_000, // Standard ETH gas limit
                allowed_tx_types: vec!["transfer".to_string(), "contract_call".to_string()],
                company_signature: vec![0x01, 0x02, 0x03, 0x04],
                is_active: true,
                created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            },
            GasFreeApp {
                app_id: "enterprise_project_1".to_string(),
                company_name: "ArthaCorp".to_string(),
                app_type: GasFreeAppType::Discounted { percentage: 50 },
                duration: 86400 * 90, // 90 days
                start_time: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                end_time: 0,
                max_tx_per_day: 5000,
                daily_tx_count: 0,
                last_reset: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                gas_limit_per_tx: 100_000,
                allowed_tx_types: vec!["transfer".to_string(), "contract_deploy".to_string()],
                company_signature: vec![0x05, 0x06, 0x07, 0x08],
                is_active: true,
                created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            },
        ];

        for app in demo_apps {
            self.register_app(app).await?;
        }

        Ok(())
    }
}

impl Default for GasFreeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gas_free_app_registration() {
        let manager = GasFreeManager::new();
        
        let app = GasFreeApp {
            app_id: "test_app".to_string(),
            company_name: "ArthaChain".to_string(),
            app_type: GasFreeAppType::CompletelyFree,
            duration: 86400,
            start_time: 0,
            end_time: 0,
            max_tx_per_day: 100,
            daily_tx_count: 0,
            last_reset: 0,
            gas_limit_per_tx: 21000,
            allowed_tx_types: vec!["transfer".to_string()],
            company_signature: vec![0x01],
            is_active: true,
            created_at: 0,
        };

        let result = manager.register_app(app).await.unwrap();
        assert!(result);
    }

    #[tokio::test]
    async fn test_gas_free_eligibility() {
        let manager = GasFreeManager::new();
        
        // Create demo apps first
        manager.create_demo_apps().await.unwrap();

        let request = GasFreeTxRequest {
            app_id: "demo_product_1".to_string(),
            from_address: vec![0x01],
            to_address: vec![0x02],
            data: vec![],
            value: 1000,
            gas_limit: 21000,
            tx_type: "transfer".to_string(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };

        let result = manager.is_gas_free_eligible(&request).await.unwrap();
        assert!(result.is_some());
    }
}
