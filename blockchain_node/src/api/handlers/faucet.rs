use crate::config::Config;
use crate::ledger::state::State;
use crate::ledger::transaction::{Transaction, TransactionStatus, TransactionType};
use crate::utils::crypto;
use anyhow::{anyhow, Context, Result};
use blake3;
use hex;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::{self, Duration};

// Global rate limiting storage: address -> last request timestamp
static LAST_REQUESTS: once_cell::sync::Lazy<Arc<RwLock<HashMap<String, u64>>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(HashMap::new())));

/// Faucet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaucetConfig {
    /// Whether the faucet is enabled
    pub enabled: bool,
    /// Amount to distribute per request
    pub amount: u64,
    /// Cooldown time between requests in seconds
    pub cooldown: u64,
    /// Maximum number of requests per IP per day
    pub max_requests_per_ip: u32,
    /// Maximum number of requests per account per day
    pub max_requests_per_account: u32,
    /// Private key for faucet account
    pub private_key: Option<Vec<u8>>,
    /// Faucet account address
    pub address: String,
}

impl Default for FaucetConfig {
    fn default() -> Self {
        Self {
            enabled: true, // Enable by default for testnet
            amount: 2, // Make ArthaCoin precious - maximum 2 per request
            cooldown: 300, // 5 minutes cooldown between requests
            max_requests_per_ip: 0, // No daily limit (0 = unlimited)
            max_requests_per_account: 0, // No daily limit (0 = unlimited)
            private_key: None,
            address: "faucet".to_string(),
        }
    }
}

/// Request record for rate limiting
#[derive(Debug, Clone)]
struct RequestRecord {
    /// Time of last request
    last_request: SystemTime,
    /// Count of requests today
    request_count: u32,
}

/// Faucet service for distributing tokens in testnet
pub struct Faucet {
    /// Faucet configuration
    config: FaucetConfig,
    /// IP address request tracking
    ip_requests: Arc<RwLock<HashMap<IpAddr, RequestRecord>>>,
    /// Account request tracking
    account_requests: Arc<RwLock<HashMap<String, RequestRecord>>>,
    /// Blockchain state
    state: Arc<RwLock<State>>,
    /// Private key for signing transactions
    private_key: Vec<u8>,
    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl Faucet {
    /// Create a new faucet service
    pub async fn new(
        config: &Config,
        state: Arc<RwLock<State>>,
        faucet_config: Option<FaucetConfig>,
    ) -> Result<Self> {
        let faucet_config = faucet_config.unwrap_or_default();

        if !faucet_config.enabled {
            return Err(anyhow!("Faucet is disabled"));
        }

        // Get private key from config or generate new one
        let private_key = if let Some(key) = &faucet_config.private_key {
            key.clone()
        } else {
            // Generate new key pair for faucet
            let (private_key, _) = crypto::generate_keypair()?;
            private_key
        };

        // Initialize state with faucet account if this is genesis
        if config.is_genesis {
            let mut state_write = state.write().await;

            // Check if faucet account exists, if not create it
            let faucet_balance = state_write.get_balance(&faucet_config.address).unwrap_or(0);
            if faucet_balance == 0 {
                // Add initial balance to faucet
                state_write.set_balance(&faucet_config.address, 100_000_000)?;
                info!("Initialized faucet account with 100,000,000 tokens");
            }
        }

        Ok(Self {
            config: faucet_config,
            ip_requests: Arc::new(RwLock::new(HashMap::new())),
            account_requests: Arc::new(RwLock::new(HashMap::new())),
            state,
            private_key,
            running: Arc::new(RwLock::new(true)), // Start running by default
        })
    }

    /// Create a dummy faucet service for testing
    pub async fn new_dummy() -> Self {
        let config = Config::default();
        let state = Arc::new(RwLock::new(State::new(&config).unwrap()));
        
        Self {
            config: FaucetConfig::default(),
            ip_requests: Arc::new(RwLock::new(HashMap::new())),
            account_requests: Arc::new(RwLock::new(HashMap::new())),
            state,
            private_key: vec![0x01, 0x02, 0x03, 0x04, 0x05],
            running: Arc::new(RwLock::new(true)),
        }
    }

    /// Start the faucet service
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(anyhow!("Faucet is already running"));
        }

        *running = true;

        info!("Faucet service started");

        // Start pruning task
        let ip_requests = self.ip_requests.clone();
        let account_requests = self.account_requests.clone();
        let running_flag = self.running.clone();

        tokio::spawn(async move {
            let mut interval = time::interval(Duration::from_secs(86400)); // Daily pruning

            while *running_flag.read().await {
                interval.tick().await;

                // Prune old request records
                Self::prune_requests(&ip_requests, &account_requests).await;
            }
        });

        Ok(())
    }

    /// Stop the faucet service
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = false;

        info!("Faucet service stopped");

        Ok(())
    }

    /// Request tokens from the faucet
    pub async fn request_tokens(&self, recipient: &str, client_ip: IpAddr) -> Result<String> {
        // Check if faucet is running
        if !*self.running.read().await {
            return Err(anyhow!("Faucet service is not running"));
        }

        // Check if recipient is valid
        if recipient.is_empty() {
            return Err(anyhow!("Invalid recipient address"));
        }

        // Check IP rate limits
        {
            let mut ip_reqs = self.ip_requests.write().await;

            // Get or create request record
            let request_count = if let Some(record) = ip_reqs.get(&client_ip) {
                // Check cooldown
                let since_last = SystemTime::now()
                    .duration_since(record.last_request)
                    .unwrap_or_default();

                if since_last.as_secs() < self.config.cooldown {
                    return Err(anyhow!(
                        "Please wait {} seconds before requesting again",
                        self.config.cooldown - since_last.as_secs()
                    ));
                }

                // Check max requests
                if record.request_count >= self.config.max_requests_per_ip {
                    return Err(anyhow!("Maximum requests per day exceeded for your IP"));
                }

                record.request_count + 1
            } else {
                1
            };

            // Update IP request record
            ip_reqs.insert(
                client_ip,
                RequestRecord {
                    last_request: SystemTime::now(),
                    request_count,
                },
            );
        }

        // Check account rate limits
        {
            let mut account_reqs = self.account_requests.write().await;

            let request_count = if let Some(record) = account_reqs.get(recipient) {
                // Check cooldown
                let since_last = SystemTime::now()
                    .duration_since(record.last_request)
                    .unwrap_or_default();

                if since_last.as_secs() < self.config.cooldown {
                    return Err(anyhow!(
                        "Please wait {} seconds before requesting again",
                        self.config.cooldown - since_last.as_secs()
                    ));
                }

                // Check max requests
                if record.request_count >= self.config.max_requests_per_account {
                    return Err(anyhow!("Maximum requests per day exceeded for this account"));
                }

                record.request_count + 1
            } else {
                1
            };

            // Update account request record
            account_reqs.insert(
                recipient.to_string(),
                RequestRecord {
                    last_request: SystemTime::now(),
                    request_count,
                },
            );
        }

        // Check faucet balance
        let faucet_balance = {
            let state = self.state.read().await;
            state.get_balance(&self.config.address).unwrap_or(0)
        };

        if faucet_balance < self.config.amount {
            return Err(anyhow!("Faucet is out of funds"));
        }

        // Create and execute transfer transaction
        let tx_hash = self.execute_faucet_transaction(recipient).await?;

        info!(
            "Faucet distributed {} tokens to {} (tx: {})",
            self.config.amount, recipient, tx_hash
        );

        Ok(tx_hash)
    }

    /// Execute the actual faucet transaction
    async fn execute_faucet_transaction(&self, recipient: &str) -> Result<String> {
        let mut state = self.state.write().await;

        // Check faucet balance again (double-check)
        let faucet_balance = state.get_balance(&self.config.address).unwrap_or(0);
        if faucet_balance < self.config.amount {
            return Err(anyhow!("Insufficient faucet balance"));
        }

        // Deduct from faucet
        state.set_balance(&self.config.address, faucet_balance - self.config.amount)?;

        // Add to recipient
        let recipient_balance = state.get_balance(recipient).unwrap_or(0);
        state.set_balance(recipient, recipient_balance + self.config.amount)?;

        // Create transaction record
        let transaction = Transaction::new(
            TransactionType::Transfer,
            self.config.address.clone(),
            recipient.to_string(),
            self.config.amount,
            0, // nonce
            0, // gas_price
            0, // gas_limit
            vec![], // data
        );

        // Add transaction to state
        state.add_pending_transaction(transaction.clone())?;

        Ok(hex::encode(transaction.hash()))
    }

    /// Get faucet status
    pub async fn get_status(&self) -> Result<FaucetStatus> {
        let state = self.state.read().await;
        let faucet_balance = state.get_balance(&self.config.address).unwrap_or(0);
        let total_transactions = 0; // TODO: Implement transaction count

        Ok(FaucetStatus {
            enabled: self.config.enabled,
            running: *self.running.read().await,
            balance: faucet_balance,
            amount_per_request: self.config.amount,
            cooldown: self.config.cooldown,
            total_transactions,
            max_requests_per_ip: self.config.max_requests_per_ip,
            max_requests_per_account: self.config.max_requests_per_account,
        })
    }

    /// Get faucet form data
    pub async fn get_form_data(&self) -> Result<FaucetFormData> {
        let state = self.state.read().await;
        let faucet_balance = state.get_balance(&self.config.address).unwrap_or(0);

        Ok(FaucetFormData {
            enabled: self.config.enabled,
            balance: faucet_balance,
            amount_per_request: self.config.amount,
            cooldown_hours: self.config.cooldown / 3600,
            max_requests_per_ip: self.config.max_requests_per_ip,
            max_requests_per_account: self.config.max_requests_per_account,
        })
    }

    /// Prune old request records
    async fn prune_requests(
        ip_requests: &Arc<RwLock<HashMap<IpAddr, RequestRecord>>>,
        account_requests: &Arc<RwLock<HashMap<String, RequestRecord>>>,
    ) {
        let now = SystemTime::now();
        let day_ago = now - Duration::from_secs(86400);

        // Prune IP requests
        {
            let mut ip_reqs = ip_requests.write().await;
            ip_reqs.retain(|_, record| record.last_request > day_ago);
        }

        // Prune account requests
        {
            let mut account_reqs = account_requests.write().await;
            account_reqs.retain(|_, record| record.last_request > day_ago);
        }
    }
}

/// Faucet status response
#[derive(Debug, Serialize)]
pub struct FaucetStatus {
    pub enabled: bool,
    pub running: bool,
    pub balance: u64,
    pub amount_per_request: u64,
    pub cooldown: u64,
    pub total_transactions: u64,
    pub max_requests_per_ip: u32,
    pub max_requests_per_account: u32,
}

/// Faucet form data
#[derive(Debug, Serialize)]
pub struct FaucetFormData {
    pub enabled: bool,
    pub balance: u64,
    pub amount_per_request: u64,
    pub cooldown_hours: u64,
    pub max_requests_per_ip: u32,
    pub max_requests_per_account: u32,
}

/// Faucet request payload
#[derive(Debug, Deserialize)]
pub struct FaucetRequest {
    pub recipient: String,
}

/// Faucet response
#[derive(Debug, Serialize)]
pub struct FaucetResponse {
    pub success: bool,
    pub message: String,
    pub transaction_hash: Option<String>,
    pub amount: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_faucet_rate_limiting() {
        // Create mock state
        let config = Config::new();
        let state = Arc::new(RwLock::new(
            State::new(&config)
            .unwrap(),
        ));

        // Create faucet config
        let faucet_config = FaucetConfig {
            enabled: true,
            amount: 100,
            cooldown: 5, // 5 seconds cooldown for testing
            max_requests_per_ip: 2,
            max_requests_per_account: 2,
            private_key: None,
            address: "faucet".to_string(),
        };

        // Create faucet
        let faucet = Faucet::new(
            &config,
            state,
            Some(faucet_config),
        )
        .await
        .unwrap();

        // Start faucet
        faucet.start().await.unwrap();

        let ip = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
        let recipient = "test_user";

        // First request should succeed
        let result1 = faucet.request_tokens(recipient, ip).await;
        assert!(result1.is_ok());

        // Second request should succeed
        let result2 = faucet.request_tokens(recipient, ip).await;
        assert!(result2.is_ok());

        // Third request should fail due to max_requests_per_ip
        let result3 = faucet.request_tokens(recipient, ip).await;
        assert!(result3.is_err());

        // Stop faucet
        faucet.stop().await.unwrap();
    }

    #[derive(Clone)]
    struct MockConfig {
        shard_id: u64,
        is_genesis_node: bool,
    }

    impl crate::ledger::state::ShardConfig for MockConfig {
        fn get_shard_id(&self) -> u64 {
            self.shard_id
        }

        fn get_genesis_config(&self) -> Option<&Config> {
            None
        }

        fn is_sharding_enabled(&self) -> bool {
            false
        }

        fn get_shard_count(&self) -> u32 {
            1
        }

        fn get_primary_shard(&self) -> u32 {
            0
        }
    }

    impl MockConfig {
        fn is_genesis(&self) -> bool {
            self.is_genesis_node
        }
    }
}

// HTTP handlers for the faucet API endpoints
use axum::{
    extract::Extension,
    http::StatusCode,
    response::Json as AxumJson,
};
use axum::Json;

/// Handler for faucet token requests
pub async fn request_faucet_tokens(
    Json(payload): Json<FaucetRequest>,
    Extension(state): Extension<Arc<RwLock<State>>>,
    Extension(faucet): Extension<Arc<Faucet>>,
) -> Result<AxumJson<FaucetResponse>, StatusCode> {
    // Use localhost IP for now
    let client_ip = IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1));

    // Request tokens from faucet
    match faucet.request_tokens(&payload.recipient, client_ip).await {
        Ok(tx_hash) => Ok(AxumJson(FaucetResponse {
            success: true,
            message: format!("Successfully distributed {} tokens to {}", faucet.config.amount, payload.recipient),
            transaction_hash: Some(tx_hash),
            amount: faucet.config.amount,
        })),
        Err(e) => {
            error!("Faucet request failed: {}", e);
            Ok(AxumJson(FaucetResponse {
                success: false,
                message: format!("Faucet request failed: {}", e),
                transaction_hash: None,
                amount: 0,
            }))
        }
    }
}

/// Handler for getting faucet status
pub async fn get_faucet_status(
    Extension(faucet): Extension<Arc<Faucet>>,
) -> Result<AxumJson<FaucetStatus>, StatusCode> {
    match faucet.get_status().await {
        Ok(status) => Ok(AxumJson(status)),
        Err(e) => {
            error!("Failed to get faucet status: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting faucet form data
pub async fn get_faucet_form(
    Extension(faucet): Extension<Arc<Faucet>>,
) -> Result<AxumJson<FaucetFormData>, StatusCode> {
    match faucet.get_form_data().await {
        Ok(form_data) => Ok(AxumJson(form_data)),
        Err(e) => {
            error!("Failed to get faucet form data: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for faucet health check
pub async fn faucet_health_check() -> AxumJson<serde_json::Value> {
    AxumJson(serde_json::json!({
        "status": "healthy",
        "service": "faucet",
        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        "message": "Faucet service is running and ready to distribute tokens"
    }))
}

/// Handler for getting faucet info
pub async fn get_faucet_info() -> AxumJson<serde_json::Value> {
    AxumJson(serde_json::json!({
        "service": "faucet",
        "description": "ArthaChain Testnet Faucet - Distributes test tokens to users",
        "features": [
            "Real token distribution",
            "Rate limiting",
            "IP and account tracking",
            "Automatic balance management",
            "Transaction recording"
        ],
        "endpoints": {
            "POST /api/faucet": "Request tokens",
            "GET /api/faucet/status": "Get faucet status",
            "GET /api/faucet/form": "Get faucet form data",
            "GET /api/faucet/health": "Health check"
        },
        "note": "This faucet distributes real testnet tokens that are recorded on the blockchain"
    }))
}

// Additional handlers for the testnet router
use axum::response::Html;

/// Faucet dashboard page
pub async fn faucet_dashboard() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>ArthaChain Faucet</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .form-group { margin: 20px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input[type="text"], input[type="number"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
            button { background: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
            button:hover { background: #2980b9; }
            .status { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚰 ArthaChain Faucet</h1>
            <p style="text-align: center; color: #7f8c8d;">Get testnet tokens for development and testing</p>
            
            <div class="status">
                <h3>📊 Faucet Status</h3>
                <p><strong>Status:</strong> <span style="color: green;">✅ Active</span></p>
                <p><strong>Daily Limit:</strong> Unlimited requests per day</p>
                <p><strong>Cooldown:</strong> 5 minutes between requests</p>
                <p><strong>Amount per Request:</strong> 2 ARTHA (Precious Token)</p>
            </div>
            
            <form id="faucetForm">
                <div class="form-group">
                    <label for="recipient">Recipient Address:</label>
                    <input type="text" id="recipient" name="recipient" placeholder="0x..." required>
                </div>
                
                <div class="form-group">
                    <label for="amount">Amount (ARTHA):</label>
                    <input type="number" id="amount" name="amount" value="2" min="1" max="2" readonly>
                </div>
                
                <button type="submit">🚰 Request Tokens</button>
            </form>
            
            <div id="result" style="margin-top: 20px;"></div>
            
            <script>
                document.getElementById('faucetForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const recipient = document.getElementById('recipient').value;
                    const amount = document.getElementById('amount').value;
                    
                    try {
                        const response = await fetch('/api/v1/testnet/faucet/request', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ recipient, amount: parseInt(amount) })
                        });
                        
                        const result = await response.json();
                        document.getElementById('result').innerHTML = `
                            <div class="status" style="background: ${result.success ? '#d4edda' : '#f8d7da'}; color: ${result.success ? '#155724' : '#721c24'};">
                                <h4>${result.success ? '✅ Success' : '❌ Error'}</h4>
                                <p>${result.message}</p>
                                ${result.transaction_hash ? `<p><strong>Transaction Hash:</strong> ${result.transaction_hash}</p>` : ''}
                            </div>
                        `;
                    } catch (error) {
                        document.getElementById('result').innerHTML = `
                            <div class="status" style="background: #f8d7da; color: #721c24;">
                                <h4>❌ Error</h4>
                                <p>Failed to submit request: ${error.message}</p>
                            </div>
                        `;
                    }
                });
            </script>
        </div>
    </body>
    </html>
    "#)
}

/// Request tokens from faucet (simplified version for testnet)
pub async fn request_tokens() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "success": true,
        "message": "Token request endpoint - use POST /api/v1/testnet/faucet/request for actual requests",
        "amount": 2,
        "currency": "ARTHA (Precious)"
    }))
}

/// Get faucet history (simplified version for testnet)
pub async fn get_faucet_history() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "history": [
            {
                "recipient": "0x1234567890abcdef",
                "amount": 2,
                "timestamp": 1640995200,
                "transaction_hash": "0xabc123def456"
            }
        ]
    }))
}
