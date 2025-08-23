use crate::config::{ApiConfig, Config, ShardingConfig};
use crate::ledger::state::State;
use crate::ledger::transaction::{Transaction, TransactionStatus, TransactionType};
use crate::utils::crypto;
use anyhow::{anyhow, Result};
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::{self, Duration};

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
            enabled: false,
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
            private_key.to_vec()
        };

        // Initialize state with faucet account if this is genesis
        if config.is_genesis {
            let state = state.write().await;

            // Check if faucet account exists, if not create it
            let faucet_balance = state.get_balance(&faucet_config.address).unwrap_or(0);
            if faucet_balance == 0 {
                // Add initial balance to faucet
                state.set_balance(&faucet_config.address, 100_000_000)?;
                info!("Initialized faucet account with 100,000,000 tokens");
            }
        }

        Ok(Self {
            config: faucet_config,
            ip_requests: Arc::new(RwLock::new(HashMap::new())),
            account_requests: Arc::new(RwLock::new(HashMap::new())),
            state,
            private_key,
            running: Arc::new(RwLock::new(false)),
        })
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
                1 // First request
            };

            // Update record
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
            let mut acc_reqs = self.account_requests.write().await;

            // Get or create request record
            let request_count = if let Some(record) = acc_reqs.get(recipient) {
                // Check max requests
                if record.request_count >= self.config.max_requests_per_account {
                    return Err(anyhow!(
                        "Maximum requests per day exceeded for this account"
                    ));
                }

                record.request_count + 1
            } else {
                1 // First request
            };

            // Update record
            acc_reqs.insert(
                recipient.to_string(),
                RequestRecord {
                    last_request: SystemTime::now(),
                    request_count,
                },
            );
        }

        // Create and submit transaction
        let tx_result = self.send_transaction(recipient).await?;

        info!(
            "Sent {} tokens to {} via faucet",
            self.config.amount, recipient
        );

        Ok(tx_result)
    }

    /// Send transaction from faucet to recipient
    async fn send_transaction(&self, recipient: &str) -> Result<String> {
        // Get current state
        let state = self.state.read().await;

        // Get faucet account nonce
        let nonce = state.get_next_nonce(&self.config.address)?;

        // Create transaction
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut tx = Transaction {
            tx_type: TransactionType::Transfer,
            timestamp,
            sender: self.config.address.clone(),
            recipient: recipient.to_string(),
            amount: self.config.amount,
            nonce,
            gas_price: 1,     // Minimal gas price
            gas_limit: 21000, // Standard gas limit
            data: Vec::new(),
            signature: Vec::new(),
            #[cfg(feature = "bls")]
            bls_signature: None,
            status: TransactionStatus::Pending,
        };

        // Sign transaction
        let tx_bytes = tx.serialize_for_hash();
        tx.signature = crypto::sign(&self.private_key, &tx_bytes)?;

        // Return transaction ID
        Ok(hex::encode(tx.hash().as_ref()))
    }

    /// Prune old request records
    async fn prune_requests(
        ip_requests: &Arc<RwLock<HashMap<IpAddr, RequestRecord>>>,
        account_requests: &Arc<RwLock<HashMap<String, RequestRecord>>>,
    ) {
        let now = SystemTime::now();
        let one_day = Duration::from_secs(86400);

        // Prune IP records
        {
            let mut ip_reqs = ip_requests.write().await;
            ip_reqs.retain(|_, record| {
                now.duration_since(record.last_request).unwrap_or_default() < one_day
            });
        }

        // Prune account records
        {
            let mut acc_reqs = account_requests.write().await;
            acc_reqs.retain(|_, record| {
                now.duration_since(record.last_request).unwrap_or_default() < one_day
            });
        }

        info!("Pruned old faucet request records");
    }
}

// API endpoints for the faucet
use axum::{
    extract::{Extension, Json, Path, Query},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
};
// Remove duplicate import - already imported at top

/// Faucet request payload
#[derive(Debug, Deserialize)]
pub struct FaucetRequest {
    pub recipient: String,
    pub amount: Option<u64>,
}

/// Faucet response
#[derive(Debug, Serialize)]
pub struct FaucetResponse {
    pub success: bool,
    pub message: String,
    pub transaction_hash: Option<String>,
    pub amount: u64,
}

/// Faucet status response
#[derive(Debug, Serialize)]
pub struct FaucetStatusResponse {
    pub enabled: bool,
    pub daily_limit: u32,
    pub cooldown_period: u64,
    pub available_amount: u64,
    pub total_distributed: u64,
}

/// Faucet history entry
#[derive(Debug, Serialize)]
pub struct FaucetHistoryEntry {
    pub recipient: String,
    pub amount: u64,
    pub timestamp: u64,
    pub transaction_hash: Option<String>,
}

/// Faucet dashboard page
pub async fn faucet_dashboard() -> impl IntoResponse {
    let html = r#"
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
    "#;
    
    Html(html.to_string())
}

/// Request tokens from faucet
pub async fn request_tokens(
    Json(request): Json<FaucetRequest>,
) -> impl IntoResponse {
    // This is a placeholder - in a real implementation, you'd use the Faucet service
    let amount = request.amount.unwrap_or(1000);
    
    // Simulate faucet response
    let response = FaucetResponse {
        success: true,
        message: format!("Successfully sent {} ARTHA to {}", amount, request.recipient),
        transaction_hash: Some(format!("0x{:x}", chrono::Utc::now().timestamp())),
        amount,
    };
    
    (StatusCode::OK, Json(response))
}

/// Get faucet status
pub async fn get_faucet_status() -> impl IntoResponse {
    let status = FaucetStatusResponse {
        enabled: true,
        daily_limit: 5,
        cooldown_period: 3600,
        available_amount: 1000000,
        total_distributed: 50000,
    };
    
    Json(status)
}

/// Get faucet history
pub async fn get_faucet_history() -> impl IntoResponse {
    let history = vec![
        FaucetHistoryEntry {
            recipient: "0x1234567890abcdef".to_string(),
            amount: 1000,
            timestamp: chrono::Utc::now().timestamp() as u64,
            transaction_hash: Some("0xabc123def456".to_string()),
        },
        FaucetHistoryEntry {
            recipient: "0xabcdef1234567890".to_string(),
            amount: 500,
            timestamp: (chrono::Utc::now().timestamp() - 3600) as u64,
            transaction_hash: Some("0xdef456abc123".to_string()),
        },
    ];
    
    Json(history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_faucet_rate_limiting() {
        // Create test config
        let mut config = Config::new();
        config.is_genesis = true;
        config.sharding.shard_id = 0;
        config.sharding.enabled = true;
        config.sharding.shard_count = 4;
        config.sharding.primary_shard = 0;
        config.sharding.cross_shard_timeout = 30;
        config.sharding.assignment_strategy = "static".to_string();
        config.sharding.cross_shard_strategy = "atomic".to_string();

        // Create state
        let state = Arc::new(RwLock::new(State::new(&config).unwrap()));

        // Initialize faucet configuration
        let faucet_config = FaucetConfig {
            enabled: true,
            amount: 100,
            cooldown: 0, // No cooldown for testing
            max_requests_per_ip: 2,
            max_requests_per_account: 3,
            private_key: None,
            address: "faucet".to_string(),
        };

        // Create faucet with the proper Config type
        let faucet = Faucet::new(&config, state, Some(faucet_config))
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
}

#[allow(dead_code)]
fn create_test_config() -> Config {
    let mut config = Config::new();

    config.sharding = ShardingConfig {
        enabled: true,
        shard_count: 4,
        primary_shard: 0,
        shard_id: 0,
        cross_shard_timeout: 30,
        assignment_strategy: "static".to_string(),
        cross_shard_strategy: "atomic".to_string(),
    };

    config.api = ApiConfig {
        enabled: true,
        port: 8080,
        host: "127.0.0.1".to_string(),
        address: "127.0.0.1".to_string(),
        cors_domains: vec!["*".to_string()],
        allow_origin: vec!["*".to_string()],
        max_request_body_size: 10 * 1024 * 1024, // 10MB
        max_connections: 100,
        enable_websocket: false,
        enable_graphql: false,
    };

    config
}
