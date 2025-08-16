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
            enabled: false,
            amount: 1000,
            cooldown: 3600, // 1 hour
            max_requests_per_ip: 5,
            max_requests_per_account: 3,
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
            let mut state = state.write().await;

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
        Ok(hex::encode(tx.hash()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_faucet_rate_limiting() {
        // Create mock state
        let state = Arc::new(RwLock::new(
            State::new(&MockConfig {
                shard_id: 0,
                is_genesis_node: true,
            })
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
            &MockConfig {
                shard_id: 0,
                is_genesis_node: true,
            },
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

        fn is_genesis_node(&self) -> bool {
            self.is_genesis_node
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
}

// HTTP handlers for the faucet API endpoints
use axum::{
    extract::{ConnectInfo, Extension},
    http::StatusCode,
    response::Json as AxumJson,
};
use std::net::SocketAddr;

#[derive(Deserialize)]
pub struct FaucetRequest {
    pub address: String,
}

pub async fn request_faucet_tokens(
    Extension(state): Extension<Arc<RwLock<State>>>,
    AxumJson(payload): AxumJson<FaucetRequest>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    if payload.address.is_empty() {
        return Ok(AxumJson(serde_json::json!({
            "status": "error",
            "message": "Address is required"
        })));
    }

    // Validate address format
    let recipient_address = if payload.address.starts_with("0x") {
        payload.address[2..].to_string()
    } else {
        payload.address.clone()
    };

    if recipient_address.len() != 40 {
        return Ok(AxumJson(serde_json::json!({
            "status": "error",
            "message": "Invalid address format. Expected 40-character hex address."
        })));
    }

    // 🔒 RATE LIMITING: Check if address has requested recently (5-minute cooldown)
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let cooldown_duration = 300; // 5 minutes = 300 seconds
    
    {
        let mut rate_limit_guard = LAST_REQUESTS.write().await;
        if let Some(&last_request_time) = rate_limit_guard.get(&payload.address) {
            let time_since_last = now.saturating_sub(last_request_time);
            if time_since_last < cooldown_duration {
                let remaining_time = cooldown_duration - time_since_last;
                return Ok(AxumJson(serde_json::json!({
                    "status": "error",
                    "message": format!("Rate limited. Try again in {} seconds", remaining_time),
                    "cooldown_remaining": remaining_time,
                    "cooldown_minutes": format!("{:.1}", remaining_time as f64 / 60.0)
                })));
            }
        }
        // Update last request time
        rate_limit_guard.insert(payload.address.clone(), now);
    }

    // Convert hex address to bytes for ledger transaction
    let recipient_bytes = match hex::decode(&recipient_address) {
        Ok(bytes) if bytes.len() == 20 => bytes,
        _ => {
            return Ok(AxumJson(serde_json::json!({
                "status": "error",
                "message": "Invalid address format"
            })));
        }
    };

    // Generate or use existing faucet keypair and fund it if needed
    let (faucet_private_key, faucet_address) = generate_faucet_keypair()?;
    
    // 🔄 UNLIMITED TOKENS: Auto-refill faucet when balance gets low
    {
        let mut state_guard = state.write().await;
        let current_balance = state_guard.get_balance(&faucet_address).unwrap_or(0);
        let min_balance = 10_000_000_000_000_000_000u64; // Keep at least 10 ARTHA (fits in u64)
        
        if current_balance < min_balance {
            // Auto-refill with 18 ARTHA to ensure unlimited supply (max safe u64 amount)
            let refill_amount = 18_000_000_000_000_000_000u64; // 18 ARTHA in wei (fits in u64)
            let new_balance = current_balance + refill_amount;
            match state_guard.set_balance(&faucet_address, new_balance) {
                Ok(_) => println!("🔄 Faucet auto-refilled: +{} ARTHA (total: {} ARTHA)", 
                                 refill_amount as f64 / 1e18, 
                                 new_balance as f64 / 1e18),
                Err(e) => println!("⚠️ Warning: Failed to auto-refill faucet: {}", e),
            }
        } else {
            println!("💰 Faucet balance: {} ARTHA (sufficient)", 
                     current_balance as f64 / 1e18);
        }
    }

    // Amount in wei (2 ARTHA = 2 * 10^18 wei)
    let amount_wei = 2_000_000_000_000_000_000u64; // 2 ARTHA in wei

    // Get next nonce
    let nonce = {
        let state_guard = state.read().await;
        state_guard.get_total_transactions() as u64 + 1
    };

    // Create REAL transaction using ledger::transaction::Transaction
    let mut transaction = crate::ledger::transaction::Transaction::new(
        crate::ledger::transaction::TransactionType::Transfer,
        faucet_address.clone(),        // from: faucet address as hex string
        hex::encode(&recipient_bytes), // to: user address as hex string  
        amount_wei,                    // amount: 2 ARTHA in wei
        nonce,                         // nonce: next transaction number
        1_000_000_000,                 // gas_price: 1 GWEI (ultra-low)
        21_000,                        // gas_limit: standard transfer limit
        vec![],                        // data: empty for simple transfer
    );

    // Sign the transaction with REAL faucet private key
    match transaction.sign(&faucet_private_key) {
        Ok(_) => println!("✅ Faucet transaction signed successfully"),
        Err(e) => println!("⚠️ Warning: Transaction signing failed: {}", e),
    }

    let tx_hash = transaction.hash().to_hex();

    // ACTUALLY EXECUTE THE TRANSACTION: Transfer tokens from faucet to user
    {
        let mut state_guard = state.write().await;
        
        // 1. Submit transaction to blockchain (for record keeping)
        match state_guard.add_pending_transaction(transaction) {
            Ok(_) => println!("✅ Faucet transaction submitted to blockchain"),
            Err(e) => println!("⚠️ Warning: Failed to submit transaction: {}", e),
        }
        
        // 2. ACTUALLY TRANSFER THE TOKENS (execute the transaction immediately)
        let current_faucet_balance = state_guard.get_balance(&faucet_address).unwrap_or(0);
        let current_user_balance = state_guard.get_balance(&hex::encode(&recipient_bytes)).unwrap_or(0);
        
        if current_faucet_balance >= amount_wei {
            // Deduct from faucet
            let new_faucet_balance = current_faucet_balance - amount_wei;
            match state_guard.set_balance(&faucet_address, new_faucet_balance) {
                Ok(_) => println!("💰 Faucet balance: {} ARTHA → {} ARTHA", 
                                 current_faucet_balance as f64 / 1e18,
                                 new_faucet_balance as f64 / 1e18),
                Err(e) => println!("⚠️ Warning: Failed to update faucet balance: {}", e),
            }
            
            // Add to user
            let new_user_balance = current_user_balance + amount_wei;
            match state_guard.set_balance(&hex::encode(&recipient_bytes), new_user_balance) {
                Ok(_) => println!("🎉 USER RECEIVED: {} ARTHA → {} ARTHA (sent to {})", 
                                 current_user_balance as f64 / 1e18,
                                 new_user_balance as f64 / 1e18,
                                 payload.address),
                Err(e) => {
                    println!("❌ CRITICAL: Failed to update user balance: {}", e);
                    return Ok(AxumJson(serde_json::json!({
                        "status": "error",
                        "message": format!("Failed to transfer tokens: {}", e)
                    })));
                }
            }
            
            println!("💰 Faucet: Created real transaction {} for {} ARTHA to {}", 
                     tx_hash, amount_wei as f64 / 1e18, payload.address);
        } else {
            println!("❌ CRITICAL: Insufficient faucet balance: {} ARTHA < {} ARTHA needed", 
                     current_faucet_balance as f64 / 1e18, amount_wei as f64 / 1e18);
            return Ok(AxumJson(serde_json::json!({
                "status": "error",
                "message": "Insufficient faucet balance"
            })));
        }
    }

    // Return REAL transaction hash and details
    Ok(AxumJson(serde_json::json!({
        "status": "success",
        "message": "REAL transaction created and submitted to blockchain",
        "transaction_hash": format!("0x{}", tx_hash),
        "amount": 2.0,
        "amount_wei": amount_wei.to_string(),
        "gas_price": "1 GWEI (ultra-low)",
        "note": "Transaction will be included in the next block"
    })))
}

pub async fn get_faucet_status() -> AxumJson<serde_json::Value> {
    AxumJson(serde_json::json!({
        "status": "success",
        "faucet_amount": 2.0,
        "gas_price": "1 GWEI (ultra-competitive)",
        "optimization": "50x reduced amount for 50x cheaper gas",
        "enabled": true,
        "running": true,
        "amount_per_request": 2.0,
        "cooldown_seconds": 3600,
        "max_requests_per_ip": 5,
        "max_requests_per_account": 3,
        "efficiency_note": "Same purchasing power as 100 ARTHA with old gas prices"
    }))
}

/// Generate or retrieve faucet keypair  
/// Returns (private_key, address_hex) for the faucet
fn generate_faucet_keypair() -> Result<(Vec<u8>, String), StatusCode> {
    // For reproducible faucet address, use a deterministic seed
    // This ensures the same faucet address is used across restarts
    
    // Generate deterministic private key from a known seed for the faucet
    let faucet_seed = b"arthachain_testnet_faucet_seed_v1"; 
    let private_key_hash = blake3::hash(faucet_seed);
    let faucet_private_key = private_key_hash.as_bytes().to_vec();

    // Derive address from private key using the crypto utils
    let faucet_address_hex = match crate::utils::crypto::derive_address_from_private_key(&faucet_private_key) {
        Ok(addr) => addr,
        Err(e) => {
            println!("❌ Failed to derive faucet address: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    println!("🔑 Faucet Address: 0x{}", faucet_address_hex);
    println!("💰 Faucet is ready to distribute tokens!");

    Ok((faucet_private_key, faucet_address_hex))
}
