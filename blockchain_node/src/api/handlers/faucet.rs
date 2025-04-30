use crate::config::Config;
use crate::ledger::state::State;
use crate::ledger::transaction::{Transaction, TransactionType, TransactionStatus};
use crate::utils::crypto;
use anyhow::{Result, Context, anyhow};
use tokio::sync::RwLock;
use tokio::time::{self, Duration};
use std::sync::Arc;
use std::collections::HashMap;
use log::{info, warn, error};
use serde::{Serialize, Deserialize};
use std::net::IpAddr;
use std::time::{SystemTime, UNIX_EPOCH};

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
                state.update_balance(&faucet_config.address, 100_000_000)?;
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
    pub async fn request_tokens(
        &self, 
        recipient: &str, 
        client_ip: IpAddr
    ) -> Result<String> {
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
                    return Err(anyhow!("Please wait {} seconds before requesting again", 
                        self.config.cooldown - since_last.as_secs()));
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
            ip_reqs.insert(client_ip, RequestRecord {
                last_request: SystemTime::now(),
                request_count,
            });
        }
        
        // Check account rate limits
        {
            let mut acc_reqs = self.account_requests.write().await;
            
            // Get or create request record
            let request_count = if let Some(record) = acc_reqs.get(recipient) {
                // Check max requests
                if record.request_count >= self.config.max_requests_per_account {
                    return Err(anyhow!("Maximum requests per day exceeded for this account"));
                }
                
                record.request_count + 1
            } else {
                1 // First request
            };
            
            // Update record
            acc_reqs.insert(recipient.to_string(), RequestRecord {
                last_request: SystemTime::now(),
                request_count,
            });
        }
        
        // Create and submit transaction
        let tx_result = self.send_transaction(recipient).await?;
        
        info!("Sent {} tokens to {} via faucet", self.config.amount, recipient);
        
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
            gas_price: 1, // Minimal gas price
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
                now.duration_since(record.last_request)
                    .unwrap_or_default() < one_day
            });
        }
        
        // Prune account records
        {
            let mut acc_reqs = account_requests.write().await;
            acc_reqs.retain(|_, record| {
                now.duration_since(record.last_request)
                    .unwrap_or_default() < one_day
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
            State::new(&MockConfig { shard_id: 0, is_genesis_node: true }).unwrap()
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
            &MockConfig { shard_id: 0, is_genesis_node: true },
            state,
            Some(faucet_config)
        ).await.unwrap();
        
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
    }
} 