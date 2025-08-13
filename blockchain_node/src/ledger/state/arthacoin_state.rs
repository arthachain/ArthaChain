//! ArthaCoin-integrated state management
//! This replaces the simple balance tracking with ArthaCoin native integration

use crate::config::Config;
use crate::ledger::block::Block;
use crate::ledger::transaction::Transaction;
use crate::native_token::arthacoin_native::ArthaCoinConfig;
use crate::native_token::{ArthaCoinNative, BalanceBridge};
use anyhow::Result;
use log::{debug, info};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

/// ArthaCoin-integrated blockchain state
#[derive(Debug)]
pub struct ArthaCoinState {
    /// ArthaCoin native integration
    arthacoin: Arc<ArthaCoinNative>,
    /// Balance bridge for compatibility
    balance_bridge: Arc<BalanceBridge>,
    /// Account nonces
    nonces: RwLock<HashMap<String, u64>>,
    /// Contract storage
    storage: RwLock<HashMap<String, Vec<u8>>>,
    /// Current block height
    height: RwLock<u64>,
    /// Shard ID
    shard_id: u64,
    /// Pending transactions
    pending_transactions: RwLock<VecDeque<Transaction>>,
    /// Transaction history by account
    tx_history: RwLock<HashMap<String, Vec<String>>>,
    /// Blocks by height
    blocks: RwLock<HashMap<u64, Block>>,
    /// Blocks by hash
    blocks_by_hash: RwLock<HashMap<String, Block>>,
    /// Latest block hash
    latest_block_hash: RwLock<String>,
}

impl ArthaCoinState {
    /// Create new ArthaCoin-integrated state
    pub async fn new(config: &Config) -> Result<Self> {
        info!("Initializing ArthaCoin-integrated blockchain state");

        // Create ArthaCoin configuration
        let arthacoin_config = ArthaCoinConfig {
            contract_address: "0x0000000000000000000000000000000000000001".to_string(),
            initial_supply: 0, // Emission-based
            genesis_emission: if config.is_genesis {
                50_000_000 * 10_u128.pow(18)
            } else {
                0
            },
            gas_price: 20_000_000_000, // 20 gwei equivalent
            min_gas_limit: 21_000,
            max_gas_limit: 30_000_000,
        };

        // Initialize ArthaCoin
        let arthacoin = Arc::new(ArthaCoinNative::new(arthacoin_config).await?);
        let balance_bridge = Arc::new(BalanceBridge::new(arthacoin.clone()));

        info!("ArthaCoin native currency initialized");
        if config.is_genesis {
            info!("Genesis emission: 50M ARTHA distributed to pools");
        }

        Ok(Self {
            arthacoin,
            balance_bridge,
            nonces: RwLock::new(HashMap::new()),
            storage: RwLock::new(HashMap::new()),
            height: RwLock::new(0),
            shard_id: config.sharding.shard_id,
            pending_transactions: RwLock::new(VecDeque::new()),
            tx_history: RwLock::new(HashMap::new()),
            blocks: RwLock::new(HashMap::new()),
            blocks_by_hash: RwLock::new(HashMap::new()),
            latest_block_hash: RwLock::new(
                "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            ),
        })
    }

    /// Get account balance (ArthaCoin integration)
    pub async fn get_balance(&self, address: &str) -> Result<u64> {
        self.balance_bridge.get_balance(address).await
    }

    /// Set account balance (ArthaCoin integration)
    pub async fn set_balance(&self, address: &str, amount: u64) -> Result<()> {
        self.balance_bridge.set_balance(address, amount).await
    }

    /// Get account balance in native ArthaCoin units
    pub async fn get_balance_native(&self, address: &str) -> Result<u128> {
        self.balance_bridge.get_balance_native(address).await
    }

    /// Set account balance in native ArthaCoin units
    pub async fn set_balance_native(&self, address: &str, amount: u128) -> Result<()> {
        self.balance_bridge
            .set_balance_native(address, amount)
            .await
    }

    /// Transfer tokens (with ArthaCoin burn mechanics)
    pub async fn transfer(&self, from: &str, to: &str, amount: u64) -> Result<()> {
        debug!("ArthaCoin transfer: {} -> {} amount: {}", from, to, amount);
        self.balance_bridge.transfer(from, to, amount).await
    }

    /// Transfer tokens in native units
    pub async fn transfer_native(&self, from: &str, to: &str, amount: u128) -> Result<()> {
        self.balance_bridge.transfer_native(from, to, amount).await
    }

    /// Get account nonce (unchanged)
    pub fn get_nonce(&self, address: &str) -> Result<u64> {
        let nonces = self.nonces.read().unwrap();
        Ok(*nonces.get(address).unwrap_or(&0))
    }

    /// Set account nonce (unchanged)
    pub fn set_nonce(&self, address: &str, nonce: u64) -> Result<()> {
        let mut nonces = self.nonces.write().unwrap();
        nonces.insert(address.to_string(), nonce);
        Ok(())
    }

    /// Get next nonce for account
    pub fn get_next_nonce(&self, address: &str) -> Result<u64> {
        let current_nonce = self.get_nonce(address)?;
        Ok(current_nonce + 1)
    }

    /// Get storage value (unchanged)
    pub fn get_storage(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let storage = self.storage.read().unwrap();
        Ok(storage.get(key).cloned())
    }

    /// Set storage value (unchanged)
    pub fn set_storage(&self, key: &str, value: Vec<u8>) -> Result<()> {
        let mut storage = self.storage.write().unwrap();
        storage.insert(key.to_string(), value);
        Ok(())
    }

    /// Delete storage value (unchanged)
    pub fn delete_storage(&self, key: &str) -> Result<()> {
        let mut storage = self.storage.write().unwrap();
        storage.remove(key);
        Ok(())
    }

    /// Get current block height (unchanged)
    pub fn get_height(&self) -> Result<u64> {
        Ok(*self.height.read().unwrap())
    }

    /// Set current block height (unchanged)
    pub fn set_height(&self, height: u64) -> Result<()> {
        let mut h = self.height.write().unwrap();
        *h = height;
        Ok(())
    }

    /// Get shard ID (unchanged)
    pub fn get_shard_id(&self) -> Result<u64> {
        Ok(self.shard_id)
    }

    /// Check if account has sufficient balance
    pub async fn has_sufficient_balance(&self, address: &str, amount: u64) -> Result<bool> {
        self.balance_bridge
            .has_sufficient_balance(address, amount)
            .await
    }

    /// Deduct amount from account (for fees, etc.)
    pub async fn deduct_balance(&self, address: &str, amount: u64) -> Result<()> {
        self.balance_bridge.deduct(address, amount).await
    }

    /// Add amount to account
    pub async fn credit_balance(&self, address: &str, amount: u64) -> Result<()> {
        self.balance_bridge.credit(address, amount).await
    }

    /// Distribute validator reward
    pub async fn distribute_validator_reward(&self, validator: &str, amount: u64) -> Result<()> {
        self.balance_bridge
            .distribute_validator_reward(validator, amount)
            .await
    }

    /// Distribute staking reward
    pub async fn distribute_staking_reward(&self, staker: &str, amount: u64) -> Result<()> {
        self.balance_bridge
            .distribute_staking_reward(staker, amount)
            .await
    }

    /// Get pool balance
    pub async fn get_pool_balance(&self, pool_name: &str) -> Result<u64> {
        self.balance_bridge.get_pool_balance(pool_name).await
    }

    /// Mint cycle emission (called by consensus/governance)
    pub async fn mint_cycle_emission(&self) -> Result<u64> {
        info!("Minting new ArthaCoin emission cycle");
        self.balance_bridge.mint_cycle_emission().await
    }

    /// Get total supply
    pub async fn get_total_supply(&self) -> u64 {
        self.balance_bridge.get_total_supply().await
    }

    /// Get total supply in native units
    pub async fn get_total_supply_native(&self) -> u128 {
        self.balance_bridge.get_total_supply_native().await
    }

    /// Get total burned
    pub async fn get_total_burned(&self) -> u128 {
        self.balance_bridge.get_total_burned().await
    }

    /// Get account info
    pub async fn get_account_info(
        &self,
        address: &str,
    ) -> Result<crate::native_token::balance_bridge::AccountInfo> {
        self.balance_bridge.get_account_info(address).await
    }

    /// Get all pool balances
    pub async fn get_all_pool_balances(
        &self,
    ) -> Result<crate::native_token::balance_bridge::PoolBalances> {
        self.balance_bridge.get_all_pool_balances().await
    }

    /// Get ArthaCoin reference for advanced operations
    pub fn get_arthacoin(&self) -> Arc<ArthaCoinNative> {
        self.arthacoin.clone()
    }

    /// Get balance bridge reference
    pub fn get_balance_bridge(&self) -> Arc<BalanceBridge> {
        self.balance_bridge.clone()
    }

    // Block and transaction management (unchanged from original State)

    /// Add pending transaction
    pub fn add_pending_transaction(&self, transaction: Transaction) -> Result<()> {
        let mut pending = self.pending_transactions.write().unwrap();
        pending.push_back(transaction);
        Ok(())
    }

    /// Get pending transactions
    pub fn get_pending_transactions(&self) -> Result<Vec<Transaction>> {
        let pending = self.pending_transactions.read().unwrap();
        Ok(pending.iter().cloned().collect())
    }

    /// Clear pending transactions
    pub fn clear_pending_transactions(&self) -> Result<()> {
        let mut pending = self.pending_transactions.write().unwrap();
        pending.clear();
        Ok(())
    }

    /// Add transaction to history
    pub fn add_transaction_to_history(&self, account: &str, tx_hash: &str) -> Result<()> {
        let mut history = self.tx_history.write().unwrap();
        history
            .entry(account.to_string())
            .or_insert_with(Vec::new)
            .push(tx_hash.to_string());
        Ok(())
    }

    /// Get transaction history for account
    pub fn get_transaction_history(&self, account: &str) -> Result<Vec<String>> {
        let history = self.tx_history.read().unwrap();
        Ok(history.get(account).cloned().unwrap_or_default())
    }

    /// Add block
    pub fn add_block(&self, block: Block) -> Result<()> {
        let height = block.header.height;
        let hash = block.hash()?;
        let hash_str = hex::encode(hash.as_ref());

        let mut blocks = self.blocks.write().unwrap();
        let mut blocks_by_hash = self.blocks_by_hash.write().unwrap();
        let mut latest_hash = self.latest_block_hash.write().unwrap();

        blocks.insert(height, block.clone());
        blocks_by_hash.insert(hash_str.clone(), block);
        *latest_hash = hash_str;

        self.set_height(height)?;
        Ok(())
    }

    /// Get block by height
    pub fn get_block(&self, height: u64) -> Result<Option<Block>> {
        let blocks = self.blocks.read().unwrap();
        Ok(blocks.get(&height).cloned())
    }

    /// Get block by hash
    pub fn get_block_by_hash(&self, hash: &str) -> Result<Option<Block>> {
        let blocks = self.blocks_by_hash.read().unwrap();
        Ok(blocks.get(hash).cloned())
    }

    /// Get latest block hash
    pub fn get_latest_block_hash(&self) -> Result<String> {
        Ok(self.latest_block_hash.read().unwrap().clone())
    }

    /// Create state snapshot (simplified for ArthaCoin)
    pub async fn create_snapshot(&self) -> Result<u64> {
        // For ArthaCoin integration, we'll implement this later
        // For now, return a dummy snapshot ID
        Ok(0)
    }

    /// Commit snapshot (simplified)
    pub async fn commit_snapshot(&self, _snapshot_id: u64) -> Result<()> {
        // ArthaCoin has built-in transaction atomicity
        Ok(())
    }

    /// Revert to snapshot (simplified)
    pub async fn revert_to_snapshot(&self, _snapshot_id: u64) -> Result<()> {
        // ArthaCoin has built-in rollback capabilities
        Ok(())
    }
}
