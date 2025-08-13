//! Balance bridge for ArthaCoin integration
//! Provides compatibility layer between old balance system and ArthaCoin

use super::arthacoin_native::{balance_u128_to_u64, balance_u64_to_u128};
use super::ArthaCoinNative;
use anyhow::Result;
use std::sync::Arc;

/// Balance bridge for backward compatibility
/// This allows existing code to work with ArthaCoin through the old interface
#[derive(Debug)]
pub struct BalanceBridge {
    /// ArthaCoin native integration
    arthacoin: Arc<ArthaCoinNative>,
}

impl BalanceBridge {
    /// Create new balance bridge
    pub fn new(arthacoin: Arc<ArthaCoinNative>) -> Self {
        Self { arthacoin }
    }

    /// Get balance (compatible with old u64 interface)
    pub async fn get_balance(&self, address: &str) -> Result<u64> {
        let balance_u128 = self.arthacoin.get_balance(address).await?;
        Ok(balance_u128_to_u64(balance_u128))
    }

    /// Set balance (compatible with old u64 interface)
    pub async fn set_balance(&self, address: &str, amount: u64) -> Result<()> {
        let amount_u128 = balance_u64_to_u128(amount);
        self.arthacoin.set_balance(address, amount_u128).await
    }

    /// Transfer tokens (with ArthaCoin burn mechanics)
    pub async fn transfer(&self, from: &str, to: &str, amount: u64) -> Result<()> {
        let amount_u128 = balance_u64_to_u128(amount);
        self.arthacoin.transfer(from, to, amount_u128).await
    }

    /// Get balance in native ArthaCoin units (u128)
    pub async fn get_balance_native(&self, address: &str) -> Result<u128> {
        self.arthacoin.get_balance(address).await
    }

    /// Set balance in native ArthaCoin units (u128)
    pub async fn set_balance_native(&self, address: &str, amount: u128) -> Result<()> {
        self.arthacoin.set_balance(address, amount).await
    }

    /// Transfer in native ArthaCoin units (u128)
    pub async fn transfer_native(&self, from: &str, to: &str, amount: u128) -> Result<()> {
        self.arthacoin.transfer(from, to, amount).await
    }

    /// Get total supply (compatible interface)
    pub async fn get_total_supply(&self) -> u64 {
        let supply_u128 = self.arthacoin.get_total_supply().await;
        balance_u128_to_u64(supply_u128)
    }

    /// Get total supply in native units
    pub async fn get_total_supply_native(&self) -> u128 {
        self.arthacoin.get_total_supply().await
    }

    /// Get total burned
    pub async fn get_total_burned(&self) -> u128 {
        self.arthacoin.get_total_burned().await
    }

    /// Distribute validator rewards
    pub async fn distribute_validator_reward(&self, validator: &str, amount: u64) -> Result<()> {
        let amount_u128 = balance_u64_to_u128(amount);
        self.arthacoin
            .distribute_from_pool("validators_pool", validator, amount_u128)
            .await
    }

    /// Distribute staking rewards
    pub async fn distribute_staking_reward(&self, staker: &str, amount: u64) -> Result<()> {
        let amount_u128 = balance_u64_to_u128(amount);
        self.arthacoin
            .distribute_from_pool("staking_rewards_pool", staker, amount_u128)
            .await
    }

    /// Get pool balance
    pub async fn get_pool_balance(&self, pool_name: &str) -> Result<u64> {
        let balance_u128 = self.arthacoin.get_pool_balance(pool_name).await?;
        Ok(balance_u128_to_u64(balance_u128))
    }

    /// Mint cycle emission (for consensus/governance calls)
    pub async fn mint_cycle_emission(&self) -> Result<u64> {
        let emission_u128 = self.arthacoin.mint_cycle_emission().await?;
        Ok(balance_u128_to_u64(emission_u128))
    }

    /// Check if account has sufficient balance for amount
    pub async fn has_sufficient_balance(&self, address: &str, amount: u64) -> Result<bool> {
        let balance = self.get_balance(address).await?;
        Ok(balance >= amount)
    }

    /// Check if account has sufficient balance for amount in native units
    pub async fn has_sufficient_balance_native(&self, address: &str, amount: u128) -> Result<bool> {
        let balance = self.arthacoin.get_balance(address).await?;
        Ok(balance >= amount)
    }

    /// Deduct amount from account (for fees, etc.)
    pub async fn deduct(&self, address: &str, amount: u64) -> Result<()> {
        let current_balance = self.get_balance(address).await?;
        if current_balance < amount {
            return Err(anyhow::anyhow!("Insufficient balance"));
        }
        self.set_balance(address, current_balance - amount).await
    }

    /// Add amount to account
    pub async fn credit(&self, address: &str, amount: u64) -> Result<()> {
        let current_balance = self.get_balance(address).await?;
        self.set_balance(address, current_balance + amount).await
    }

    /// Get account info
    pub async fn get_account_info(&self, address: &str) -> Result<AccountInfo> {
        let balance = self.get_balance_native(address).await?;
        let balance_compat = balance_u128_to_u64(balance);

        Ok(AccountInfo {
            address: address.to_string(),
            balance_compat,
            balance_native: balance,
            is_pool: self.is_pool_address(address),
        })
    }

    /// Check if address is a pool address
    fn is_pool_address(&self, address: &str) -> bool {
        matches!(
            address,
            "validators_pool"
                | "staking_rewards_pool"
                | "ecosystem_grants_pool"
                | "marketing_wallet"
                | "developers_pool"
                | "dao_governance_pool"
                | "treasury_reserve"
        )
    }

    /// Get all pool balances
    pub async fn get_all_pool_balances(&self) -> Result<PoolBalances> {
        Ok(PoolBalances {
            validators_pool: self.arthacoin.get_pool_balance("validators_pool").await?,
            staking_rewards_pool: self
                .arthacoin
                .get_pool_balance("staking_rewards_pool")
                .await?,
            ecosystem_grants_pool: self
                .arthacoin
                .get_pool_balance("ecosystem_grants_pool")
                .await?,
            marketing_wallet: self.arthacoin.get_pool_balance("marketing_wallet").await?,
            developers_pool: self.arthacoin.get_pool_balance("developers_pool").await?,
            dao_governance_pool: self
                .arthacoin
                .get_pool_balance("dao_governance_pool")
                .await?,
            treasury_reserve: self.arthacoin.get_pool_balance("treasury_reserve").await?,
        })
    }
}

/// Account information
#[derive(Debug, Clone)]
pub struct AccountInfo {
    /// Account address
    pub address: String,
    /// Balance in compatible u64 format
    pub balance_compat: u64,
    /// Balance in native u128 format
    pub balance_native: u128,
    /// Whether this is a pool address
    pub is_pool: bool,
}

/// All pool balances
#[derive(Debug, Clone)]
pub struct PoolBalances {
    pub validators_pool: u128,
    pub staking_rewards_pool: u128,
    pub ecosystem_grants_pool: u128,
    pub marketing_wallet: u128,
    pub developers_pool: u128,
    pub dao_governance_pool: u128,
    pub treasury_reserve: u128,
}
