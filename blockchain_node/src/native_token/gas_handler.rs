//! Gas handling with ArthaCoin integration
//! Replaces simple gas fee deduction with ArthaCoin burn mechanics

use super::ArthaCoinNative;
use crate::ledger::transaction::Transaction;
use anyhow::{anyhow, Result};
use log::{debug, warn};
use std::sync::Arc;

/// Gas handler for ArthaCoin transactions
pub struct GasHandler {
    /// ArthaCoin native integration
    arthacoin: Arc<ArthaCoinNative>,
}

impl GasHandler {
    /// Create new gas handler
    pub fn new(arthacoin: Arc<ArthaCoinNative>) -> Self {
        Self { arthacoin }
    }

    /// Calculate transaction fee
    pub async fn calculate_fee(&self, transaction: &Transaction) -> u128 {
        let gas_price = if transaction.gas_price > 0 {
            transaction.gas_price as u128
        } else {
            self.arthacoin.get_gas_price()
        };

        transaction.gas_limit as u128 * gas_price
    }

    /// Validate gas parameters
    pub async fn validate_gas(&self, transaction: &Transaction) -> Result<()> {
        // Validate gas limit
        self.arthacoin.validate_gas_limit(transaction.gas_limit)?;

        // Validate gas price
        let min_gas_price = self.arthacoin.get_gas_price();
        if transaction.gas_price > 0 && (transaction.gas_price as u128) < min_gas_price {
            return Err(anyhow!("Gas price too low"));
        }

        Ok(())
    }

    /// Check if sender has sufficient balance for transaction + gas
    pub async fn check_sufficient_balance(&self, transaction: &Transaction) -> Result<bool> {
        let sender_balance = self.arthacoin.get_balance(&transaction.sender).await?;
        let gas_fee = self.calculate_fee(transaction).await;
        let total_required = transaction.amount as u128 + gas_fee;

        Ok(sender_balance >= total_required)
    }

    /// Execute gas payment with ArthaCoin burn mechanics
    pub async fn pay_gas(&self, transaction: &Transaction, gas_used: u64) -> Result<()> {
        let gas_price = if transaction.gas_price > 0 {
            transaction.gas_price as u128
        } else {
            self.arthacoin.get_gas_price()
        };

        debug!("Paying gas: {} units at {} price", gas_used, gas_price);

        // Pay gas with burn mechanics
        self.arthacoin
            .pay_gas(&transaction.sender, gas_used, gas_price)
            .await?;

        Ok(())
    }

    /// Estimate gas for transaction
    pub async fn estimate_gas(&self, transaction: &Transaction) -> Result<u64> {
        // Base gas costs
        let base_gas = 21000u64;

        // Additional gas based on transaction type
        let additional_gas = match transaction.tx_type {
            crate::ledger::transaction::TransactionType::Transfer => 0,
            crate::ledger::transaction::TransactionType::ContractCreate
            | crate::ledger::transaction::TransactionType::Deploy => 100000, // Contract deployment
            crate::ledger::transaction::TransactionType::ContractCall
            | crate::ledger::transaction::TransactionType::Call => 50000, // Contract call
            crate::ledger::transaction::TransactionType::Stake
            | crate::ledger::transaction::TransactionType::Delegate => 75000, // Staking operations
            crate::ledger::transaction::TransactionType::Unstake
            | crate::ledger::transaction::TransactionType::Undelegate => 75000,
            crate::ledger::transaction::TransactionType::ClaimReward
            | crate::ledger::transaction::TransactionType::ClaimRewards => 50000,
            crate::ledger::transaction::TransactionType::ValidatorRegistration => 150000,
            crate::ledger::transaction::TransactionType::Batch => {
                // Estimate based on data size
                transaction.data.len() as u64 * 10
            }
            _ => 25000, // Default for other transaction types
        };

        // Data gas cost
        let data_gas = transaction.data.len() as u64 * 4; // 4 gas per byte

        let estimated_gas = base_gas + additional_gas + data_gas;

        debug!("Estimated gas for transaction: {}", estimated_gas);
        Ok(estimated_gas)
    }

    /// Calculate maximum fee for transaction
    pub async fn calculate_max_fee(&self, transaction: &Transaction) -> Result<u128> {
        let gas_price = if transaction.gas_price > 0 {
            transaction.gas_price as u128
        } else {
            self.arthacoin.get_gas_price()
        };

        Ok(transaction.gas_limit as u128 * gas_price)
    }

    /// Refund unused gas
    pub async fn refund_gas(&self, transaction: &Transaction, gas_used: u64) -> Result<()> {
        if gas_used >= transaction.gas_limit {
            return Ok(()); // No refund needed
        }

        let gas_refund = transaction.gas_limit - gas_used;
        let gas_price = if transaction.gas_price > 0 {
            transaction.gas_price as u128
        } else {
            self.arthacoin.get_gas_price()
        };

        let refund_amount = gas_refund as u128 * gas_price;

        if refund_amount > 0 {
            let current_balance = self.arthacoin.get_balance(&transaction.sender).await?;
            self.arthacoin
                .set_balance(&transaction.sender, current_balance + refund_amount)
                .await?;

            debug!(
                "Refunded {} ARTHA gas to {}",
                refund_amount as f64 / 10_f64.powi(18),
                transaction.sender
            );
        }

        Ok(())
    }

    /// Get current gas price
    pub async fn get_gas_price(&self) -> u128 {
        self.arthacoin.get_gas_price()
    }

    /// Get gas statistics
    pub async fn get_gas_stats(&self) -> Result<GasStats> {
        Ok(GasStats {
            current_gas_price: self.arthacoin.get_gas_price(),
            total_burned_for_gas: self.arthacoin.get_total_burned().await,
            total_supply: self.arthacoin.get_total_supply().await,
        })
    }
}

/// Gas statistics
#[derive(Debug, Clone)]
pub struct GasStats {
    /// Current gas price in ArthaCoin
    pub current_gas_price: u128,
    /// Total ArthaCoin burned for gas fees
    pub total_burned_for_gas: u128,
    /// Total ArthaCoin supply
    pub total_supply: u128,
}
