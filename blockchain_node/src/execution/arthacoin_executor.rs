//! ArthaCoin-integrated transaction executor
//! This replaces simple balance operations with ArthaCoin native features

use crate::execution::ExecutionResult;
use crate::ledger::state::arthacoin_state::ArthaCoinState;
use crate::ledger::transaction::{Transaction, TransactionStatus, TransactionType};
use crate::native_token::GasHandler;
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use std::sync::Arc;

/// Transaction executor with ArthaCoin integration
pub struct ArthaCoinExecutor {
    /// Gas handler for ArthaCoin
    gas_handler: Arc<GasHandler>,
    /// Minimum gas price
    min_gas_price: u128,
    /// Maximum gas limit
    max_gas_limit: u64,
}

impl ArthaCoinExecutor {
    /// Create new ArthaCoin executor
    pub fn new(gas_handler: Arc<GasHandler>) -> Self {
        Self {
            gas_handler,
            min_gas_price: 20_000_000_000, // 20 gwei
            max_gas_limit: 30_000_000,
        }
    }

    /// Execute transaction with ArthaCoin integration
    pub async fn execute_transaction(
        &self,
        transaction: &mut Transaction,
        state: &ArthaCoinState,
    ) -> Result<ExecutionResult> {
        info!("Executing ArthaCoin transaction: {:?}", transaction.tx_type);

        // Validate transaction
        if let Err(e) = self.validate_transaction(transaction, state).await {
            transaction.status = TransactionStatus::Failed(e.to_string());
            return Ok(ExecutionResult::ValidationError(e.to_string()));
        }

        // Estimate gas usage
        let estimated_gas = self.gas_handler.estimate_gas(transaction).await?;
        let gas_to_use = estimated_gas.min(transaction.gas_limit);

        // Check sufficient balance for transaction + gas
        if !self
            .gas_handler
            .check_sufficient_balance(transaction)
            .await?
        {
            transaction.status = TransactionStatus::Failed("Insufficient balance".into());
            return Ok(ExecutionResult::InsufficientBalance);
        }

        // Execute transaction based on type
        let result = match transaction.tx_type {
            TransactionType::Transfer => {
                self.execute_transfer(transaction, state, gas_to_use)
                    .await?
            }
            TransactionType::ContractCreate | TransactionType::Deploy => {
                self.execute_deploy(transaction, state, gas_to_use).await?
            }
            TransactionType::Call | TransactionType::ContractCall => {
                self.execute_contract_call(transaction, state, gas_to_use)
                    .await?
            }
            TransactionType::ValidatorRegistration => {
                self.execute_validator_registration(transaction, state, gas_to_use)
                    .await?
            }
            TransactionType::Stake | TransactionType::Delegate => {
                self.execute_stake(transaction, state, gas_to_use).await?
            }
            TransactionType::Unstake | TransactionType::Undelegate => {
                self.execute_unstake(transaction, state, gas_to_use).await?
            }
            TransactionType::ClaimReward | TransactionType::ClaimRewards => {
                self.execute_claim_reward(transaction, state, gas_to_use)
                    .await?
            }
            TransactionType::Batch => self.execute_batch(transaction, state, gas_to_use).await?,
            TransactionType::System => self.execute_system(transaction, state, gas_to_use).await?,
            TransactionType::SetValidator => {
                self.execute_validator_registration(transaction, state, gas_to_use)
                    .await?
            }
            TransactionType::Custom(_) => {
                self.execute_system(transaction, state, gas_to_use).await?
            }
        };

        // Pay gas fees with ArthaCoin burn mechanics
        if let Err(e) = self.gas_handler.pay_gas(transaction, gas_to_use).await {
            warn!("Failed to pay gas: {}", e);
            transaction.status = TransactionStatus::Failed("Gas payment failed".into());
            return Ok(ExecutionResult::Failure("Gas payment failed".into()));
        }

        // Refund unused gas
        if let Err(e) = self.gas_handler.refund_gas(transaction, gas_to_use).await {
            warn!("Failed to refund gas: {}", e);
        }

        // Update transaction state based on result
        match &result {
            ExecutionResult::Success => {
                transaction.status = TransactionStatus::Success;
                info!("Transaction executed successfully");
            }
            ExecutionResult::Failure(reason) => {
                transaction.status = TransactionStatus::Failed(reason.clone());
                error!("Transaction failed: {}", reason);
            }
            ExecutionResult::ValidationError(reason) => {
                transaction.status = TransactionStatus::Failed(reason.clone());
                error!("Transaction validation failed: {}", reason);
            }
            ExecutionResult::InsufficientBalance => {
                transaction.status = TransactionStatus::Failed("Insufficient balance".into());
                error!("Transaction failed: Insufficient balance");
            }
            ExecutionResult::Reverted(reason) => {
                transaction.status = TransactionStatus::Failed(format!("Reverted: {}", reason));
                error!("Transaction reverted: {}", reason);
            }
            ExecutionResult::OutOfGas => {
                transaction.status = TransactionStatus::Failed("Out of gas".into());
                error!("Transaction failed: Out of gas");
            }
            ExecutionResult::InvalidNonce => {
                transaction.status = TransactionStatus::Failed("Invalid nonce".into());
                error!("Transaction failed: Invalid nonce");
            }
        }

        Ok(result)
    }

    /// Validate transaction with ArthaCoin checks
    async fn validate_transaction(
        &self,
        transaction: &Transaction,
        state: &ArthaCoinState,
    ) -> Result<()> {
        // Validate gas parameters
        self.gas_handler.validate_gas(transaction).await?;

        // Validate nonce
        let expected_nonce = state.get_next_nonce(&transaction.sender)?;
        if transaction.nonce != expected_nonce {
            return Err(anyhow!(
                "Invalid nonce: expected {}, got {}",
                expected_nonce,
                transaction.nonce
            ));
        }

        // Check if sender has sufficient balance
        if !self
            .gas_handler
            .check_sufficient_balance(transaction)
            .await?
        {
            return Err(anyhow!("Insufficient balance for transaction + gas"));
        }

        Ok(())
    }

    /// Execute transfer with ArthaCoin burn mechanics
    async fn execute_transfer(
        &self,
        transaction: &Transaction,
        state: &ArthaCoinState,
        gas_used: u64,
    ) -> Result<ExecutionResult> {
        debug!(
            "Executing ArthaCoin transfer: {} -> {} amount: {}",
            transaction.sender, transaction.recipient, transaction.amount
        );

        // ArthaCoin transfer includes burn mechanics
        if let Err(e) = state
            .transfer(
                &transaction.sender,
                &transaction.recipient,
                transaction.amount,
            )
            .await
        {
            return Ok(ExecutionResult::Failure(e.to_string()));
        }

        // Update nonce
        let current_nonce = state.get_nonce(&transaction.sender)?;
        state.set_nonce(&transaction.sender, current_nonce + 1)?;

        Ok(ExecutionResult::Success)
    }

    /// Execute contract deployment
    async fn execute_deploy(
        &self,
        transaction: &Transaction,
        state: &ArthaCoinState,
        gas_used: u64,
    ) -> Result<ExecutionResult> {
        debug!("Executing contract deployment");

        // For now, just store the contract code in state storage
        let contract_address = format!("contract_{}", hex::encode(transaction.hash().as_ref()));
        state.set_storage(&contract_address, transaction.data.clone())?;

        // Update nonce
        let current_nonce = state.get_nonce(&transaction.sender)?;
        state.set_nonce(&transaction.sender, current_nonce + 1)?;

        info!("Contract deployed at address: {}", contract_address);
        Ok(ExecutionResult::Success)
    }

    /// Execute contract call
    async fn execute_contract_call(
        &self,
        transaction: &Transaction,
        state: &ArthaCoinState,
        gas_used: u64,
    ) -> Result<ExecutionResult> {
        debug!("Executing contract call to: {}", transaction.recipient);

        // For now, just verify the contract exists
        let contract_code = state.get_storage(&transaction.recipient)?;
        if contract_code.is_none() {
            return Ok(ExecutionResult::Failure("Contract not found".into()));
        }

        // Update nonce
        let current_nonce = state.get_nonce(&transaction.sender)?;
        state.set_nonce(&transaction.sender, current_nonce + 1)?;

        Ok(ExecutionResult::Success)
    }

    /// Execute validator registration
    async fn execute_validator_registration(
        &self,
        transaction: &Transaction,
        state: &ArthaCoinState,
        gas_used: u64,
    ) -> Result<ExecutionResult> {
        debug!(
            "Executing validator registration for: {}",
            transaction.sender
        );

        // Store validator info in state
        let validator_key = format!("validator_{}", transaction.sender);
        state.set_storage(&validator_key, transaction.data.clone())?;

        // Update nonce
        let current_nonce = state.get_nonce(&transaction.sender)?;
        state.set_nonce(&transaction.sender, current_nonce + 1)?;

        info!("Validator registered: {}", transaction.sender);
        Ok(ExecutionResult::Success)
    }

    /// Execute staking
    async fn execute_stake(
        &self,
        transaction: &Transaction,
        state: &ArthaCoinState,
        gas_used: u64,
    ) -> Result<ExecutionResult> {
        debug!(
            "Executing stake for: {} amount: {}",
            transaction.sender, transaction.amount
        );

        // Transfer stake amount to staking pool
        if let Err(e) = state
            .transfer(
                &transaction.sender,
                "staking_rewards_pool",
                transaction.amount,
            )
            .await
        {
            return Ok(ExecutionResult::Failure(e.to_string()));
        }

        // Store staking info
        let stake_key = format!(
            "stake_{}_{}",
            transaction.sender,
            hex::encode(transaction.hash().as_ref())
        );
        state.set_storage(&stake_key, transaction.amount.to_le_bytes().to_vec())?;

        // Update nonce
        let current_nonce = state.get_nonce(&transaction.sender)?;
        state.set_nonce(&transaction.sender, current_nonce + 1)?;

        info!(
            "Staked {} ARTHA for {}",
            transaction.amount, transaction.sender
        );
        Ok(ExecutionResult::Success)
    }

    /// Execute unstaking
    async fn execute_unstake(
        &self,
        transaction: &Transaction,
        state: &ArthaCoinState,
        gas_used: u64,
    ) -> Result<ExecutionResult> {
        debug!("Executing unstake for: {}", transaction.sender);

        // For now, just update nonce (full unstaking logic would be more complex)
        let current_nonce = state.get_nonce(&transaction.sender)?;
        state.set_nonce(&transaction.sender, current_nonce + 1)?;

        Ok(ExecutionResult::Success)
    }

    /// Execute reward claim
    async fn execute_claim_reward(
        &self,
        transaction: &Transaction,
        state: &ArthaCoinState,
        gas_used: u64,
    ) -> Result<ExecutionResult> {
        debug!("Executing reward claim for: {}", transaction.sender);

        // Distribute reward from staking pool
        let reward_amount = transaction.amount;
        if let Err(e) = state
            .distribute_staking_reward(&transaction.sender, reward_amount)
            .await
        {
            return Ok(ExecutionResult::Failure(e.to_string()));
        }

        // Update nonce
        let current_nonce = state.get_nonce(&transaction.sender)?;
        state.set_nonce(&transaction.sender, current_nonce + 1)?;

        info!(
            "Claimed {} ARTHA reward for {}",
            reward_amount, transaction.sender
        );
        Ok(ExecutionResult::Success)
    }

    /// Execute batch transaction
    async fn execute_batch(
        &self,
        transaction: &Transaction,
        state: &ArthaCoinState,
        gas_used: u64,
    ) -> Result<ExecutionResult> {
        debug!("Executing batch transaction");

        // For now, just update nonce
        let current_nonce = state.get_nonce(&transaction.sender)?;
        state.set_nonce(&transaction.sender, current_nonce + 1)?;

        Ok(ExecutionResult::Success)
    }

    /// Execute system transaction
    async fn execute_system(
        &self,
        transaction: &Transaction,
        state: &ArthaCoinState,
        gas_used: u64,
    ) -> Result<ExecutionResult> {
        debug!("Executing system transaction");

        // Check if this is a cycle emission transaction
        if transaction.sender == "system" && transaction.data == b"mint_cycle" {
            let emission_amount = state.mint_cycle_emission().await?;
            info!("System minted cycle emission: {} ARTHA", emission_amount);
        }

        Ok(ExecutionResult::Success)
    }

    /// Get transaction hash (utility)
    fn hash(&self, transaction: &Transaction) -> String {
        hex::encode(transaction.hash().as_ref())
    }
}

// Transaction hash is implemented in ledger/transaction.rs
