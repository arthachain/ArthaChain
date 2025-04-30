mod error;
mod rpc;
mod types;
mod transaction;
mod contract;
mod wallet;

pub use error::Error;
pub use types::*;
pub use transaction::Transaction;
pub use contract::Contract;
pub use wallet::Wallet;

use std::sync::Arc;
use reqwest::Client;

/// The main SDK client
pub struct BlockchainClient {
    /// RPC endpoint URL
    endpoint: String,
    /// HTTP client
    client: Arc<Client>,
    /// Optional wallet for signing transactions
    wallet: Option<Wallet>,
}

impl BlockchainClient {
    /// Create a new client
    pub fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            client: Arc::new(Client::new()),
            wallet: None,
        }
    }
    
    /// Set a wallet for signing transactions
    pub fn with_wallet(mut self, wallet: Wallet) -> Self {
        self.wallet = Some(wallet);
        self
    }
    
    /// Get the wallet
    pub fn wallet(&self) -> Option<&Wallet> {
        self.wallet.as_ref()
    }
    
    /// Get a contract instance
    pub fn contract(&self, address: &str) -> Contract {
        Contract::new(self.endpoint.clone(), address.to_string(), self.client.clone())
    }
    
    /// Deploy a new contract
    pub async fn deploy_contract(
        &self,
        bytecode: &[u8],
        constructor_args: Option<&[u8]>,
        gas_limit: Option<u64>,
    ) -> Result<ContractReceipt, Error> {
        let wallet = self.wallet.as_ref().ok_or(Error::NoWallet)?;
        
        // Create transaction for contract deployment
        let tx = Transaction::new_contract_deployment(
            bytecode,
            constructor_args,
            gas_limit.unwrap_or(10_000_000),
            None, // gas price
            None, // nonce (will be fetched automatically)
        );
        
        // Sign and send transaction
        let signed_tx = wallet.sign_transaction(&tx)?;
        
        // Send the transaction
        let receipt = self
            .send_transaction(&signed_tx)
            .await?;
            
        Ok(receipt)
    }
    
    /// Send a transaction
    pub async fn send_transaction(
        &self,
        transaction: &SignedTransaction,
    ) -> Result<ContractReceipt, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_send_transaction_request(transaction))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<ContractReceipt> = response.json().await?;
        
        rpc::handle_response(rpc_response)
    }
    
    /// Get transaction by hash
    pub async fn get_transaction(&self, tx_hash: &str) -> Result<TransactionInfo, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_get_transaction_request(tx_hash))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<TransactionInfo> = response.json().await?;
        
        rpc::handle_response(rpc_response)
    }
    
    /// Get latest block
    pub async fn get_latest_block(&self) -> Result<BlockInfo, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_get_latest_block_request())
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<BlockInfo> = response.json().await?;
        
        rpc::handle_response(rpc_response)
    }
    
    /// Get block by hash
    pub async fn get_block_by_hash(&self, block_hash: &str) -> Result<BlockInfo, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_get_block_by_hash_request(block_hash))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<BlockInfo> = response.json().await?;
        
        rpc::handle_response(rpc_response)
    }
    
    /// Get block by number
    pub async fn get_block_by_number(&self, block_number: u64) -> Result<BlockInfo, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_get_block_by_number_request(block_number))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<BlockInfo> = response.json().await?;
        
        rpc::handle_response(rpc_response)
    }
    
    /// Get account balance
    pub async fn get_balance(&self, address: &str) -> Result<u64, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_get_balance_request(address))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<String> = response.json().await?;
        
        let balance_hex = rpc::handle_response(rpc_response)?;
        let balance = u64::from_str_radix(balance_hex.trim_start_matches("0x"), 16)
            .map_err(|_| Error::ParseError("Invalid balance format".to_string()))?;
            
        Ok(balance)
    }
    
    /// Get account nonce
    pub async fn get_nonce(&self, address: &str) -> Result<u64, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_get_nonce_request(address))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<String> = response.json().await?;
        
        let nonce_hex = rpc::handle_response(rpc_response)?;
        let nonce = u64::from_str_radix(nonce_hex.trim_start_matches("0x"), 16)
            .map_err(|_| Error::ParseError("Invalid nonce format".to_string()))?;
            
        Ok(nonce)
    }
} 