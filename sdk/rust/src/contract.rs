use std::sync::Arc;
use reqwest::Client;
use serde::{Serialize, Deserialize};
use crate::error::Error;
use crate::rpc;
use crate::types::{ContractMetadata, FunctionMetadata, TransactionInfo, ContractReceipt, SignedTransaction};
use crate::transaction::Transaction;
use crate::wallet::Wallet;

/// WASM Contract Interface
pub struct Contract {
    /// Contract address
    address: String,
    /// RPC endpoint URL
    endpoint: String,
    /// HTTP client
    client: Arc<Client>,
}

/// Contract function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub type_name: String,
    /// Parameter value (JSON encoded)
    pub value: serde_json::Value,
}

impl Contract {
    /// Create a new contract instance
    pub fn new(endpoint: String, address: String, client: Arc<Client>) -> Self {
        Self {
            endpoint,
            address,
            client,
        }
    }
    
    /// Get contract address
    pub fn address(&self) -> &str {
        &self.address
    }
    
    /// Get contract metadata
    pub async fn metadata(&self) -> Result<ContractMetadata, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_get_contract_metadata_request(&self.address))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<ContractMetadata> = response.json().await?;
        
        rpc::handle_response(rpc_response)
    }
    
    /// Check if a function exists
    pub async fn has_function(&self, function_name: &str) -> Result<bool, Error> {
        let metadata = self.metadata().await?;
        
        Ok(metadata.functions.iter().any(|f| f.name == function_name))
    }
    
    /// Get function metadata
    pub async fn function_metadata(&self, function_name: &str) -> Result<FunctionMetadata, Error> {
        let metadata = self.metadata().await?;
        
        metadata.functions.iter()
            .find(|f| f.name == function_name)
            .cloned()
            .ok_or_else(|| Error::FunctionNotFound(function_name.to_string()))
    }
    
    /// Call a view function (read-only)
    pub async fn call_view(&self, function_name: &str, args: &[u8]) -> Result<Vec<u8>, Error> {
        // Ensure function exists and is view
        let function = self.function_metadata(function_name).await?;
        
        if !function.is_view {
            return Err(Error::NotViewFunction(function_name.to_string()));
        }
        
        // Prepare RPC call
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_call_view_function_request(
                &self.address,
                function_name,
                args,
            ))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<String> = response.json().await?;
        
        let result_hex = rpc::handle_response(rpc_response)?;
        let result = hex::decode(result_hex.trim_start_matches("0x"))
            .map_err(|e| Error::ParseError(format!("Invalid hex result: {}", e)))?;
            
        Ok(result)
    }
    
    /// Call a function that modifies state (requires a transaction)
    pub async fn call(
        &self,
        wallet: &Wallet,
        function_name: &str,
        args: &[u8],
        value: Option<u64>,
        gas_limit: Option<u64>,
    ) -> Result<ContractReceipt, Error> {
        // Create transaction for contract call
        let tx = Transaction::new_contract_call(
            &self.address,
            function_name,
            args,
            value.unwrap_or(0),
            gas_limit.unwrap_or(1_000_000),
            None, // gas_price
            None, // nonce
        );
        
        // Sign transaction
        let signed_tx = wallet.sign_transaction(&tx)?;
        
        // Send transaction
        self.send_transaction(&signed_tx).await
    }
    
    /// Send a signed transaction
    async fn send_transaction(&self, transaction: &SignedTransaction) -> Result<ContractReceipt, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_send_transaction_request(transaction))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<ContractReceipt> = response.json().await?;
        
        rpc::handle_response(rpc_response)
    }
    
    /// Estimate gas for a contract call
    pub async fn estimate_gas(
        &self,
        function_name: &str,
        args: &[u8],
        value: Option<u64>,
    ) -> Result<u64, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_estimate_gas_request(
                &self.address,
                function_name,
                args,
                value.unwrap_or(0),
            ))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<String> = response.json().await?;
        
        let gas_hex = rpc::handle_response(rpc_response)?;
        let gas = u64::from_str_radix(gas_hex.trim_start_matches("0x"), 16)
            .map_err(|_| Error::ParseError("Invalid gas estimate format".to_string()))?;
            
        Ok(gas)
    }
    
    /// Get past events for this contract
    pub async fn get_events(
        &self,
        event_name: Option<&str>,
        from_block: Option<u64>,
        to_block: Option<u64>,
        limit: Option<u64>,
    ) -> Result<Vec<ContractEvent>, Error> {
        let response = self.client
            .post(&self.endpoint)
            .json(&rpc::build_get_events_request(
                &self.address,
                event_name,
                from_block,
                to_block,
                limit,
            ))
            .send()
            .await?;
            
        let rpc_response: rpc::RpcResponse<Vec<ContractEvent>> = response.json().await?;
        
        rpc::handle_response(rpc_response)
    }
}

/// Contract event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractEvent {
    /// Event name
    pub name: String,
    /// Contract address
    pub address: String,
    /// Block number
    pub block_number: u64,
    /// Transaction hash
    pub transaction_hash: String,
    /// Event data (encoded)
    pub data: String,
    /// Event topics
    pub topics: Vec<String>,
    /// Timestamp
    pub timestamp: u64,
} 