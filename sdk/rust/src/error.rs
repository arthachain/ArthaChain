use thiserror::Error;

/// Error type for the SDK
#[derive(Debug, Error)]
pub enum Error {
    /// Request error
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),
    
    /// JSON serialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    /// RPC error
    #[error("RPC error ({code}): {message}")]
    RpcError {
        /// Error code
        code: i32,
        /// Error message
        message: String,
    },
    
    /// Signature error
    #[error("Signature error: {0}")]
    SignatureError(String),
    
    /// Invalid address
    #[error("Invalid address: {0}")]
    InvalidAddress(String),
    
    /// Parsing error
    #[error("Parse error: {0}")]
    ParseError(String),
    
    /// Function not found
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    
    /// Not a view function
    #[error("Not a view function: {0}")]
    NotViewFunction(String),
    
    /// No wallet available
    #[error("No wallet available")]
    NoWallet,
    
    /// Transaction error
    #[error("Transaction error: {0}")]
    TransactionError(String),
    
    /// Execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    /// Contract deployment error
    #[error("Contract deployment error: {0}")]
    DeploymentError(String),
    
    /// Gas limit exceeded
    #[error("Gas limit exceeded")]
    GasLimitExceeded,
    
    /// Timeout
    #[error("Operation timed out")]
    Timeout,
} 