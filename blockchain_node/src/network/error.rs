use std::error::Error;
use std::fmt;
use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Failed to acquire lock: {0}")]
    LockError(String),

    #[error("Block not found: {0}")]
    BlockNotFound(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Protocol error: {0}")]
    ProtocolError(String),

    #[error("Peer error: {0}")]
    PeerError(String),

    #[error("Message error: {0}")]
    MessageError(String),

    #[error("Timeout error: {0}")]
    TimeoutError(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("Other error: {0}")]
    Other(Box<dyn Error + Send + Sync>),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", std::error::Error::description(self))
    }
}

impl Error for NetworkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            NetworkError::IoError(err) => Some(err),
            NetworkError::Other(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

impl From<io::Error> for NetworkError {
    fn from(err: io::Error) -> Self {
        NetworkError::ConnectionError(err.to_string())
    }
}

impl From<Box<dyn Error + Send + Sync>> for NetworkError {
    fn from(err: Box<dyn Error + Send + Sync>) -> Self {
        NetworkError::Other(err)
    }
}

// Helper function for converting errors
pub fn to_network_error<E: Error>(err: E) -> NetworkError {
    NetworkError::Other(Box::new(err))
}

impl From<tokio::time::error::Elapsed> for NetworkError {
    fn from(error: tokio::time::error::Elapsed) -> Self {
        NetworkError::TimeoutError(error.to_string())
    }
}

impl From<String> for NetworkError {
    fn from(error: String) -> Self {
        NetworkError::Unknown(error)
    }
}

impl From<&str> for NetworkError {
    fn from(error: &str) -> Self {
        NetworkError::Unknown(error.to_string())
    }
}

impl From<tokio::sync::AcquireError> for NetworkError {
    fn from(err: tokio::sync::AcquireError) -> Self {
        NetworkError::LockError(err.to_string())
    }
}

impl From<serde_json::Error> for NetworkError {
    fn from(err: serde_json::Error) -> Self {
        NetworkError::ProtocolError(err.to_string())
    }
}

impl From<anyhow::Error> for NetworkError {
    fn from(err: anyhow::Error) -> Self {
        NetworkError::Unknown(err.to_string())
    }
} 