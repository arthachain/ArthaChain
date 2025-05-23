// Re-exports for API models
pub use crate::api::handlers::accounts::{
    AccountResponse, TransactionListParams, TransactionListResponse,
};
pub use crate::api::handlers::blocks::BlockResponse;
pub use crate::api::handlers::consensus::ConsensusStatusResponse;
pub use crate::api::handlers::status::{PeerInfo, PeerListResponse, StatusResponse};
pub use crate::api::handlers::transactions::{
    SubmitTransactionRequest, SubmitTransactionResponse, TransactionResponse,
};

/// Response type for a successful API operation with no data
#[derive(serde::Serialize)]
pub struct SuccessResponse {
    /// Success status
    pub success: bool,
    /// Optional message
    pub message: Option<String>,
}

impl Default for SuccessResponse {
    fn default() -> Self {
        Self {
            success: true,
            message: None,
        }
    }
}

/// Response for pagination
#[derive(serde::Serialize)]
pub struct PaginatedResponse<T> {
    /// Items
    pub items: Vec<T>,
    /// Total count
    pub total: usize,
    /// Page number
    pub page: usize,
    /// Page size
    pub page_size: usize,
}

/// Request for pagination
#[derive(serde::Deserialize)]
pub struct PaginationParams {
    /// Page number (0-based)
    #[serde(default)]
    pub page: usize,
    /// Page size
    #[serde(default = "default_page_size")]
    pub page_size: usize,
}

fn default_page_size() -> usize {
    20
}
