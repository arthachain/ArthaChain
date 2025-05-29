use crate::api::fraud_monitoring::FraudMonitoringService;
use crate::ledger::transaction::Transaction;
use axum::{
    extract::{Extension, Path},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;

/// Transaction routes
pub struct TransactionRoutes;

impl TransactionRoutes {
    /// Create transaction router
    pub fn create_router(fraud_service: Arc<FraudMonitoringService>) -> Router {
        Router::new()
            .route("/api/transaction", post(Self::submit_transaction))
            .route("/api/transaction/:hash", get(Self::get_transaction))
            .layer(Extension(fraud_service))
    }

    /// Submit a new transaction
    async fn submit_transaction(
        Extension(fraud_service): Extension<Arc<FraudMonitoringService>>,
        Json(transaction): Json<Transaction>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        // Process transaction with fraud detection
        match fraud_service.process_transaction(&transaction, None).await {
            Ok(detection_result) => {
                // Check if transaction should be blocked
                if matches!(
                    detection_result.recommended_action,
                    crate::ai_engine::models::advanced_fraud_detection::RecommendedAction::Block
                    | crate::ai_engine::models::advanced_fraud_detection::RecommendedAction::TemporaryBlock
                ) {
                    // Transaction is high risk, reject it
                    return Err(StatusCode::FORBIDDEN);
                }

                // Transaction is allowed, continue processing
                // In a real implementation, you would now add to mempool, propagate, etc.

                // Return detection result with the transaction
                Ok(Json(serde_json::json!({
                    "transaction": transaction,
                    "fraud_detection": detection_result,
                    "status": "accepted"
                })))
            }
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    /// Get transaction by hash
    async fn get_transaction(
        Extension(fraud_service): Extension<Arc<FraudMonitoringService>>,
        Path(hash): Path<String>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        // In a real implementation, you would fetch the transaction from storage
        // For now, we'll just check if we have any fraud detection results for it

        match fraud_service.get_transaction_details(&hash).await {
            Some(detection_result) => Ok(Json(serde_json::json!({
                "transaction_hash": hash,
                "fraud_detection": detection_result,
            }))),
            None => Err(StatusCode::NOT_FOUND),
        }
    }
}
