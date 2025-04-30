use crate::api::metrics::MetricsService;
use axum::{
    extract::Extension,
    response::Json,
};
use serde::Serialize;
use serde_json::json;
use std::sync::Arc;

/// Response structure for TPS metrics
#[derive(Serialize)]
pub struct TPSResponse {
    /// Current transactions per second
    pub current_tps: f32,
    /// Maximum recorded TPS
    pub max_tps: f32,
    /// TPS metrics per validator node
    pub tps_per_validator: f32,
    /// Number of active validators
    pub validator_count: usize,
}

/// Retrieve all chain metrics
pub async fn get_metrics(
    Extension(metrics): Extension<Arc<MetricsService>>,
) -> Json<serde_json::Value> {
    match metrics.get_metrics() {
        Ok(metrics_str) => Json(json!({
            "status": "success",
            "data": metrics_str
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": e.to_string()
        }))
    }
}

/// Retrieve TPS metrics
pub async fn get_tps(
    Extension(metrics): Extension<Arc<MetricsService>>,
) -> Json<serde_json::Value> {
    Json(json!({
        "status": "success",
        "data": {
            "tps": metrics.get_current_tps()
        }
    }))
} 