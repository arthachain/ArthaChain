use crate::ai_engine::models::advanced_fraud_detection::{
    AdvancedFraudDetection, FraudDetectionResult, RiskLevel,
};
use crate::ledger::block::Block;
use crate::ledger::transaction::Transaction;
use anyhow::Result;
use axum::{
    extract::{Extension, Path, Query},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use chrono::{DateTime, Duration, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Dashboard statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStats {
    /// Total transactions analyzed
    pub total_transactions: u64,
    /// Total suspicious transactions
    pub suspicious_transactions: u64,
    /// Suspicious transaction rate
    pub suspicious_rate: f32,
    /// Last update time
    pub last_update: DateTime<Utc>,
    /// Count by risk level
    pub risk_level_counts: HashMap<String, u64>,
    /// Recent detections
    pub recent_detections: Vec<FraudDetectionResult>,
    /// Top risky addresses
    pub top_risky_addresses: Vec<AddressRiskScore>,
    /// Detection rate over time (hourly)
    pub detection_rate_history: Vec<TimeSeriesPoint>,
}

/// Address risk score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddressRiskScore {
    /// Address
    pub address: String,
    /// Risk score
    pub risk_score: f32,
    /// Transaction count
    pub transaction_count: u32,
    /// Last activity time
    pub last_activity: DateTime<Utc>,
}

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value
    pub value: f32,
}

/// Fraud monitoring service
pub struct FraudMonitoringService {
    /// Detection model
    detection_model: Arc<AdvancedFraudDetection>,
    /// Dashboard statistics
    dashboard_stats: Arc<RwLock<DashboardStats>>,
    /// Detection history
    detection_history: Arc<RwLock<Vec<FraudDetectionResult>>>,
    /// Hourly stats
    hourly_stats: Arc<RwLock<HashMap<DateTime<Utc>, HourlyStats>>>,
}

/// Hourly statistics
#[derive(Debug, Clone)]
struct HourlyStats {
    /// Total transactions
    total_transactions: u64,
    /// Suspicious transactions
    suspicious_transactions: u64,
    /// High risk transactions
    high_risk_transactions: u64,
}

impl FraudMonitoringService {
    /// Create a new fraud monitoring service
    pub async fn new(model: Arc<AdvancedFraudDetection>) -> Self {
        // Initialize dashboard stats
        let dashboard_stats = DashboardStats {
            total_transactions: 0,
            suspicious_transactions: 0,
            suspicious_rate: 0.0,
            last_update: Utc::now(),
            risk_level_counts: HashMap::new(),
            recent_detections: Vec::new(),
            top_risky_addresses: Vec::new(),
            detection_rate_history: Vec::new(),
        };

        Self {
            detection_model: model,
            dashboard_stats: Arc::new(RwLock::new(dashboard_stats)),
            detection_history: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            hourly_stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Process a new transaction
    pub async fn process_transaction(
        &self,
        tx: &Transaction,
        block: Option<&Block>,
    ) -> Result<FraudDetectionResult> {
        // Run fraud detection
        let result = self.detection_model.detect_fraud(tx, block).await?;

        // Update statistics
        self.update_statistics(&result).await?;

        // Add to detection history
        let mut history = self.detection_history.write().await;
        if history.len() >= 10000 {
            history.remove(0);
        }
        history.push(result.clone());

        Ok(result)
    }

    /// Update statistics based on detection result
    async fn update_statistics(&self, result: &FraudDetectionResult) -> Result<()> {
        let mut stats = self.dashboard_stats.write().await;

        // Update transaction counts
        stats.total_transactions += 1;
        if result.is_suspicious {
            stats.suspicious_transactions += 1;
        }

        // Update suspicious rate
        stats.suspicious_rate =
            stats.suspicious_transactions as f32 / stats.total_transactions as f32;

        // Update last update time
        stats.last_update = Utc::now();

        // Update risk level counts
        let risk_level = format!("{:?}", result.risk_level);
        *stats.risk_level_counts.entry(risk_level).or_insert(0) += 1;

        // Update recent detections
        if stats.recent_detections.len() >= 10 {
            stats.recent_detections.remove(0);
        }
        stats.recent_detections.push(result.clone());

        // Update hourly stats
        self.update_hourly_stats(result).await?;

        // Update detection rate history from hourly stats
        stats.detection_rate_history = self.get_detection_rate_history(24).await?;

        // Update top risky addresses
        stats.top_risky_addresses = self.get_top_risky_addresses(10).await?;

        Ok(())
    }

    /// Update hourly statistics
    async fn update_hourly_stats(&self, result: &FraudDetectionResult) -> Result<()> {
        let mut hourly_stats = self.hourly_stats.write().await;

        // Get current hour (truncated to hour)
        let now = result.timestamp;
        let hour = now
            .naive_utc()
            .date()
            .and_hms_opt(now.time().hour(), 0, 0)
            .unwrap_or_default()
            .and_utc();

        // Get or create hourly stat
        let stats = hourly_stats.entry(hour).or_insert(HourlyStats {
            total_transactions: 0,
            suspicious_transactions: 0,
            high_risk_transactions: 0,
        });

        // Update counts
        stats.total_transactions += 1;

        if result.is_suspicious {
            stats.suspicious_transactions += 1;
        }

        if result.risk_level == RiskLevel::High || result.risk_level == RiskLevel::Critical {
            stats.high_risk_transactions += 1;
        }

        Ok(())
    }

    /// Get detection rate history for past N hours
    async fn get_detection_rate_history(&self, hours: u32) -> Result<Vec<TimeSeriesPoint>> {
        let hourly_stats = self.hourly_stats.read().await;
        let now = Utc::now();

        let mut result = Vec::with_capacity(hours as usize);

        for i in 0..hours {
            let hour_time = now - Duration::hours(i as i64);
            let hour = hour_time
                .naive_utc()
                .date()
                .and_hms_opt(hour_time.time().hour(), 0, 0)
                .unwrap_or_default()
                .and_utc();

            if let Some(stats) = hourly_stats.get(&hour) {
                let rate = if stats.total_transactions > 0 {
                    stats.suspicious_transactions as f32 / stats.total_transactions as f32
                } else {
                    0.0
                };

                result.push(TimeSeriesPoint {
                    timestamp: hour,
                    value: rate,
                });
            } else {
                result.push(TimeSeriesPoint {
                    timestamp: hour,
                    value: 0.0,
                });
            }
        }

        // Sort by timestamp
        result.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        Ok(result)
    }

    /// Get top N risky addresses
    async fn get_top_risky_addresses(&self, limit: usize) -> Result<Vec<AddressRiskScore>> {
        let mut result = Vec::new();

        // In a real implementation, this would query the address profiles
        // Since we don't have direct access to that here, we'll return placeholder data
        // In a full implementation, you'd use self.detection_model.get_address_profiles()

        // Placeholder data
        for i in 0..limit {
            result.push(AddressRiskScore {
                address: format!("0x{i:040x}"),
                risk_score: 0.5 + (i as f32 * 0.05),
                transaction_count: 10 + i as u32,
                last_activity: Utc::now() - Duration::hours(i as i64),
            });
        }

        // Sort by risk score (descending)
        result.sort_by(|a, b| b.risk_score.partial_cmp(&a.risk_score).unwrap());

        Ok(result)
    }

    /// Get dashboard statistics
    pub async fn get_dashboard_stats(&self) -> DashboardStats {
        self.dashboard_stats.read().await.clone()
    }

    /// Get detection history
    pub async fn get_detection_history(&self, limit: usize) -> Vec<FraudDetectionResult> {
        let history = self.detection_history.read().await;
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get transaction details
    pub async fn get_transaction_details(&self, tx_hash: &str) -> Option<FraudDetectionResult> {
        let history = self.detection_history.read().await;
        history
            .iter()
            .find(|result| result.tx_hash == tx_hash)
            .cloned()
    }
}

// API handlers for fraud monitoring dashboard

/// Query parameters for detection history
#[derive(Debug, Deserialize)]
pub struct HistoryParams {
    /// Number of items to return
    #[serde(default = "default_limit")]
    limit: usize,
    /// Risk level filter
    risk_level: Option<String>,
}

fn default_limit() -> usize {
    100
}

/// Get dashboard statistics
pub async fn get_dashboard_stats(
    Extension(service): Extension<Arc<FraudMonitoringService>>,
) -> Json<DashboardStats> {
    let stats = service.get_dashboard_stats().await;
    Json(stats)
}

/// Get detection history
pub async fn get_detection_history(
    Extension(service): Extension<Arc<FraudMonitoringService>>,
    Query(params): Query<HistoryParams>,
) -> Json<Vec<FraudDetectionResult>> {
    let history = service.get_detection_history(params.limit).await;

    // Apply risk level filter if provided
    let history = if let Some(risk_level) = params.risk_level {
        history
            .into_iter()
            .filter(|result| {
                format!("{:?}", result.risk_level).to_lowercase() == risk_level.to_lowercase()
            })
            .collect()
    } else {
        history
    };

    Json(history)
}

/// Get transaction details
pub async fn get_transaction_details(
    Extension(service): Extension<Arc<FraudMonitoringService>>,
    Path(tx_hash): Path<String>,
) -> Result<Json<FraudDetectionResult>, StatusCode> {
    match service.get_transaction_details(&tx_hash).await {
        Some(details) => Ok(Json(details)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// Create fraud monitoring API router
pub fn create_fraud_monitoring_router(service: Arc<FraudMonitoringService>) -> Router {
    Router::new()
        .route("/api/fraud/dashboard", get(get_dashboard_stats))
        .route("/api/fraud/history", get(get_detection_history))
        .route(
            "/api/fraud/transaction/:tx_hash",
            get(get_transaction_details),
        )
        .layer(Extension(service))
}
