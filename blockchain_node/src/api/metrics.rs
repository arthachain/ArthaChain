use crate::node::Node;
use axum::{
    response::Json,
    http::StatusCode,
    response::IntoResponse,
    extract::Extension,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Serialize, Deserialize};
use tokio::sync::Mutex;

/// Performance metrics for the blockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainMetrics {
    /// Current transactions per second (estimated)
    pub current_tps: f32,
    /// Maximum observed TPS
    pub max_tps: f32,
    /// Number of active validators
    pub validator_count: usize,
    /// TPS per validator ratio
    pub tps_per_validator: f32,
    /// Timestamp of the measurement
    pub timestamp: u64,
}

impl Default for ChainMetrics {
    fn default() -> Self {
        Self {
            current_tps: 0.0,
            max_tps: 0.0,
            validator_count: 0,
            tps_per_validator: 0.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_secs(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsData {
    pub total_transactions: u64,
    pub total_blocks: u64,
    pub current_tps: f64,
    pub average_block_time: f64,
    pub active_peers: u32,
    pub memory_usage: u64,
    pub cpu_usage: f64,
}

/// Metrics service for tracking blockchain performance
pub struct MetricsService {
    total_transactions: AtomicU64,
    total_blocks: AtomicU64,
    tps: AtomicU64,
    active_peers: AtomicU64,
    memory_usage: AtomicU64,
    cpu_usage: AtomicU64,
}

impl MetricsService {
    /// Create a new metrics service
    pub fn new() -> Self {
        Self {
            total_transactions: AtomicU64::new(0),
            total_blocks: AtomicU64::new(0),
            tps: AtomicU64::new(0),
            active_peers: AtomicU64::new(0),
            memory_usage: AtomicU64::new(0),
            cpu_usage: AtomicU64::new(0),
        }
    }
    
    pub fn record_transaction(&self) {
        self.total_transactions.fetch_add(1, Ordering::SeqCst);
    }
    
    pub fn record_block(&self, tps: f64) {
        self.total_blocks.fetch_add(1, Ordering::SeqCst);
        self.tps.store(tps as u64, Ordering::SeqCst);
    }
    
    pub fn update_peer_count(&self, count: u64) {
        self.active_peers.store(count, Ordering::SeqCst);
    }
    
    pub fn update_resource_usage(&self, memory: u64, cpu: u64) {
        self.memory_usage.store(memory, Ordering::SeqCst);
        self.cpu_usage.store(cpu, Ordering::SeqCst);
    }
    
    pub fn get_metrics(&self) -> Result<String, anyhow::Error> {
        Ok(format!(
            "total_transactions: {}\ntotal_blocks: {}\ntps: {}\nactive_peers: {}\nmemory_usage: {}\ncpu_usage: {}",
            self.total_transactions.load(Ordering::SeqCst),
            self.total_blocks.load(Ordering::SeqCst),
            self.tps.load(Ordering::SeqCst),
            self.active_peers.load(Ordering::SeqCst),
            self.memory_usage.load(Ordering::SeqCst),
            self.cpu_usage.load(Ordering::SeqCst)
        ))
    }

    pub fn get_current_tps(&self) -> f64 {
        self.tps.load(Ordering::SeqCst) as f64
    }
}

pub struct MetricsApi {
    node: Arc<Mutex<Node>>,
}

impl MetricsApi {
    pub fn new(node: Arc<Mutex<Node>>) -> Self {
        Self { node }
    }

    pub async fn get_metrics(&self) -> impl IntoResponse {
        let _node = self.node.lock().await;
        let metrics = _node.get_metrics().await;
        
        match metrics {
            Ok(metrics) => (StatusCode::OK, Json(metrics)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": format!("Failed to get metrics: {}", e)
                }))
            ).into_response()
        }
    }

    pub async fn get_node_info(&self) -> impl IntoResponse {
        let _node = self.node.lock().await;
        let info = _node.get_info().await;
        
        match info {
            Ok(info) => (StatusCode::OK, Json(info)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": format!("Failed to get node info: {}", e)
                }))
            ).into_response()
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    
    #[tokio::test]
    async fn test_metrics_service() {
        let config = Config::default();
        let _node = Arc::new(Node::new(config).await.unwrap());
        let service = MetricsService::new();
        
        // Test metrics recording
        service.record_transaction();
        service.record_block(0.5);
        
        let metrics = service.get_metrics().unwrap();
        assert!(metrics.contains("total_transactions: 1"));
        assert!(metrics.contains("total_blocks: 1"));
    }
} 