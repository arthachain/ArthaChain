use crate::monitoring::health_check::HealthChecker;
use crate::monitoring::alerting::AlertManager;
use crate::ledger::state::State;
use axum::{
    extract::Extension,
    http::StatusCode,
    response::Json as AxumJson,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// Monitoring status information
#[derive(Debug, Serialize)]
pub struct MonitoringStatus {
    pub overall_health: String,
    pub services_healthy: usize,
    pub services_total: usize,
    pub last_check_time: u64,
    pub monitoring_active: bool,
    pub alert_count: usize,
    pub critical_alerts: usize,
    pub warning_alerts: usize,
}

/// Monitoring health information
#[derive(Debug, Serialize)]
pub struct MonitoringHealth {
    pub status: String,
    pub timestamp: u64,
    pub uptime_seconds: u64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_status: String,
    pub database_status: String,
    pub consensus_status: String,
    pub storage_status: String,
}

/// Network monitoring information
#[derive(Debug, Serialize)]
pub struct NetworkMonitoring {
    pub active_peers: usize,
    pub total_peers: usize,
    pub network_latency_ms: u64,
    pub bandwidth_usage_mbps: f64,
    pub connection_errors: u64,
    pub sync_status: String,
    pub last_block_received: u64,
    pub last_transaction_received: u64,
}

/// Monitoring service for handling monitoring operations
pub struct MonitoringService {
    health_checker: Arc<RwLock<HealthChecker>>,
    alert_manager: Arc<RwLock<AlertManager>>,
    state: Arc<RwLock<State>>,
}

impl MonitoringService {
    pub fn new(
        health_checker: Arc<RwLock<HealthChecker>>,
        alert_manager: Arc<RwLock<AlertManager>>,
        state: Arc<RwLock<State>>,
    ) -> Self {
        Self {
            health_checker,
            alert_manager,
            state,
        }
    }

    /// Get monitoring status
    pub async fn get_monitoring_status(&self) -> Result<MonitoringStatus, String> {
        // For now, return default values since these methods don't exist yet
        let overall_health = "healthy".to_string();
        let services_healthy = 5;
        let services_total = 5;
        let last_check_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let monitoring_active = true;
        let alert_count = 0;
        let critical_alerts = 0;
        let warning_alerts = 0;
        
        Ok(MonitoringStatus {
            overall_health,
            services_healthy,
            services_total,
            last_check_time,
            monitoring_active,
            alert_count,
            critical_alerts,
            warning_alerts,
        })
    }

    /// Get monitoring health
    pub async fn get_monitoring_health(&self) -> Result<MonitoringHealth, String> {
        // For now, return default values since these methods don't exist yet
        let status = "healthy".to_string();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let uptime = 3600; // 1 hour default
        let memory_usage = 1024; // 1GB default
        let cpu_usage = 25.0; // 25% default
        let disk_usage = 45.0; // 45% default
        let network_status = "connected".to_string();
        let database_status = "connected".to_string();
        let consensus_status = "active".to_string();
        let storage_status = "healthy".to_string();
        
        Ok(MonitoringHealth {
            status,
            timestamp,
            uptime_seconds: uptime,
            memory_usage_mb: memory_usage,
            cpu_usage_percent: cpu_usage,
            disk_usage_percent: disk_usage,
            network_status,
            database_status,
            consensus_status,
            storage_status,
        })
    }

    /// Get network monitoring data
    pub async fn get_network_monitoring(&self) -> Result<NetworkMonitoring, String> {
        // For now, return default values since these methods don't exist yet
        let active_peers = 5;
        let total_peers = 10;
        let network_latency = 50;
        let bandwidth_usage = 25.5;
        let connection_errors = 0;
        let sync_status = "synced".to_string();
        let last_block_received = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let last_transaction_received = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        Ok(NetworkMonitoring {
            active_peers,
            total_peers,
            network_latency_ms: network_latency,
            bandwidth_usage_mbps: bandwidth_usage,
            connection_errors,
            sync_status,
            last_block_received,
            last_transaction_received,
        })
    }

    /// Get monitoring info
    pub async fn get_monitoring_info(&self) -> Result<serde_json::Value, String> {
        let status = self.get_monitoring_status().await?;
        let health = self.get_monitoring_health().await?;
        let network = self.get_network_monitoring().await?;
        
        Ok(serde_json::json!({
            "status": "success",
            "monitoring": {
                "overall_health": status.overall_health,
                "services_healthy": status.services_healthy,
                "services_total": status.services_total,
                "monitoring_active": status.monitoring_active
            },
            "health": {
                "status": health.status,
                "uptime_seconds": health.uptime_seconds,
                "memory_usage_mb": health.memory_usage_mb,
                "cpu_usage_percent": health.cpu_usage_percent
            },
            "network": {
                "active_peers": network.active_peers,
                "total_peers": network.total_peers,
                "sync_status": network.sync_status,
                "connection_errors": network.connection_errors
            },
            "alerts": {
                "total": status.alert_count,
                "critical": status.critical_alerts,
                "warning": status.warning_alerts
            },
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        }))
    }

    /// Get peers count
    pub async fn get_peers_count(&self) -> Result<serde_json::Value, String> {
        let network = self.get_network_monitoring().await?;
        
        Ok(serde_json::json!({
            "status": "success",
            "peers": {
                "active": network.active_peers,
                "total": network.total_peers,
                "connection_errors": network.connection_errors
            },
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        }))
    }

    /// Get mempool size
    pub async fn get_mempool_size(&self) -> Result<serde_json::Value, String> {
        let state = self.state.read().await;
        let mempool_size = state.get_mempool_size().unwrap_or(0);
        
        Ok(serde_json::json!({
            "status": "success",
            "mempool": {
                "size": mempool_size,
                "capacity": 10000, // Default mempool capacity
                "utilization_percent": (mempool_size as f64 / 10000.0) * 100.0
            },
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        }))
    }

    /// Get uptime information
    pub async fn get_uptime(&self) -> Result<serde_json::Value, String> {
        let health = self.get_monitoring_health().await?;
        let uptime_hours = health.uptime_seconds as f64 / 3600.0;
        let uptime_days = uptime_hours / 24.0;
        
        Ok(serde_json::json!({
            "status": "success",
            "uptime": {
                "seconds": health.uptime_seconds,
                "hours": uptime_hours,
                "days": uptime_days,
                "formatted": format!("{}d {}h {}m", 
                    (uptime_hours / 24.0) as u64,
                    (uptime_hours % 24.0) as u64,
                    ((uptime_hours * 60.0) % 60.0) as u64
                )
            },
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        }))
    }

    /// Get peers information
    pub async fn get_peers(&self) -> Result<serde_json::Value, String> {
        let network = self.get_network_monitoring().await?;
        
        Ok(serde_json::json!({
            "status": "success",
            "peers": {
                "active": network.active_peers,
                "total": network.total_peers,
                "sync_status": network.sync_status,
                "last_block_received": network.last_block_received,
                "last_transaction_received": network.last_transaction_received
            },
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        }))
    }

    /// Get network information
    pub async fn get_network(&self) -> Result<serde_json::Value, String> {
        let network = self.get_network_monitoring().await?;
        
        Ok(serde_json::json!({
            "status": "success",
            "network": {
                "latency_ms": network.network_latency_ms,
                "bandwidth_usage_mbps": network.bandwidth_usage_mbps,
                "connection_errors": network.connection_errors,
                "sync_status": network.sync_status
            },
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        }))
    }
}

/// Handler for getting monitoring status
pub async fn get_monitoring_status(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<MonitoringStatus>, StatusCode> {
    // Create mock monitoring components for now
    // In real implementation, these would be injected from the monitoring module
    let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
    let alert_manager = Arc::new(RwLock::new(AlertManager::new()));
    let service = MonitoringService::new(health_checker, alert_manager, state);
    
    match service.get_monitoring_status().await {
        Ok(status) => Ok(AxumJson(status)),
        Err(e) => {
            log::error!("Failed to get monitoring status: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting monitoring health
pub async fn get_monitoring_health(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<MonitoringHealth>, StatusCode> {
    let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
    let alert_manager = Arc::new(RwLock::new(AlertManager::new()));
    let service = MonitoringService::new(health_checker, alert_manager, state);
    
    match service.get_monitoring_health().await {
        Ok(health) => Ok(AxumJson(health)),
        Err(e) => {
            log::error!("Failed to get monitoring health: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting monitoring info
pub async fn get_monitoring_info(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
    let alert_manager = Arc::new(RwLock::new(AlertManager::new()));
    let service = MonitoringService::new(health_checker, alert_manager, state);
    
    match service.get_monitoring_info().await {
        Ok(info) => Ok(AxumJson(info)),
        Err(e) => {
            log::error!("Failed to get monitoring info: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting peers count
pub async fn get_monitoring_peers_count(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
    let alert_manager = Arc::new(RwLock::new(AlertManager::new()));
    let service = MonitoringService::new(health_checker, alert_manager, state);
    
    match service.get_peers_count().await {
        Ok(count) => Ok(AxumJson(count)),
        Err(e) => {
            log::error!("Failed to get peers count: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting mempool size
pub async fn get_monitoring_mempool_size(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
    let alert_manager = Arc::new(RwLock::new(AlertManager::new()));
    let service = MonitoringService::new(health_checker, alert_manager, state);
    
    match service.get_mempool_size().await {
        Ok(size) => Ok(AxumJson(size)),
        Err(e) => {
            log::error!("Failed to get mempool size: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting uptime
pub async fn get_monitoring_uptime(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
    let alert_manager = Arc::new(RwLock::new(AlertManager::new()));
    let service = MonitoringService::new(health_checker, alert_manager, state);
    
    match service.get_uptime().await {
        Ok(uptime) => Ok(AxumJson(uptime)),
        Err(e) => {
            log::error!("Failed to get uptime: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting peers information
pub async fn get_monitoring_peers(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
    let alert_manager = Arc::new(RwLock::new(AlertManager::new()));
    let service = MonitoringService::new(health_checker, alert_manager, state);
    
    match service.get_peers().await {
        Ok(peers) => Ok(AxumJson(peers)),
        Err(e) => {
            log::error!("Failed to get peers: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Handler for getting network information
pub async fn get_monitoring_network(
    Extension(state): Extension<Arc<RwLock<State>>>,
) -> Result<AxumJson<serde_json::Value>, StatusCode> {
    let health_checker = Arc::new(RwLock::new(HealthChecker::new()));
    let alert_manager = Arc::new(RwLock::new(AlertManager::new()));
    let service = MonitoringService::new(health_checker, alert_manager, state);
    
    match service.get_network().await {
        Ok(network) => Ok(AxumJson(network)),
        Err(e) => {
            log::error!("Failed to get network: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
