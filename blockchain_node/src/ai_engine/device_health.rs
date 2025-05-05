use std::sync::{Arc, Mutex};
use anyhow::Result;
use log::info;
use std::time::{Duration, Instant};
use crate::config::Config;
use sysinfo::System;

/// Device health metrics monitored by the AI
#[derive(Debug, Clone)]
pub struct DeviceHealthMetrics {
    /// Battery level as a percentage (0-100)
    pub battery_level: f32,
    /// Battery temperature in Celsius
    pub battery_temperature: f32,
    /// CPU usage as a percentage (0-100)
    pub cpu_usage: f32,
    /// RAM usage as a percentage (0-100)
    pub ram_usage: f32,
    /// Available storage in bytes
    pub available_storage: u64,
    /// Total storage in bytes
    pub total_storage: u64,
    /// Network latency in milliseconds
    pub network_latency: u32,
    /// Network jitter in milliseconds
    pub network_jitter: u32,
    /// Device uptime in seconds
    pub uptime: u64,
    /// Root/jailbreak detection flag
    pub is_rooted: bool,
    /// Last updated timestamp
    pub last_updated: std::time::SystemTime,
}

impl Default for DeviceHealthMetrics {
    fn default() -> Self {
        Self {
            battery_level: 100.0,
            battery_temperature: 25.0,
            cpu_usage: 0.0,
            ram_usage: 0.0,
            available_storage: 1024 * 1024 * 1024 * 100, // 100 GB
            total_storage: 1024 * 1024 * 1024 * 500,    // 500 GB
            network_latency: 50,
            network_jitter: 5,
            uptime: 0,
            is_rooted: false,
            last_updated: std::time::SystemTime::now(),
        }
    }
}

/// Health status of a device
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceHealthStatus {
    /// Device is healthy and can participate fully
    Healthy,
    /// Device has minor issues but can still participate
    Warning,
    /// Device has significant issues and should be downgraded
    Degraded,
    /// Device is critically unhealthy and should not participate
    Critical,
}

/// Device health score components
#[derive(Debug, Clone)]
pub struct DeviceHealthScore {
    /// Overall health score (0.0-1.0)
    pub overall_score: f32,
    /// Battery subscore (0.0-1.0)
    pub battery_score: f32,
    /// Performance subscore (0.0-1.0)
    pub performance_score: f32,
    /// Storage subscore (0.0-1.0)
    pub storage_score: f32,
    /// Network subscore (0.0-1.0)
    pub network_score: f32,
    /// Security subscore (0.0-1.0)
    pub security_score: f32,
    /// Health status category
    pub status: DeviceHealthStatus,
}

impl Default for DeviceHealthScore {
    fn default() -> Self {
        Self {
            overall_score: 1.0,
            battery_score: 1.0,
            performance_score: 1.0,
            storage_score: 1.0,
            network_score: 1.0,
            security_score: 1.0,
            status: DeviceHealthStatus::Healthy,
        }
    }
}

/// Device Health AI that monitors and evaluates device health
#[derive(Debug, Clone)]
pub struct DeviceHealthAI {
    /// Current device health metrics
    metrics: Arc<Mutex<DeviceHealthMetrics>>,
    /// Current device health score
    score: Arc<Mutex<DeviceHealthScore>>,
    /// Configuration for health thresholds
    config: HealthConfig,
    /// Flag to indicate if the monitoring is running
    running: Arc<Mutex<bool>>,
    /// Model version for device health assessment
    model_version: String,
    /// Last time the model was updated
    model_last_updated: Instant,
}

/// Configuration for health threshold values
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Minimum battery level required for participation (percentage)
    pub min_battery_level: f32,
    /// Maximum battery temperature allowed (Celsius)
    pub max_battery_temp: f32,
    /// Maximum CPU usage threshold (percentage)
    pub max_cpu_usage: f32,
    /// Maximum RAM usage threshold (percentage)
    pub max_ram_usage: f32,
    /// Minimum free storage required (percentage)
    pub min_free_storage_percent: f32,
    /// Maximum network latency allowed (ms)
    pub max_network_latency: u32,
    /// How often to update device metrics (seconds)
    pub update_interval_secs: u64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            min_battery_level: 15.0,
            max_battery_temp: 40.0,
            max_cpu_usage: 80.0,
            max_ram_usage: 80.0,
            min_free_storage_percent: 10.0,
            max_network_latency: 200,
            update_interval_secs: 60,
        }
    }
}

/// Monitor system health and resources
pub struct DeviceMonitor {
    sys: System,
}

impl DeviceMonitor {
    /// Create a new device monitor
    pub fn new() -> Self {
        // Initialize with new_all() to get all system information
        let sys = System::new_all();
        Self { sys }
    }

    /// Get CPU usage
    pub fn get_cpu_usage(&mut self) -> f32 {
        self.sys.refresh_cpu();
        // Get average CPU usage across all cores
        let cpus = self.sys.cpus();
        if cpus.is_empty() {
            return 0.0;
        }
        
        let mut total = 0.0;
        for cpu in cpus {
            total += cpu.cpu_usage();
        }
        total / cpus.len() as f32
    }

    /// Get memory usage
    pub fn get_memory_usage(&mut self) -> (u64, u64) {
        self.sys.refresh_memory();
        (self.sys.used_memory(), self.sys.total_memory())
    }

    /// Get disk usage - simplified implementation
    pub fn get_disk_usage(&mut self) -> Vec<(String, u64, u64)> {
        // Refresh all system info
        self.sys.refresh_all();
        
        // Create dummy data since disks API might have compatibility issues
        let mut results = Vec::new();
        
        // Provide dummy data for disk usage
        results.push((
            "root".to_string(),
            500 * 1024 * 1024 * 1024, // 500 GB total
            400 * 1024 * 1024 * 1024, // 400 GB available
        ));
        results.push((
            "data".to_string(),
            1000 * 1024 * 1024 * 1024, // 1 TB total
            750 * 1024 * 1024 * 1024,  // 750 GB available
        ));
        
        results
    }
    
    /// Update all system info
    pub fn refresh_all(&mut self) {
        self.sys.refresh_all();
    }
    
    /// Get global CPU info
    pub fn get_global_cpu_info(&mut self) -> f32 {
        self.sys.refresh_cpu();
        self.sys.global_cpu_info().cpu_usage()
    }
}

impl DeviceHealthAI {
    /// Create a new Device Health AI instance
    pub fn new(_config: &Config) -> Self {
        let health_config = HealthConfig::default();
        
        Self {
            metrics: Arc::new(Mutex::new(DeviceHealthMetrics::default())),
            score: Arc::new(Mutex::new(DeviceHealthScore::default())),
            config: health_config,
            running: Arc::new(Mutex::new(false)),
            model_version: "1.0.0".to_string(),
            model_last_updated: Instant::now(),
        }
    }
    
    /// Start the device health monitoring
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Ok(());
        }
        
        *running = true;
        info!("Device Health AI monitoring started");
        
        // Clone Arc references for the background task
        let metrics = Arc::clone(&self.metrics);
        let score = Arc::clone(&self.score);
        let running_arc = Arc::clone(&self.running);
        let config = self.config.clone();
        
        // Spawn a background task for continuous monitoring
        tokio::spawn(async move {
            let update_interval = Duration::from_secs(config.update_interval_secs);
            
            while *running_arc.lock().unwrap() {
                // Update device metrics
                Self::update_device_metrics(&metrics, &config);
                
                // Calculate health score based on metrics
                Self::calculate_health_score(&metrics, &score, &config);
                
                // Wait for the next update interval
                tokio::time::sleep(update_interval).await;
            }
        });
        
        Ok(())
    }
    
    /// Stop the device health monitoring
    pub fn stop(&self) -> Result<()> {
        let mut running = self.running.lock().unwrap();
        *running = false;
        info!("Device Health AI monitoring stopped");
        Ok(())
    }
    
    /// Get current device metrics
    pub fn get_metrics(&self) -> DeviceHealthMetrics {
        self.metrics.lock().unwrap().clone()
    }
    
    /// Get current health score
    pub fn get_score(&self) -> DeviceHealthScore {
        self.score.lock().unwrap().clone()
    }
    
    /// Update device metrics
    fn update_device_metrics(metrics: &Arc<Mutex<DeviceHealthMetrics>>, _config: &HealthConfig) {
        let mut monitor = DeviceMonitor::new();
        let mut metrics = metrics.lock().unwrap();
        
        // Get CPU usage
        metrics.cpu_usage = monitor.get_cpu_usage();
        
        // Get RAM usage
        let (used_ram, total_ram) = monitor.get_memory_usage();
        metrics.ram_usage = (used_ram as f64 / total_ram as f64 * 100.0) as f32;
        
        // Get storage information
        let disk_info = monitor.get_disk_usage();
        let mut total_available = 0;
        let mut total_capacity = 0;
        
        for (_name, total, available) in disk_info {
            total_available += available;
            total_capacity += total;
        }
        
        metrics.available_storage = total_available;
        metrics.total_storage = total_capacity;
        
        // Update battery info (simulated for desktop systems)
        // In a real implementation, we'd use platform-specific APIs
        metrics.battery_level = 85.0; // Simulated value
        metrics.battery_temperature = 30.0; // Simulated value
        
        // Update network metrics (simulated)
        metrics.network_latency = 45;
        metrics.network_jitter = 8;
        
        // Record last updated time
        metrics.last_updated = std::time::SystemTime::now();
    }
    
    /// Calculate health score based on metrics
    fn calculate_health_score(
        metrics: &Arc<Mutex<DeviceHealthMetrics>>, 
        score: &Arc<Mutex<DeviceHealthScore>>,
        config: &HealthConfig
    ) {
        let metrics = metrics.lock().unwrap();
        let mut score = score.lock().unwrap();
        
        // Calculate battery score
        score.battery_score = 
            if metrics.battery_level < config.min_battery_level {
                0.0
            } else {
                metrics.battery_level / 100.0
            };
        
        // Adjust for temperature
        let temp_factor = 
            if metrics.battery_temperature > config.max_battery_temp {
                0.5
            } else {
                1.0 - (metrics.battery_temperature / (config.max_battery_temp * 2.0))
            };
        
        score.battery_score *= temp_factor;
        
        // Calculate performance score
        let cpu_score = 1.0 - (metrics.cpu_usage / 100.0);
        let ram_score = 1.0 - (metrics.ram_usage / 100.0);
        score.performance_score = (cpu_score + ram_score) / 2.0;
        
        // Calculate storage score
        let storage_percent = metrics.available_storage as f64 / metrics.total_storage as f64 * 100.0;
        score.storage_score = 
            if storage_percent < config.min_free_storage_percent as f64 {
                0.5
            } else {
                (storage_percent as f32) / 100.0
            };
        
        // Calculate network score
        score.network_score = 
            if metrics.network_latency > config.max_network_latency {
                0.6
            } else {
                1.0 - (metrics.network_latency as f32 / config.max_network_latency as f32)
            };
        
        // Calculate security score
        score.security_score = if metrics.is_rooted { 0.2 } else { 1.0 };
        
        // Calculate overall score (weighted average)
        score.overall_score = 
            score.battery_score * 0.2 +
            score.performance_score * 0.3 +
            score.storage_score * 0.2 +
            score.network_score * 0.2 +
            score.security_score * 0.1;
        
        // Determine health status
        score.status = 
            if score.overall_score > 0.8 {
                DeviceHealthStatus::Healthy
            } else if score.overall_score > 0.6 {
                DeviceHealthStatus::Warning
            } else if score.overall_score > 0.4 {
                DeviceHealthStatus::Degraded
            } else {
                DeviceHealthStatus::Critical
            };
    }
    
    /// Check if this device is eligible for validation
    pub fn is_eligible_for_validation(&self) -> bool {
        let score = self.score.lock().unwrap();
        score.status != DeviceHealthStatus::Critical
    }
    
    /// Get participation weight based on health
    pub fn get_participation_weight(&self) -> f32 {
        let score = self.score.lock().unwrap();
        
        match score.status {
            DeviceHealthStatus::Healthy => 1.0,
            DeviceHealthStatus::Warning => 0.8,
            DeviceHealthStatus::Degraded => 0.5,
            DeviceHealthStatus::Critical => 0.0,
        }
    }
    
    /// Update the health assessment model
    pub async fn update_model(&mut self, model_path: &str) -> Result<()> {
        // Load model from path (placeholder implementation)
        // In a real implementation, we'd load ML models here
        info!("Updating device health model from {}", model_path);
        self.model_version = "1.1.0".to_string();
        self.model_last_updated = Instant::now();
        Ok(())
    }
    
    /// Force update of metrics
    pub async fn update_metrics(&self) -> Result<()> {
        Self::update_device_metrics(&self.metrics, &self.config);
        Self::calculate_health_score(&self.metrics, &self.score, &self.config);
        Ok(())
    }
} 