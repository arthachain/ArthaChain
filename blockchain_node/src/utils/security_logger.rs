use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Serialize, Deserialize};
use log::{error, warn, info, debug};
use std::fs::OpenOptions;
use std::path::Path;
use std::io::Write;
use anyhow::{Result, Context};

/// Security event severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityLevel {
    /// Informational security event
    Info,
    /// Low risk security event
    Low,
    /// Medium risk security event
    Medium,
    /// High risk security event
    High,
    /// Critical security event
    Critical,
}

/// Security event categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityCategory {
    /// Authentication related events
    Authentication,
    /// Authorization related events
    Authorization,
    /// Consensus related events
    Consensus,
    /// Network related events
    Network,
    /// Storage related events
    Storage,
    /// Smart contract related events
    SmartContract,
    /// Node behavior related events
    NodeBehavior,
    /// API related events
    Api,
    /// System related events
    System,
}

/// Security event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    /// Timestamp of the event
    pub timestamp: u64,
    /// Security level
    pub level: SecurityLevel,
    /// Category of the security event
    pub category: SecurityCategory,
    /// Associated node ID if applicable
    pub node_id: Option<String>,
    /// Event message
    pub message: String,
    /// Additional structured data
    pub data: serde_json::Value,
}

/// Security logger for recording security events
pub struct SecurityLogger {
    /// Path to the security log file
    log_path: String,
    /// In-memory cache of recent events
    recent_events: Arc<Mutex<Vec<SecurityEvent>>>,
    /// Maximum number of events to keep in memory
    max_events: usize,
}

impl SecurityLogger {
    /// Create a new security logger
    pub fn new(log_path: &str, max_events: usize) -> Result<Self> {
        // Create log directory if it doesn't exist
        if let Some(parent) = Path::new(log_path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Create or open the log file to ensure it's writable
        let _file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .context("Failed to open security log file")?;
        
        Ok(Self {
            log_path: log_path.to_string(),
            recent_events: Arc::new(Mutex::new(Vec::with_capacity(max_events))),
            max_events,
        })
    }
    
    /// Log a security event
    pub async fn log_event(
        &self,
        level: SecurityLevel,
        category: SecurityCategory,
        node_id: Option<&str>,
        message: &str,
        data: serde_json::Value,
    ) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Time went backwards")?
            .as_secs();
        
        let event = SecurityEvent {
            timestamp: now,
            level,
            category,
            node_id: node_id.map(String::from),
            message: message.to_string(),
            data,
        };
        
        // Log to standard logger based on severity
        match level {
            SecurityLevel::Critical | SecurityLevel::High => {
                error!(
                    "[SECURITY][{:?}] {}: {}",
                    category, 
                    node_id.unwrap_or("-"), 
                    message
                );
            }
            SecurityLevel::Medium => {
                warn!(
                    "[SECURITY][{:?}] {}: {}",
                    category, 
                    node_id.unwrap_or("-"), 
                    message
                );
            }
            SecurityLevel::Low => {
                info!(
                    "[SECURITY][{:?}] {}: {}",
                    category, 
                    node_id.unwrap_or("-"), 
                    message
                );
            }
            SecurityLevel::Info => {
                debug!(
                    "[SECURITY][{:?}] {}: {}",
                    category, 
                    node_id.unwrap_or("-"), 
                    message
                );
            }
        }
        
        // Write to log file
        let event_json = serde_json::to_string(&event)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
            .context("Failed to open security log file")?;
        
        writeln!(file, "{}", event_json)
            .context("Failed to write to security log file")?;
        
        // Update in-memory cache
        let mut events = self.recent_events.lock().await;
        events.push(event);
        
        // Trim if exceeding max size
        if events.len() > self.max_events {
            events.remove(0);
        }
        
        Ok(())
    }
    
    /// Get recent security events
    pub async fn get_recent_events(&self) -> Vec<SecurityEvent> {
        let events = self.recent_events.lock().await;
        events.clone()
    }
    
    /// Get recent events by security level
    pub async fn get_events_by_level(&self, level: SecurityLevel) -> Vec<SecurityEvent> {
        let events = self.recent_events.lock().await;
        events.iter()
            .filter(|e| e.level == level)
            .cloned()
            .collect()
    }
    
    /// Get recent events by category
    pub async fn get_events_by_category(&self, category: SecurityCategory) -> Vec<SecurityEvent> {
        let events = self.recent_events.lock().await;
        events.iter()
            .filter(|e| e.category == category)
            .cloned()
            .collect()
    }
    
    /// Get events for a specific node
    pub async fn get_events_by_node(&self, node_id: &str) -> Vec<SecurityEvent> {
        let events = self.recent_events.lock().await;
        events.iter()
            .filter(|e| e.node_id.as_deref() == Some(node_id))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_security_logger() {
        let temp_dir = tempdir().unwrap();
        let log_path = temp_dir.path().join("security.log");
        
        let logger = SecurityLogger::new(
            log_path.to_str().unwrap(),
            100
        ).unwrap();
        
        // Log a test event
        logger.log_event(
            SecurityLevel::Medium,
            SecurityCategory::Consensus,
            Some("test-node"),
            "Suspicious block proposal",
            serde_json::json!({
                "block_height": 1000,
                "hash": "0x1234567890abcdef"
            })
        ).await.unwrap();
        
        // Verify event was recorded
        let events = logger.get_recent_events().await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].level, SecurityLevel::Medium);
        assert_eq!(events[0].category, SecurityCategory::Consensus);
        assert_eq!(events[0].node_id, Some("test-node".to_string()));
        
        // Verify file was written
        let content = std::fs::read_to_string(log_path).unwrap();
        assert!(content.contains("Suspicious block proposal"));
    }
} 