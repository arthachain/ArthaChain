use std::sync::RwLock;
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use log::info;

/// Network condition metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Current block time
    pub block_time: u64,
    /// Network latency
    pub latency: u64,
    /// Network throughput
    pub throughput: u64,
    /// Number of active validators
    pub active_validators: u64,
    /// Network load
    pub network_load: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Difficulty adjustment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyConfig {
    /// Target block time in seconds
    pub target_block_time: u64,
    /// Maximum block time
    pub max_block_time: f64,
    /// Minimum block time
    pub min_block_time: f64,
    /// Difficulty adjustment factor
    pub adjustment_factor: f64,
    /// Maximum difficulty adjustment per block
    pub max_adjustment: f64,
    /// Minimum difficulty
    pub min_difficulty: u64,
    /// Maximum difficulty
    pub max_difficulty: u64,
    /// Metrics history size
    pub metrics_history_size: usize,
    /// Adjustment period
    pub adjustment_period: u64,
}

/// Difficulty manager state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyState {
    /// Current difficulty
    pub current_difficulty: u64,
    /// Network metrics history
    pub metrics_history: Vec<NetworkMetrics>,
    /// Configuration
    pub config: DifficultyConfig,
}

/// Difficulty manager
pub struct DifficultyManager {
    state: RwLock<DifficultyState>,
}

impl DifficultyManager {
    /// Create a new difficulty manager
    pub fn new(config: DifficultyConfig) -> Self {
        let state = DifficultyState {
            current_difficulty: config.min_difficulty,
            metrics_history: Vec::new(),
            config,
        };
        Self {
            state: RwLock::new(state),
        }
    }

    /// Adjust difficulty based on network conditions
    pub fn adjust_difficulty(&self, metrics: NetworkMetrics) -> Result<()> {
        let mut state = self.state.write().map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
        
        // Add metrics to history
        state.metrics_history.push(metrics.clone());
        
        // Keep only recent metrics
        if state.metrics_history.len() > state.config.adjustment_period as usize {
            state.metrics_history.remove(0);
        }
        
        // Calculate average block time
        let avg_block_time = state.metrics_history.iter()
            .map(|m| m.block_time)
            .sum::<u64>() as f64 / state.metrics_history.len() as f64;
        
        // Adjust difficulty based on block time
        let target_time = state.config.target_block_time as f64;
        let max_adjustment = state.config.max_adjustment;
        
        let adjustment = if avg_block_time > target_time {
            // Block time too high, decrease difficulty
            (target_time / avg_block_time).max(1.0 - max_adjustment)
        } else {
            // Block time too low, increase difficulty
            (avg_block_time / target_time).min(1.0 + max_adjustment)
        };
        
        // Apply adjustment
        let new_difficulty = ((state.current_difficulty as f64 * adjustment) as u64)
            .max(state.config.min_difficulty)
            .min(state.config.max_difficulty);
            
        info!("Adjusting difficulty from {} to {}", state.current_difficulty, new_difficulty);
        state.current_difficulty = new_difficulty;

        Ok(())
    }

    /// Get current difficulty
    pub fn get_current_difficulty(&self) -> Result<u64> {
        let state = self.state.read().map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
        Ok(state.current_difficulty)
    }

    /// Get network metrics history
    pub fn get_metrics_history(&self) -> Result<Vec<NetworkMetrics>> {
        let state = self.state.read().map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
        Ok(state.metrics_history.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> DifficultyConfig {
        DifficultyConfig {
            target_block_time: 10,
            adjustment_period: 10,
            max_adjustment: 0.5,
            min_difficulty: 1000,
            max_difficulty: 1000000,
            max_block_time: 10.0,
            min_block_time: 5.0,
            metrics_history_size: 10,
            adjustment_factor: 0.25,
        }
    }

    fn create_test_metrics(block_time: u64) -> NetworkMetrics {
        NetworkMetrics {
            block_time,
            latency: 100,
            throughput: 1000,
            active_validators: 10,
            network_load: 0.5,
            timestamp: 12345,
        }
    }

    #[test]
    fn test_difficulty_adjustment() {
        let manager = DifficultyManager::new(create_test_config());
        
        // Initial difficulty should be minimum
        assert_eq!(manager.get_current_difficulty().unwrap(), 1000);
        
        // Add metrics with high block time
        manager.adjust_difficulty(create_test_metrics(20)).unwrap();
        
        // Difficulty should decrease, but not below min_difficulty
        let new_difficulty = manager.get_current_difficulty().unwrap();
        assert!(new_difficulty >= 1000);
        
        // Add metrics with low block time
        manager.adjust_difficulty(create_test_metrics(5)).unwrap();
        
        // Difficulty should increase
        let final_difficulty = manager.get_current_difficulty().unwrap();
        assert!(final_difficulty >= new_difficulty);
    }

    #[test]
    fn test_metrics_history() {
        let manager = DifficultyManager::new(create_test_config());
        
        // Add some metrics
        for i in 0..5 {
            manager.adjust_difficulty(create_test_metrics(10 + i as u64)).unwrap();
        }
        
        // Check history length
        let history = manager.get_metrics_history().unwrap();
        assert_eq!(history.len(), 5);
        
        // Check values are stored correctly
        for (i, metrics) in history.iter().enumerate() {
            assert_eq!(metrics.block_time, 10 + i as u64);
        }
    }
} 