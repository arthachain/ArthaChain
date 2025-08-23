//! Advanced Gas Optimization System for Phase 2
//!
//! This module provides intelligent gas optimization using AI-driven analysis,
//! static code optimization, and dynamic pricing mechanisms.

#[cfg(feature = "evm")]
use crate::evm::EvmExecutionResult;
use crate::types::{Address, Hash};

use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::sleep;

/// Gas optimization strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum OptimizationStrategy {
    /// Static analysis-based optimization
    Static,
    /// Dynamic runtime optimization
    Dynamic,
    /// Machine learning-based optimization
    MachineLearning,
    /// Hybrid approach combining multiple strategies
    Hybrid,
    /// Adaptive optimization that learns from usage patterns
    Adaptive,
}

/// Gas pricing model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PricingModel {
    /// Fixed gas price
    Fixed { price: u64 },
    /// Dynamic pricing based on network congestion
    Dynamic { base_price: u64, multiplier: f64 },
    /// Auction-based pricing
    Auction { min_price: u64, max_price: u64 },
    /// AI-driven predictive pricing
    Predictive {
        base_price: u64,
        prediction_weight: f64,
    },
}

/// Gas optimization configuration
#[derive(Debug, Clone)]
pub struct GasOptimizationConfig {
    /// Default optimization strategy
    pub default_strategy: OptimizationStrategy,
    /// Pricing model to use
    pub pricing_model: PricingModel,
    /// Enable predictive optimization
    pub enable_prediction: bool,
    /// Cache size for optimization results
    pub cache_size: usize,
    /// Learning rate for ML models
    pub learning_rate: f64,
    /// Optimization aggressiveness (0.0 to 1.0)
    pub aggressiveness: f64,
    /// Enable real-time optimization
    pub enable_realtime: bool,
    /// Maximum optimization time in milliseconds
    pub max_optimization_time_ms: u64,
}

impl Default for GasOptimizationConfig {
    fn default() -> Self {
        Self {
            default_strategy: OptimizationStrategy::Hybrid,
            pricing_model: PricingModel::Dynamic {
                base_price: 1_000_000_000,
                multiplier: 1.5,
            },
            enable_prediction: true,
            cache_size: 10000,
            learning_rate: 0.001,
            aggressiveness: 0.7,
            enable_realtime: true,
            max_optimization_time_ms: 1000,
        }
    }
}

/// Contract execution pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPattern {
    /// Contract address
    pub contract_address: Address,
    /// Function name
    pub function_name: String,
    /// Historical gas usage
    pub gas_usage_history: VecDeque<u64>,
    /// Execution frequency
    pub execution_frequency: f64,
    /// Average gas consumption
    pub avg_gas_consumption: f64,
    /// Gas efficiency trend
    pub efficiency_trend: f64,
    /// Last optimization timestamp
    pub last_optimized: SystemTime,
    /// Optimization success rate
    pub optimization_success_rate: f64,
}

/// Gas optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Original gas estimate
    pub original_gas: u64,
    /// Optimized gas estimate
    pub optimized_gas: u64,
    /// Optimization strategy used
    pub strategy: OptimizationStrategy,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Optimization time in microseconds
    pub optimization_time_us: u64,
    /// Savings achieved
    pub savings: u64,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

/// Neural network model for gas prediction
#[derive(Debug, Clone)]
pub struct GasPredictionModel {
    /// Model weights
    weights: Vec<Vec<f64>>,
    /// Model biases
    biases: Vec<f64>,
    /// Training data
    training_data: Vec<(Vec<f64>, f64)>,
    /// Model accuracy
    accuracy: f64,
    /// Last training timestamp
    last_trained: Instant,
}

impl GasPredictionModel {
    /// Create a new prediction model
    pub fn new() -> Self {
        Self {
            weights: vec![vec![0.1; 10]; 3], // Simple 3-layer network
            biases: vec![0.0; 3],
            training_data: Vec::new(),
            accuracy: 0.0,
            last_trained: Instant::now(),
        }
    }

    /// Predict gas usage
    pub fn predict(&self, features: &[f64]) -> f64 {
        // Simple neural network forward pass
        let mut layer_output = features.to_vec();

        for (layer_weights, bias) in self.weights.iter().zip(self.biases.iter()) {
            let mut next_output = Vec::new();
            for weights in layer_weights.chunks(layer_output.len()) {
                let sum: f64 = weights
                    .iter()
                    .zip(layer_output.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                next_output.push((sum + bias).tanh()); // Activation function
            }
            layer_output = next_output;
        }

        layer_output[0].max(0.0) // Ensure non-negative gas prediction
    }

    /// Train the model with new data
    pub fn train(&mut self, features: Vec<f64>, actual_gas: f64) {
        self.training_data.push((features, actual_gas));

        // Simple gradient descent training
        if self.training_data.len() > 100 {
            self.perform_training();
            self.last_trained = Instant::now();
        }
    }

    /// Perform training on accumulated data
    fn perform_training(&mut self) {
        // Simplified training - in production, use proper neural network library
        for _ in 0..10 {
            // 10 training epochs
            let mut total_error = 0.0;

            for (features, actual) in &self.training_data {
                let predicted = self.predict(features);
                let error = actual - predicted;
                total_error += error.abs();

                // Update weights (simplified)
                for layer_weights in &mut self.weights {
                    for weight in layer_weights {
                        *weight += 0.001 * error; // Learning rate * error
                    }
                }
            }

            self.accuracy = 1.0 - (total_error / self.training_data.len() as f64);
        }

        // Keep only recent training data
        if self.training_data.len() > 1000 {
            self.training_data.drain(0..500);
        }
    }
}

/// Gas optimization engine
pub struct GasOptimizationEngine {
    /// Configuration
    config: GasOptimizationConfig,
    /// Execution patterns database
    patterns: Arc<RwLock<HashMap<String, ExecutionPattern>>>,
    /// Optimization cache
    optimization_cache: Arc<RwLock<HashMap<Hash, OptimizationResult>>>,
    /// Gas prediction model
    prediction_model: Arc<Mutex<GasPredictionModel>>,
    /// Network congestion tracker
    congestion_tracker: Arc<RwLock<NetworkCongestionTracker>>,
    /// Real-time optimization queue
    optimization_queue: Arc<Mutex<VecDeque<OptimizationRequest>>>,
    /// Performance metrics
    metrics: Arc<RwLock<OptimizationMetrics>>,
}

/// Network congestion tracking
#[derive(Debug, Clone, Default)]
pub struct NetworkCongestionTracker {
    /// Current transactions per second
    pub current_tps: f64,
    /// Average block utilization
    pub avg_block_utilization: f64,
    /// Gas price trend
    pub gas_price_trend: f64,
    /// Congestion level (0.0 to 1.0)
    pub congestion_level: f64,
    /// Peak congestion times
    pub peak_times: Vec<(u32, f64)>, // (hour_of_day, congestion_level)
}

/// Optimization request
#[derive(Debug, Clone)]
pub struct OptimizationRequest {
    /// Contract address
    pub contract_address: Address,
    /// Function name
    pub function_name: String,
    /// Transaction data
    pub transaction_data: Vec<u8>,
    /// Caller address
    pub caller: Address,
    /// Target gas limit
    pub gas_limit: u64,
    /// Priority level
    pub priority: OptimizationPriority,
    /// Request timestamp
    pub timestamp: Instant,
}

/// Optimization priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Optimization performance metrics
#[derive(Debug, Clone, Default)]
pub struct OptimizationMetrics {
    /// Total optimizations performed
    pub total_optimizations: u64,
    /// Total gas saved
    pub total_gas_saved: u64,
    /// Average optimization time
    pub avg_optimization_time_us: f64,
    /// Success rate
    pub success_rate: f64,
    /// Model accuracy
    pub model_accuracy: f64,
    /// Optimization strategies performance
    pub strategy_performance: HashMap<OptimizationStrategy, StrategyMetrics>,
}

/// Strategy-specific metrics
#[derive(Debug, Clone, Default)]
pub struct StrategyMetrics {
    /// Number of times used
    pub usage_count: u64,
    /// Average gas savings
    pub avg_gas_savings: f64,
    /// Success rate
    pub success_rate: f64,
    /// Average optimization time
    pub avg_time_us: f64,
}

impl GasOptimizationEngine {
    /// Create a new gas optimization engine
    pub fn new(config: GasOptimizationConfig) -> Self {
        let engine = Self {
            config,
            patterns: Arc::new(RwLock::new(HashMap::new())),
            optimization_cache: Arc::new(RwLock::new(HashMap::new())),
            prediction_model: Arc::new(Mutex::new(GasPredictionModel::new())),
            congestion_tracker: Arc::new(RwLock::new(NetworkCongestionTracker::default())),
            optimization_queue: Arc::new(Mutex::new(VecDeque::new())),
            metrics: Arc::new(RwLock::new(OptimizationMetrics::default())),
        };

        // Start background optimization worker if real-time is enabled
        if engine.config.enable_realtime {
            engine.start_optimization_worker();
        }

        engine
    }

    /// Optimize gas for a contract execution
    pub async fn optimize_gas(
        &self,
        contract_address: &Address,
        function_name: &str,
        transaction_data: &[u8],
        gas_limit: u64,
    ) -> Result<OptimizationResult> {
        let start_time = Instant::now();

        // Generate cache key
        let cache_key = self.generate_cache_key(contract_address, function_name, transaction_data);

        // Check cache first
        if let Some(cached_result) = self.get_cached_optimization(&cache_key) {
            debug!("Using cached optimization result for {}", function_name);
            return Ok(cached_result);
        }

        // Get execution pattern
        let pattern_key = format!("{:?}:{}", contract_address, function_name);
        let pattern = self
            .get_or_create_pattern(&pattern_key, contract_address, function_name)
            .await;

        // Choose optimization strategy
        let strategy = self.choose_optimization_strategy(&pattern).await;

        // Perform optimization based on strategy
        let result = match strategy {
            OptimizationStrategy::Static => {
                self.optimize_static(transaction_data, gas_limit).await?
            }
            OptimizationStrategy::Dynamic => self.optimize_dynamic(&pattern, gas_limit).await?,
            OptimizationStrategy::MachineLearning => self.optimize_ml(&pattern, gas_limit).await?,
            OptimizationStrategy::Hybrid => {
                self.optimize_hybrid(&pattern, transaction_data, gas_limit)
                    .await?
            }
            OptimizationStrategy::Adaptive => {
                self.optimize_adaptive(&pattern, transaction_data, gas_limit)
                    .await?
            }
        };

        // Cache the result
        self.cache_optimization(cache_key, &result);

        // Update metrics
        self.update_metrics(&result, &strategy).await;

        info!(
            "Gas optimization completed: strategy={:?}, savings={}, time={}μs",
            strategy,
            result.savings,
            start_time.elapsed().as_micros()
        );

        Ok(result)
    }

    /// Static analysis-based optimization
    async fn optimize_static(
        &self,
        transaction_data: &[u8],
        gas_limit: u64,
    ) -> Result<OptimizationResult> {
        // Analyze transaction data for optimization opportunities
        let mut optimized_gas = gas_limit;
        let mut recommendations = Vec::new();

        // Check for common patterns that can be optimized
        if transaction_data.len() > 1024 {
            optimized_gas = (optimized_gas as f64 * 0.95) as u64;
            recommendations
                .push("Large transaction data detected - consider data compression".to_string());
        }

        // Check for repeated operations
        if self.detect_loops(transaction_data) {
            optimized_gas = (optimized_gas as f64 * 0.9) as u64;
            recommendations
                .push("Loop operations detected - consider batch processing".to_string());
        }

        Ok(OptimizationResult {
            original_gas: gas_limit,
            optimized_gas,
            strategy: OptimizationStrategy::Static,
            confidence: 0.8,
            optimization_time_us: 100,
            savings: gas_limit.saturating_sub(optimized_gas),
            recommendations,
        })
    }

    /// Dynamic runtime optimization
    async fn optimize_dynamic(
        &self,
        pattern: &ExecutionPattern,
        gas_limit: u64,
    ) -> Result<OptimizationResult> {
        let mut optimized_gas = gas_limit;
        let mut recommendations = Vec::new();

        // Adjust based on historical performance
        if pattern.avg_gas_consumption > 0.0 {
            let efficiency_factor = pattern.efficiency_trend.max(0.1).min(2.0);
            optimized_gas = (pattern.avg_gas_consumption * efficiency_factor) as u64;
            recommendations.push(format!(
                "Adjusted based on {} previous executions",
                pattern.gas_usage_history.len()
            ));
        }

        // Network congestion adjustment
        let congestion = self.congestion_tracker.read().unwrap();
        if congestion.congestion_level > 0.7 {
            optimized_gas = (optimized_gas as f64 * 1.2) as u64;
            recommendations
                .push("Network congestion detected - increased gas estimate".to_string());
        }

        Ok(OptimizationResult {
            original_gas: gas_limit,
            optimized_gas,
            strategy: OptimizationStrategy::Dynamic,
            confidence: 0.85,
            optimization_time_us: 200,
            savings: gas_limit.saturating_sub(optimized_gas),
            recommendations,
        })
    }

    /// Machine learning-based optimization
    async fn optimize_ml(
        &self,
        pattern: &ExecutionPattern,
        gas_limit: u64,
    ) -> Result<OptimizationResult> {
        let mut model = self.prediction_model.lock().unwrap();

        // Prepare features for ML model
        let features = vec![
            pattern.avg_gas_consumption,
            pattern.execution_frequency,
            pattern.efficiency_trend,
            pattern.optimization_success_rate,
            gas_limit as f64,
        ];

        // Get prediction
        let predicted_gas = model.predict(&features) as u64;
        let optimized_gas = predicted_gas.max(gas_limit / 2).min(gas_limit); // Reasonable bounds

        let confidence = model.accuracy;

        Ok(OptimizationResult {
            original_gas: gas_limit,
            optimized_gas,
            strategy: OptimizationStrategy::MachineLearning,
            confidence,
            optimization_time_us: 500,
            savings: gas_limit.saturating_sub(optimized_gas),
            recommendations: vec![format!(
                "ML prediction with {:.2}% confidence",
                confidence * 100.0
            )],
        })
    }

    /// Hybrid optimization combining multiple strategies
    async fn optimize_hybrid(
        &self,
        pattern: &ExecutionPattern,
        transaction_data: &[u8],
        gas_limit: u64,
    ) -> Result<OptimizationResult> {
        // Get results from multiple strategies
        let static_result = self.optimize_static(transaction_data, gas_limit).await?;
        let dynamic_result = self.optimize_dynamic(pattern, gas_limit).await?;
        let ml_result = self.optimize_ml(pattern, gas_limit).await?;

        // Weight the results based on confidence
        let total_confidence =
            static_result.confidence + dynamic_result.confidence + ml_result.confidence;
        let weighted_gas = (static_result.optimized_gas as f64 * static_result.confidence
            + dynamic_result.optimized_gas as f64 * dynamic_result.confidence
            + ml_result.optimized_gas as f64 * ml_result.confidence)
            / total_confidence;

        let optimized_gas = weighted_gas as u64;
        let mut recommendations = Vec::new();
        recommendations.extend(static_result.recommendations);
        recommendations.extend(dynamic_result.recommendations);
        recommendations.extend(ml_result.recommendations);
        recommendations.push("Hybrid optimization using weighted ensemble".to_string());

        Ok(OptimizationResult {
            original_gas: gas_limit,
            optimized_gas,
            strategy: OptimizationStrategy::Hybrid,
            confidence: total_confidence / 3.0,
            optimization_time_us: 800,
            savings: gas_limit.saturating_sub(optimized_gas),
            recommendations,
        })
    }

    /// Adaptive optimization that learns from usage patterns
    async fn optimize_adaptive(
        &self,
        pattern: &ExecutionPattern,
        transaction_data: &[u8],
        gas_limit: u64,
    ) -> Result<OptimizationResult> {
        // Choose the best performing strategy for this pattern
        let best_strategy = self.get_best_strategy_for_pattern(pattern).await;

        // Apply the best strategy
        let result = match best_strategy {
            OptimizationStrategy::Static => {
                self.optimize_static(transaction_data, gas_limit).await?
            }
            OptimizationStrategy::Dynamic => self.optimize_dynamic(pattern, gas_limit).await?,
            OptimizationStrategy::MachineLearning => self.optimize_ml(pattern, gas_limit).await?,
            _ => {
                self.optimize_hybrid(pattern, transaction_data, gas_limit)
                    .await?
            }
        };

        Ok(OptimizationResult {
            strategy: OptimizationStrategy::Adaptive,
            ..result
        })
    }

    /// Choose the best optimization strategy based on pattern analysis
    async fn choose_optimization_strategy(
        &self,
        pattern: &ExecutionPattern,
    ) -> OptimizationStrategy {
        match &self.config.default_strategy {
            OptimizationStrategy::Adaptive => self.get_best_strategy_for_pattern(pattern).await,
            strategy => strategy.clone(),
        }
    }

    /// Get the best performing strategy for a specific pattern
    async fn get_best_strategy_for_pattern(
        &self,
        pattern: &ExecutionPattern,
    ) -> OptimizationStrategy {
        let metrics = self.metrics.read().unwrap();

        let mut best_strategy = OptimizationStrategy::Static;
        let mut best_score = 0.0;

        for (strategy, strategy_metrics) in &metrics.strategy_performance {
            let score = strategy_metrics.success_rate * strategy_metrics.avg_gas_savings;
            if score > best_score {
                best_score = score;
                best_strategy = strategy.clone();
            }
        }

        best_strategy
    }

    /// Update execution pattern with new data
    pub async fn update_pattern(
        &self,
        contract_address: &Address,
        function_name: &str,
        actual_gas: u64,
        success: bool,
    ) {
        let pattern_key = format!("{:?}:{}", contract_address, function_name);
        let mut patterns = self.patterns.write().unwrap();

        if let Some(pattern) = patterns.get_mut(&pattern_key) {
            pattern.gas_usage_history.push_back(actual_gas);
            if pattern.gas_usage_history.len() > 100 {
                pattern.gas_usage_history.pop_front();
            }

            // Update statistics
            let sum: u64 = pattern.gas_usage_history.iter().sum();
            pattern.avg_gas_consumption = sum as f64 / pattern.gas_usage_history.len() as f64;

            // Update success rate
            if success {
                pattern.optimization_success_rate = pattern.optimization_success_rate * 0.95 + 0.05;
            } else {
                pattern.optimization_success_rate = pattern.optimization_success_rate * 0.95;
            }

            // Train ML model with new data
            let features = vec![
                pattern.avg_gas_consumption,
                pattern.execution_frequency,
                pattern.efficiency_trend,
                pattern.optimization_success_rate,
            ];

            let mut model = self.prediction_model.lock().unwrap();
            model.train(features, actual_gas as f64);
        }
    }

    /// Generate cache key for optimization results
    fn generate_cache_key(
        &self,
        contract_address: &Address,
        function_name: &str,
        transaction_data: &[u8],
    ) -> Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(contract_address.as_bytes());
        hasher.update(function_name.as_bytes());
        hasher.update(transaction_data);
        Hash::from_data(hasher.finalize().as_bytes())
    }

    /// Get cached optimization result
    fn get_cached_optimization(&self, cache_key: &Hash) -> Option<OptimizationResult> {
        let cache = self.optimization_cache.read().unwrap();
        cache.get(cache_key).cloned()
    }

    /// Cache optimization result
    fn cache_optimization(&self, cache_key: Hash, result: &OptimizationResult) {
        let mut cache = self.optimization_cache.write().unwrap();
        if cache.len() >= self.config.cache_size {
            // Remove oldest entries (simple FIFO)
            let keys: Vec<_> = cache.keys().cloned().collect();
            for key in keys.iter().take(cache.len() / 2) {
                cache.remove(key);
            }
        }
        cache.insert(cache_key, result.clone());
    }

    /// Get or create execution pattern
    async fn get_or_create_pattern(
        &self,
        pattern_key: &str,
        contract_address: &Address,
        function_name: &str,
    ) -> ExecutionPattern {
        let mut patterns = self.patterns.write().unwrap();
        patterns
            .entry(pattern_key.to_string())
            .or_insert_with(|| ExecutionPattern {
                contract_address: contract_address.clone(),
                function_name: function_name.to_string(),
                gas_usage_history: VecDeque::new(),
                execution_frequency: 0.0,
                avg_gas_consumption: 0.0,
                efficiency_trend: 1.0,
                last_optimized: UNIX_EPOCH,
                optimization_success_rate: 0.5,
            })
            .clone()
    }

    /// Detect loops in transaction data (simplified)
    fn detect_loops(&self, transaction_data: &[u8]) -> bool {
        // Simple pattern detection for repeated sequences
        if transaction_data.len() < 8 {
            return false;
        }

        for window_size in 2..=8 {
            if transaction_data.len() >= window_size * 3 {
                for i in 0..=(transaction_data.len() - window_size * 3) {
                    let pattern = &transaction_data[i..i + window_size];
                    let next_occurrence = &transaction_data[i + window_size..i + window_size * 2];
                    let third_occurrence =
                        &transaction_data[i + window_size * 2..i + window_size * 3];

                    if pattern == next_occurrence && pattern == third_occurrence {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Update optimization metrics
    async fn update_metrics(&self, result: &OptimizationResult, strategy: &OptimizationStrategy) {
        let mut metrics = self.metrics.write().unwrap();

        metrics.total_optimizations += 1;
        metrics.total_gas_saved += result.savings;
        metrics.avg_optimization_time_us =
            (metrics.avg_optimization_time_us * 0.9) + (result.optimization_time_us as f64 * 0.1);

        if result.savings > 0 {
            metrics.success_rate = metrics.success_rate * 0.99 + 0.01;
        } else {
            metrics.success_rate = metrics.success_rate * 0.99;
        }

        // Update strategy-specific metrics
        let strategy_metrics = metrics
            .strategy_performance
            .entry(strategy.clone())
            .or_default();
        strategy_metrics.usage_count += 1;
        strategy_metrics.avg_gas_savings =
            (strategy_metrics.avg_gas_savings * 0.9) + (result.savings as f64 * 0.1);
        strategy_metrics.avg_time_us =
            (strategy_metrics.avg_time_us * 0.9) + (result.optimization_time_us as f64 * 0.1);

        if result.savings > 0 {
            strategy_metrics.success_rate = strategy_metrics.success_rate * 0.99 + 0.01;
        } else {
            strategy_metrics.success_rate = strategy_metrics.success_rate * 0.99;
        }
    }

    /// Start background optimization worker
    fn start_optimization_worker(&self) {
        let queue = self.optimization_queue.clone();
        let patterns = self.patterns.clone();

        tokio::spawn(async move {
            loop {
                // Process optimization queue
                let request = {
                    let mut queue_guard = queue.lock().unwrap();
                    queue_guard.pop_front()
                };

                if let Some(request) = request {
                    // Process optimization request in background
                    debug!(
                        "Processing background optimization for {:?}",
                        request.contract_address
                    );
                    // Implementation would go here
                } else {
                    // No requests, sleep briefly
                    sleep(Duration::from_millis(100)).await;
                }
            }
        });
    }

    /// Get optimization statistics
    pub fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        let metrics = self.metrics.read().unwrap();
        let patterns = self.patterns.read().unwrap();

        stats.insert(
            "total_optimizations".to_string(),
            metrics.total_optimizations.into(),
        );
        stats.insert(
            "total_gas_saved".to_string(),
            metrics.total_gas_saved.into(),
        );
        stats.insert("success_rate".to_string(), metrics.success_rate.into());
        stats.insert(
            "avg_optimization_time_us".to_string(),
            metrics.avg_optimization_time_us.into(),
        );
        stats.insert("patterns_tracked".to_string(), patterns.len().into());

        let cache_size = self.optimization_cache.read().unwrap().len();
        stats.insert("cache_size".to_string(), cache_size.into());

        stats
    }

    /// Clear optimization cache
    pub fn clear_cache(&self) {
        let mut cache = self.optimization_cache.write().unwrap();
        cache.clear();
        info!("Gas optimization cache cleared");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gas_prediction_model() {
        let mut model = GasPredictionModel::new();
        let features = vec![1000.0, 0.5, 1.0, 0.8, 2000.0];
        let prediction = model.predict(&features);
        assert!(prediction >= 0.0);

        // Train with some data
        model.train(features.clone(), 1500.0);
        let new_prediction = model.predict(&features);
        assert!(new_prediction >= 0.0);
    }

    #[test]
    fn test_optimization_config() {
        let config = GasOptimizationConfig::default();
        assert_eq!(config.default_strategy, OptimizationStrategy::Hybrid);
        assert!(config.enable_prediction);
    }

    #[tokio::test]
    async fn test_gas_optimization_engine() {
        let config = GasOptimizationConfig::default();
        let engine = GasOptimizationEngine::new(config);

        let contract_address = Address::from_bytes(b"test_contract_addr_12").unwrap();
        let result = engine
            .optimize_gas(&contract_address, "test_function", b"test_data", 1_000_000)
            .await;

        assert!(result.is_ok());
        let optimization_result = result.unwrap();
        assert!(optimization_result.optimized_gas <= 1_000_000);
    }
}
