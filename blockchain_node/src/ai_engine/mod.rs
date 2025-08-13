pub mod advanced_detection;
pub mod config;
pub mod data_chunking;
#[cfg(test)]
pub mod data_chunking_tests;
pub mod device_health;
pub mod explainability;
pub mod fraud_detection;
pub mod models;
pub mod neural_network;
pub mod performance_monitor;
pub mod security;
pub mod user_identification;

// Re-export commonly used types (fixing import names to match actual exports)
pub use device_health::DeviceMonitor;
pub use fraud_detection::FraudDetectionConfig;
pub use neural_network::{
    ActivationType, AdvancedNeuralNetwork, InitMethod, LossFunction, NetworkConfig,
};
pub use performance_monitor::{
    NeuralMonitorConfig, QuantumNeuralMonitor, TrainingExample, TrainingStatistics,
};
pub use user_identification::UserIdentificationAI;
