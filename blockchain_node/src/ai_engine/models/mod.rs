pub mod neural_base;
pub mod bci_interface;
pub mod self_learning;
pub mod registry;
pub mod types;

// Re-export main types
pub use neural_base::{NeuralNetwork, NeuralConfig};
pub use bci_interface::{BCIModel, SignalParams, FilterParams as BciFilterParams, BCIOutput, Spike, Initialize};
pub use self_learning::{SelfLearningSystem, SelfLearningConfig};
pub use registry::ModelRegistry;
pub use types::Experience;

#[derive(Debug, Clone)]
pub struct FilterParams {
    pub high_pass: f32,
    pub low_pass: f32,
    pub notch: f32,
    pub order: usize,
} 