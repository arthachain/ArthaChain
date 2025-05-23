pub mod bci_interface;
pub mod neural_base;
pub mod registry;
pub mod self_learning;
pub mod types;

// Re-export main types
pub use bci_interface::{
    BCIModel, BCIOutput, FilterParams as BciFilterParams, Initialize, SignalParams, Spike,
};
pub use neural_base::{NeuralConfig, NeuralNetwork};
pub use registry::ModelRegistry;
pub use self_learning::{SelfLearningConfig, SelfLearningSystem};
pub use types::Experience;

#[derive(Debug, Clone)]
pub struct FilterParams {
    pub high_pass: f32,
    pub low_pass: f32,
    pub notch: f32,
    pub order: usize,
}
