use serde::{Deserialize, Serialize};

/// Types of activation functions supported by neural networks
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ELU,
    GELU,
}

/// Parameters for filtering and processing data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterParams {
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Maximum number of results to return
    pub max_results: usize,
    /// Whether to sort results by confidence
    pub sort_by_confidence: bool,
    /// Minimum similarity threshold
    pub min_similarity: f32,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_results: 100,
            sort_by_confidence: true,
            min_similarity: 0.7,
        }
    }
}

/// Experience data for reinforcement learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Current state
    pub state: Vec<f32>,
    /// Action taken
    pub action: usize,
    /// Reward received
    pub reward: f32,
    /// Next state
    pub next_state: Vec<f32>,
    /// Whether the episode ended
    pub done: bool,
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Loss value
    pub loss: f32,
    /// Accuracy
    pub accuracy: f32,
    /// Number of training steps
    pub steps: u64,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average reward
    pub avg_reward: f32,
    /// Success rate
    pub success_rate: f32,
    /// Number of episodes
    pub episodes: u64,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// GPU memory usage in bytes (if available)
    pub gpu_memory: Option<u64>,
}
