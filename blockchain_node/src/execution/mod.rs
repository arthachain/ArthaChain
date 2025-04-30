// Execution module for handling transaction processing

pub mod parallel;
pub mod executor;

// Re-export key types
pub use parallel::{ParallelExecutionManager, ParallelConfig, ConflictStrategy};
pub use executor::TransactionExecutor; 