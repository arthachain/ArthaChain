use crate::wasm::types::WasmError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, Duration};

/// Gas costs for various operations
pub struct GasCosts {
    /// Base cost for any operation
    pub base: u64,
    /// Cost per byte for memory allocation
    pub memory_byte: u64,
    /// Cost per byte for storage read
    pub storage_read_byte: u64,
    /// Cost per byte for storage write
    pub storage_write_byte: u64,
    /// Cost per byte for storage delete
    pub storage_delete_byte: u64,
    /// Cost per byte for computation
    pub compute_byte: u64,
    /// Cost for external function call
    pub external_call: u64,
}

impl Default for GasCosts {
    fn default() -> Self {
        Self {
            base: 1,
            memory_byte: 1,
            storage_read_byte: 5,
            storage_write_byte: 10,
            storage_delete_byte: 5,
            compute_byte: 1,
            external_call: 100,
        }
    }
}

/// Gas meter for tracking and limiting WASM execution
pub struct GasMeter {
    /// Gas limit
    limit: u64,
    /// Gas used so far
    used: AtomicU64,
    /// Gas costs
    costs: GasCosts,
    /// Start time for execution
    start_time: Instant,
    /// Timeout duration
    timeout: Duration,
}

impl GasMeter {
    /// Create a new gas meter with the given limit
    pub fn new(limit: u64, timeout_ms: u64) -> Self {
        Self {
            limit,
            used: AtomicU64::new(0),
            costs: GasCosts::default(),
            start_time: Instant::now(),
            timeout: Duration::from_millis(timeout_ms),
        }
    }
    
    /// Get the gas limit
    pub fn limit(&self) -> u64 {
        self.limit
    }
    
    /// Get the gas used so far
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }
    
    /// Get the remaining gas
    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.used())
    }
    
    /// Check if we have enough gas for an operation
    pub fn has_gas(&self, amount: u64) -> bool {
        self.remaining() >= amount
    }
    
    /// Check if execution has timed out
    pub fn has_timed_out(&self) -> bool {
        self.start_time.elapsed() > self.timeout
    }
    
    /// Consume gas for an operation
    pub fn consume(&self, amount: u64) -> Result<(), WasmError> {
        // Check for timeout first
        if self.has_timed_out() {
            return Err(WasmError::ExecutionTimeout);
        }
        
        // Get current gas used
        let current = self.used();
        
        // Calculate new gas used
        let new = current.saturating_add(amount);
        
        // Check if we have enough gas
        if new > self.limit {
            return Err(WasmError::OutOfGas);
        }
        
        // Update gas used
        self.used.store(new, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Consume gas for a memory allocation operation
    pub fn consume_memory(&self, bytes: u64) -> Result<(), WasmError> {
        let amount = self.costs.base + (bytes * self.costs.memory_byte);
        self.consume(amount)
    }
    
    /// Consume gas for a storage read operation
    pub fn consume_storage_read(&self, key_size: u64) -> Result<(), WasmError> {
        let amount = self.costs.base + (key_size * self.costs.storage_read_byte);
        self.consume(amount)
    }
    
    /// Consume gas for a storage write operation
    pub fn consume_storage_write(&self, key_size: u64, value_size: u64) -> Result<(), WasmError> {
        let amount = self.costs.base + 
            (key_size * self.costs.storage_read_byte) + 
            (value_size * self.costs.storage_write_byte);
        self.consume(amount)
    }
    
    /// Consume gas for a storage delete operation
    pub fn consume_storage_delete(&self, key_size: u64) -> Result<(), WasmError> {
        let amount = self.costs.base + (key_size * self.costs.storage_delete_byte);
        self.consume(amount)
    }
    
    /// Consume gas for a computation operation
    pub fn consume_compute(&self, complexity: u64) -> Result<(), WasmError> {
        let amount = self.costs.base + (complexity * self.costs.compute_byte);
        self.consume(amount)
    }
    
    /// Consume gas for an external function call
    pub fn consume_external_call(&self) -> Result<(), WasmError> {
        self.consume(self.costs.external_call)
    }
} 