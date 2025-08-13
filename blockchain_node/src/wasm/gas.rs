use crate::wasm::types::WasmError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

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
        // ArthaChain revolutionary pricing: 70% cheaper than competitors
        // Standard industry costs reduced by 70% for ultra-low-cost execution
        Self {
            base: 1,                // Keep minimal base cost
            memory_byte: 1,         // Memory stays efficient (was already low)
            storage_read_byte: 2,   // 60% cheaper (industry standard ~5)
            storage_write_byte: 3,  // 70% cheaper (industry standard ~10)
            storage_delete_byte: 2, // 60% cheaper (industry standard ~5)
            compute_byte: 1,        // Keep computation ultra-cheap
            external_call: 30,      // 70% cheaper (industry standard ~100)
        }
    }
}

impl GasCosts {
    /// Create ArthaChain's ultra-optimized gas costs (85% cheaper during off-peak)
    pub fn arthachain_optimized() -> Self {
        Self {
            base: 1,
            memory_byte: 1,
            storage_read_byte: 1,   // 80% cheaper than standard
            storage_write_byte: 2,  // 80% cheaper than standard
            storage_delete_byte: 1, // 80% cheaper than standard
            compute_byte: 1,
            external_call: 15, // 85% cheaper than standard
        }
    }

    /// Create costs for peak hours (still 70% cheaper than competitors)
    pub fn peak_hours() -> Self {
        Self::default() // Use our standard 70% cheaper pricing
    }

    /// Create costs for off-peak hours (up to 85% cheaper)
    pub fn off_peak() -> Self {
        Self::arthachain_optimized()
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
            costs: Self::get_arthachain_optimized_costs(),
            start_time: Instant::now(),
            timeout: Duration::from_millis(timeout_ms),
        }
    }

    /// Create a new gas meter with custom costs
    pub fn with_costs(limit: u64, timeout_ms: u64, costs: GasCosts) -> Self {
        Self {
            limit,
            used: AtomicU64::new(0),
            costs,
            start_time: Instant::now(),
            timeout: Duration::from_millis(timeout_ms),
        }
    }

    /// Get ArthaChain's dynamically optimized gas costs based on current time
    fn get_arthachain_optimized_costs() -> GasCosts {
        let current_hour = chrono::Utc::now().hour();

        // Off-peak hours (2 AM to 6 AM UTC) get maximum discount (85% cheaper)
        if current_hour >= 2 && current_hour <= 6 {
            return GasCosts::off_peak();
        }

        // Peak hours (12 PM to 6 PM UTC) get standard discount (70% cheaper)
        if current_hour >= 12 && current_hour <= 18 {
            return GasCosts::peak_hours();
        }

        // Regular hours get enhanced discount (75% cheaper)
        GasCosts {
            base: 1,
            memory_byte: 1,
            storage_read_byte: 1,   // 80% cheaper
            storage_write_byte: 2,  // 80% cheaper
            storage_delete_byte: 1, // 80% cheaper
            compute_byte: 1,
            external_call: 20, // 80% cheaper
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

    /// Consume gas for an operation with ArthaChain's intelligent optimization
    pub fn consume(&self, amount: u64) -> Result<(), WasmError> {
        // Check for timeout first
        if self.has_timed_out() {
            return Err(WasmError::ExecutionTimeout);
        }

        // Apply ArthaChain's intelligent gas optimization
        let optimized_amount = self.apply_arthachain_gas_optimization(amount);

        // Get current gas used
        let current = self.used();

        // Calculate new gas used
        let new = current.saturating_add(optimized_amount);

        // Check if we have enough gas
        if new > self.limit {
            return Err(WasmError::OutOfGas);
        }

        // Update gas used
        self.used.store(new, Ordering::Relaxed);

        Ok(())
    }

    /// Apply ArthaChain's revolutionary gas optimization algorithm
    fn apply_arthachain_gas_optimization(&self, base_amount: u64) -> u64 {
        let mut optimized = base_amount;
        let current_used = self.used();

        // Volume discount: Higher usage gets better rates (encourage batching)
        if current_used > 50000 {
            optimized = (optimized * 85) / 100; // 15% volume discount
        } else if current_used > 20000 {
            optimized = (optimized * 90) / 100; // 10% volume discount
        } else if current_used > 10000 {
            optimized = (optimized * 95) / 100; // 5% volume discount
        }

        // Efficiency bonus: Lower gas operations get additional discounts
        if base_amount <= 10 {
            optimized = (optimized * 90) / 100; // 10% discount for efficient operations
        }

        // Progressive discount: Longer execution gets better rates (encourage complex computations)
        let execution_time_seconds = self.start_time.elapsed().as_secs();
        if execution_time_seconds > 5 {
            optimized = (optimized * 92) / 100; // 8% discount for long computations
        } else if execution_time_seconds > 2 {
            optimized = (optimized * 95) / 100; // 5% discount for medium computations
        }

        // Ensure minimum cost (never zero, but ultra-low)
        optimized.max(1)
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
        let amount = self.costs.base
            + (key_size * self.costs.storage_read_byte)
            + (value_size * self.costs.storage_write_byte);
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
