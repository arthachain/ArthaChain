use std::time::{Duration, Instant};
use std::thread;

// Simulate blockchain metrics test
fn main() {
    println!("Blockchain Metrics Test");
    println!("======================");
    
    // Test configuration
    let node_counts = [1, 4, 8, 16, 32, 48];
    let test_duration_secs = 10;
    let transaction_batch_size = 1000;
    
    for &node_count in &node_counts {
        run_metrics_test(node_count, test_duration_secs, transaction_batch_size);
    }
}

fn run_metrics_test(node_count: usize, duration_secs: u64, batch_size: usize) {
    println!("\nRunning metrics test with {} nodes", node_count);
    println!("-----------------------------------");
    
    // Initialize counters
    let mut total_transactions = 0;
    let mut total_blocks = 0;
    let mut block_times = Vec::new();
    
    // Start test timer
    let start_time = Instant::now();
    
    // Simulate block creation (faster with more nodes)
    let block_interval_ms = 5000 / node_count.max(1);
    let transactions_per_block = batch_size * node_count;
    
    // Run for the specified duration
    while start_time.elapsed() < Duration::from_secs(duration_secs) {
        // Simulate creating a block
        let block_start = Instant::now();
        
        // Simulate processing transactions
        thread::sleep(Duration::from_millis(block_interval_ms as u64));
        
        // Record block metrics
        let block_time = block_start.elapsed();
        block_times.push(block_time);
        total_blocks += 1;
        total_transactions += transactions_per_block;
        
        // Print progress
        print!(".");
    }
    
    // Calculate metrics
    let test_duration = start_time.elapsed();
    let tps = total_transactions as f64 / test_duration.as_secs_f64();
    let bps = total_blocks as f64 / test_duration.as_secs_f64();
    
    // Calculate average block time
    let avg_block_time = if !block_times.is_empty() {
        block_times.iter().sum::<Duration>().as_secs_f64() / block_times.len() as f64
    } else {
        0.0
    };
    
    // Calculate transactions per block
    let avg_tx_per_block = if total_blocks > 0 {
        total_transactions as f64 / total_blocks as f64
    } else {
        0.0
    };
    
    println!("\n\nResults for {} nodes:", node_count);
    println!("  Test duration: {:.2} seconds", test_duration.as_secs_f64());
    println!("  Total transactions: {}", total_transactions);
    println!("  Total blocks: {}", total_blocks);
    println!("  Transactions per second (TPS): {:.2}", tps);
    println!("  Blocks per second: {:.2}", bps);
    println!("  Average block time: {:.2} seconds", avg_block_time);
    println!("  Average transactions per block: {:.2}", avg_tx_per_block);
    
    // Calculate theoretical maximum
    let theoretical_tps = (batch_size * node_count) as f64 / (block_interval_ms as f64 / 1000.0);
    println!("  Theoretical maximum TPS: {:.2}", theoretical_tps);
    println!("  Efficiency: {:.2}%", (tps / theoretical_tps) * 100.0);
} 