use anyhow::Result;
use std::sync::Arc;
use std::thread;
/**
 * Sustainability Testing for the Blockchain Node
 *
 * This test runs transactions continuously for a set duration (5-10 minutes) to:
 * 1. Detect performance degradation over time
 * 2. Monitor memory usage for potential leaks
 * 3. Ensure the system maintains consistent throughput
 *
 * It reports periodic statistics and final summary metrics to evaluate
 * the long-term stability of the system.
 */
use std::time::{Duration, Instant};

use blockchain_node::execution::executor::TransactionExecutor;
use blockchain_node::execution::parallel::{
    ConflictStrategy, ParallelConfig, ParallelExecutionManager,
};
use blockchain_node::ledger::state::storage::StateStorage;
use blockchain_node::ledger::state::StateTree;
use blockchain_node::transaction::Transaction;

// Test configuration
struct SustainabilityConfig {
    duration_secs: u64,        // How long to run the test
    batch_size: usize,         // Transactions per batch
    max_parallel: usize,       // Max parallel transaction executions
    report_interval_secs: u64, // How often to report stats
}

async fn run_sustainability_test(config: &SustainabilityConfig) -> Result<()> {
    println!("Starting sustainability test with:");
    println!("  - {} seconds duration", config.duration_secs);
    println!("  - {} batch size", config.batch_size);
    println!("  - {} max parallel executions", config.max_parallel);
    println!(
        "  - {} seconds reporting interval",
        config.report_interval_secs
    );

    // Initialize state
    let _storage = Arc::new(StateStorage::new());
    let state_tree = Arc::new(StateTree::new());

    // Create parallel execution manager
    let executor = Arc::new(TransactionExecutor::new());
    let parallel_config = ParallelConfig {
        max_parallel: config.max_parallel,
        max_group_size: 10,
        conflict_strategy: ConflictStrategy::Retry,
        execution_timeout: 5000,
        retry_attempts: 3,
        enable_work_stealing: true,
        enable_simd: true,
        worker_threads: 0, // Auto
        simd_batch_size: 32,
        memory_pool_size: 1024 * 1024 * 256, // 256MB pre-allocated memory
    };
    let mut execution_manager =
        ParallelExecutionManager::new(parallel_config, state_tree.clone(), executor.clone());

    // Test metrics
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(config.duration_secs);
    let mut total_transactions = 0;
    let mut total_processing_time = Duration::new(0, 0);
    let mut batch_counter = 0;

    // For periodic reporting
    let mut last_report_time = start_time;
    let mut transactions_since_last_report = 0;

    // For performance degradation tracking
    let mut first_minute_tps = 0.0;
    let first_minute_end = start_time + Duration::from_secs(60);
    let mut first_minute_txs = 0;
    let mut first_minute_time = Duration::new(0, 0);

    println!("\nStarting continuous transaction processing...");
    println!("---------------------------------------------------");

    // Run until the test duration is reached
    while Instant::now() < end_time {
        // Generate a new batch of transactions
        let batch_id = batch_counter;
        let transactions = generate_batch(config.batch_size, batch_id);
        let batch_len = transactions.len();
        batch_counter += 1;

        // Process the batch
        let batch_start = Instant::now();
        let results = execution_manager.process_transactions(transactions).await?;
        let batch_duration = batch_start.elapsed();

        // Update metrics
        total_transactions += batch_len;
        total_processing_time += batch_duration;
        transactions_since_last_report += batch_len;

        // Track first minute performance as baseline
        if Instant::now() < first_minute_end {
            first_minute_txs += batch_len;
            first_minute_time += batch_duration;

            if Instant::now() >= first_minute_end || batch_counter % 10 == 0 {
                first_minute_tps = first_minute_txs as f64 / first_minute_time.as_secs_f64();
            }
        }

        // Calculate batch TPS
        let batch_tps = batch_len as f64 / batch_duration.as_secs_f64();

        // Count successes and failures
        let _successes = results.values().filter(|r| r.is_ok()).count();
        let _failures = results.values().filter(|r| r.is_err()).count();

        // Report periodically
        let now = Instant::now();
        if now - last_report_time >= Duration::from_secs(config.report_interval_secs) {
            let elapsed = now - start_time;
            let interval = now - last_report_time;
            let interval_tps = transactions_since_last_report as f64 / interval.as_secs_f64();
            let overall_tps = total_transactions as f64 / elapsed.as_secs_f64();

            // Calculate performance degradation percentage
            let degradation_pct = if first_minute_tps > 0.0 {
                let decrease = first_minute_tps - interval_tps;
                if decrease > 0.0 {
                    (decrease / first_minute_tps) * 100.0
                } else {
                    0.0 // No degradation
                }
            } else {
                0.0
            };

            println!(
                "[{:5.1?}] Processed {:5} txs | Interval: {:8.0} TPS | Overall: {:8.0} TPS | Batch: {:8.0} TPS | Degradation: {:4.1}%",
                elapsed,
                transactions_since_last_report,
                interval_tps,
                overall_tps,
                batch_tps,
                degradation_pct
            );

            // Check for severe degradation
            if degradation_pct > 20.0 {
                println!(
                    "WARNING: Performance degradation detected! TPS decreased by {:.1}%",
                    degradation_pct
                );
            }

            // Reset interval metrics
            last_report_time = now;
            transactions_since_last_report = 0;
        }

        // Print a dot every few batches to show progress
        if batch_counter % 10 == 0 {
            print!(".");
            if batch_counter % 500 == 0 {
                println!();
            }
        }

        // Optional: Simulate varying load by adding small delays between batches
        if batch_counter % 50 == 0 {
            thread::sleep(Duration::from_millis(50));
        }
    }

    // Final report
    let total_duration = start_time.elapsed();
    let overall_tps = total_transactions as f64 / total_duration.as_secs_f64();
    let processing_tps = total_transactions as f64 / total_processing_time.as_secs_f64();

    println!("\n\nSustainability Test Complete");
    println!("============================");
    println!("Total test duration: {:.2?}", total_duration);
    println!("Total transactions processed: {}", total_transactions);
    println!("Total batches processed: {}", batch_counter);
    println!("Overall TPS (including delays): {:.2}", overall_tps);
    println!("Processing TPS (excluding delays): {:.2}", processing_tps);
    println!("First minute baseline TPS: {:.2}", first_minute_tps);

    // Calculate final degradation
    let final_minute_txs = transactions_since_last_report;
    let final_minute_duration = Instant::now() - last_report_time;
    let final_minute_tps = final_minute_txs as f64 / final_minute_duration.as_secs_f64();

    let degradation = if first_minute_tps > 0.0 {
        let decrease = first_minute_tps - final_minute_tps;
        if decrease > 0.0 {
            (decrease / first_minute_tps) * 100.0
        } else {
            0.0 // No degradation or improved performance
        }
    } else {
        0.0
    };

    println!("Final minute TPS: {:.2}", final_minute_tps);
    println!("Performance degradation: {:.2}%", degradation);

    if degradation < 5.0 {
        println!("\nSUSTAINABILITY TEST PASSED ✅ - Minimal performance degradation detected");
    } else if degradation < 15.0 {
        println!("\nSUSTAINABILITY TEST WARNING ⚠️ - Moderate performance degradation detected");
    } else {
        println!("\nSUSTAINABILITY TEST FAILED ❌ - Significant performance degradation detected!");
    }

    Ok(())
}

fn generate_batch(count: usize, batch_id: usize) -> Vec<Transaction> {
    let mut transactions = Vec::with_capacity(count);

    for i in 0..count {
        // Create a transaction with a unique ID based on batch and index
        let tx_id = batch_id * count + i;
        let tx = Transaction::new_test_transaction(tx_id);
        transactions.push(tx);
    }

    transactions
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Blockchain Sustainability Test");
    println!("================================\n");

    // Different test durations for different needs
    let configs = vec![
        // Short test (30 seconds) for quick verification
        SustainabilityConfig {
            duration_secs: 30,
            batch_size: 50,
            max_parallel: 8,
            report_interval_secs: 5,
        },
        // Medium test (2 minutes)
        SustainabilityConfig {
            duration_secs: 120,
            batch_size: 100,
            max_parallel: 16,
            report_interval_secs: 10,
        },
        // Long test (5 minutes) for thorough sustainability verification
        SustainabilityConfig {
            duration_secs: 300,
            batch_size: 200,
            max_parallel: 32,
            report_interval_secs: 15,
        },
    ];

    // Run the selected configuration (uncomment the one you want to run)
    // For quick tests, use the first config (30 seconds)
    run_sustainability_test(&configs[0]).await?;

    // For more thorough testing, use the medium or long configs
    // run_sustainability_test(&configs[1]).await?;
    // run_sustainability_test(&configs[2]).await?;

    Ok(())
}
