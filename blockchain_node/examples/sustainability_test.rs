use anyhow::Result;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use blockchain_node::execution::executor::{ExecutionResult, TransactionExecutor};
use blockchain_node::ledger::state::State;
use blockchain_node::ledger::transaction::{Transaction, TransactionType};

// Test configuration
struct SustainabilityConfig {
    duration_secs: u64,        // How long to run the test
    batch_size: usize,         // Transactions per batch
    max_parallel: usize, // Max parallel transaction executions (kept for config struct, but not used)
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
    let state_tree = Arc::new(State::new(&blockchain_node::config::Config::default()).unwrap());

    // Create transaction executor
    let executor = Arc::new(TransactionExecutor::new(
        None,      // wasm_executor: no WASM for examples
        1.0,       // gas_price_adjustment
        1_000_000, // max_gas_limit
        1,         // min_gas_price
    ));

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
        let mut _batch_successes = 0;
        let mut _batch_failures = 0;

        for tx in transactions {
            let mut mutable_tx = tx.clone(); // Clone and make mutable
            let result = executor
                .execute_transaction(&mut mutable_tx, &state_tree)
                .await;
            match result {
                Ok(ExecutionResult::Success) => _batch_successes += 1,
                Ok(_) => _batch_failures += 1,
                Err(_) => _batch_failures += 1,
            }
        }
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

    let sender_prefix = "sender_";
    let recipient_prefix = "recipient_";

    for i in 0..count {
        let tx_index = batch_id * count + i;
        let sender = format!("{}{}", sender_prefix, tx_index);
        let recipient = format!("{}{}", recipient_prefix, tx_index + 1);

        // Create a transaction with a unique ID based on batch and index
        let tx = Transaction::new(
            TransactionType::Transfer,
            sender,
            recipient,
            100 + tx_index as u64,
            tx_index as u64,
            10,
            21000,
            vec![],
        );
        transactions.push(tx);
    }

    transactions
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Blockchain Sustainability Test");
    println!("================================\n");

    // Different test durations for different needs
    let configs = [
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
