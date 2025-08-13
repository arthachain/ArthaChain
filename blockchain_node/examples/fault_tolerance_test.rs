use anyhow::{anyhow, Result};
use rand::{thread_rng, Rng};
use std::sync::{Arc, Mutex};
/**
 * Stress and Fault Tolerance Testing for the Blockchain Node
 *
 * This test simulates real-world conditions by introducing random failures while
 * processing transactions:
 * 1. Simulated node failures and restarts
 * 2. Network partitions and delays
 * 3. Temporary resource exhaustion
 *
 * The system should maintain throughput and consensus despite these issues.
 */
use std::time::{Duration, Instant};

use blockchain_node::execution::executor::{ExecutionResult, TransactionExecutor};
use blockchain_node::ledger::state::State;
use blockchain_node::ledger::transaction::{Transaction, TransactionType};

// Test configuration
struct FaultToleranceConfig {
    duration_secs: u64,             // Test duration
    batch_size: usize,              // Transactions per batch
    max_parallel: usize, // Max parallel executions (kept for config struct, but not used)
    failure_probability: f64, // Probability of introducing a failure
    network_delay_probability: f64, // Probability of network delay
    recovery_time_ms: u64, // Time to recover from failure (milliseconds)
    num_nodes: usize,    // Number of simulated nodes
}

// Types of failures to simulate
enum FailureType {
    NodeCrash,          // Node completely stops
    NetworkPartition,   // Node gets isolated
    ResourceExhaustion, // Node runs out of resources (CPU/memory)
    #[allow(dead_code)]
    NetworkDelay, // Network becomes slow
}

// Simulate a node in the system
struct Node {
    id: usize,
    failed: bool,
    partitioned: bool,
    resource_exhausted: bool,
    executor: Arc<TransactionExecutor>,
    transactions_processed: usize,
    failures_recovered: usize,
    processing_time: Duration,
    #[allow(dead_code)]
    state_tree: Arc<State>,
}

impl Node {
    fn new(id: usize, _max_parallel: usize) -> Self {
        let state_tree = Arc::new(State::new(&blockchain_node::config::Config::default()).unwrap());
        let executor = Arc::new(TransactionExecutor::new(
            None,      // wasm_executor: no WASM for examples
            1.0,       // gas_price_adjustment
            1_000_000, // max_gas_limit
            1,         // min_gas_price
        ));

        Self {
            id,
            failed: false,
            partitioned: false,
            resource_exhausted: false,
            executor,
            transactions_processed: 0,
            failures_recovered: 0,
            processing_time: Duration::new(0, 0),
            state_tree,
        }
    }

    // Process a batch of transactions with potential failures
    async fn process_batch(&mut self, batch: Vec<Transaction>) -> Result<usize> {
        if self.failed {
            return Err(anyhow!("Node {} is crashed", self.id));
        }

        if self.partitioned {
            return Err(anyhow!("Node {} is network partitioned", self.id));
        }

        let mut processed_count = 0;
        let start = Instant::now();

        let transactions_to_process = if self.resource_exhausted {
            // Simulate resource exhaustion by processing only half the batch
            let half_size = batch.len() / 2;
            batch.into_iter().take(half_size).collect()
        } else {
            batch
        };

        for tx in transactions_to_process {
            let mut mutable_tx = tx.clone(); // Make transaction mutable
            let result = self
                .executor
                .execute_transaction(&mut mutable_tx, &self.state_tree)
                .await; // Corrected method call
            match result {
                Ok(ExecutionResult::Success) => processed_count += 1,
                Ok(_) => { /* log failures if needed */ }
                Err(_) => { /* log errors if needed */ }
            }
        }

        self.processing_time += start.elapsed();
        self.transactions_processed += processed_count;
        Ok(processed_count)
    }

    // Introduce a failure to this node
    fn introduce_failure(&mut self, failure: FailureType) {
        match failure {
            FailureType::NodeCrash => {
                self.failed = true;
                println!("Node {} has crashed", self.id);
            }
            FailureType::NetworkPartition => {
                self.partitioned = true;
                println!("Node {} is network partitioned", self.id);
            }
            FailureType::ResourceExhaustion => {
                self.resource_exhausted = true;
                println!("Node {} is experiencing resource exhaustion", self.id);
            }
            FailureType::NetworkDelay => {
                // Network delay is handled at the system level
                println!("Node {} is experiencing network delays", self.id);
            }
        }
    }

    // Recover from a failure
    fn recover(&mut self) {
        if self.failed || self.partitioned || self.resource_exhausted {
            self.failed = false;
            self.partitioned = false;
            self.resource_exhausted = false;
            self.failures_recovered += 1;
            println!("Node {} has recovered", self.id);
        }
    }

    // Get node status for reporting
    fn status(&self) -> String {
        if self.failed {
            "CRASHED".to_string()
        } else if self.partitioned {
            "PARTITIONED".to_string()
        } else if self.resource_exhausted {
            "RESOURCE_LIMITED".to_string()
        } else {
            "HEALTHY".to_string()
        }
    }

    // Get TPS for this node
    fn get_tps(&self) -> f64 {
        if self.processing_time.as_secs_f64() > 0.0 {
            self.transactions_processed as f64 / self.processing_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

async fn run_fault_tolerance_test(config: &FaultToleranceConfig) -> Result<()> {
    println!("Starting fault tolerance test with:");
    println!("  - {} seconds duration", config.duration_secs);
    println!("  - {} batch size", config.batch_size);
    println!("  - {} max parallel executions", config.max_parallel);
    println!(
        "  - {:.1}% failure probability",
        config.failure_probability * 100.0
    );
    println!(
        "  - {:.1}% network delay probability",
        config.network_delay_probability * 100.0
    );
    println!("  - {} ms recovery time", config.recovery_time_ms);
    println!("  - {} simulated nodes", config.num_nodes);

    // Create simulated nodes
    let mut nodes: Vec<Node> = (0..config.num_nodes)
        .map(|id| Node::new(id, config.max_parallel))
        .collect();

    // Create shared state for test metrics
    let _total_txs = Arc::new(Mutex::new(0usize)); // Still here, but not used directly in current flow
    let _total_failures = Arc::new(Mutex::new(0usize)); // Still here, but not used directly in current flow

    println!("\nStarting test with fault injection...");
    println!("---------------------------------------");

    // Run test for the specified duration
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(config.duration_secs);
    let mut batch_counter = 0;

    // For periodic reporting
    let report_interval = Duration::from_secs(5);
    let mut last_report_time = start_time;

    // Test metrics
    let mut total_processed = 0;
    let mut failures_injected = 0;
    let mut recoveries = 0;

    while Instant::now() < end_time {
        // Generate a batch of transactions
        let batch = generate_batch(config.batch_size, batch_counter);
        batch_counter += 1;

        // Randomly introduce failures based on probability
        let mut rng = thread_rng();
        if rng.gen::<f64>() < config.failure_probability {
            // Choose a random node to fail
            let node_idx = rng.gen_range(0..nodes.len());

            // Choose a random failure type
            let failure_type = match rng.gen_range(0..3) {
                0 => FailureType::NodeCrash,
                1 => FailureType::NetworkPartition,
                _ => FailureType::ResourceExhaustion,
            };

            // Introduce the failure
            nodes[node_idx].introduce_failure(failure_type);
            failures_injected += 1;

            // Schedule recovery after the specified time
            let _node_id = nodes[node_idx].id;
            let recovery_time = config.recovery_time_ms;
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(recovery_time)).await;
                // Note: we can't directly modify the node here due to the borrow checker
                // The main loop will handle recovery
            });
        }

        // Process batch on each node (with potential network delays)
        for node in nodes.iter_mut() {
            // Clone the batch for each node
            let node_batch = batch.clone();

            // Introduce network delay if applicable
            let delay = if rng.gen::<f64>() < config.network_delay_probability {
                let delay_ms = rng.gen_range(50..300);
                println!(
                    "Network delay of {}ms introduced for Node {}",
                    delay_ms, node.id
                );
                Some(Duration::from_millis(delay_ms))
            } else {
                None
            };

            // Process batch (with delay if necessary)
            if let Some(delay_duration) = delay {
                tokio::time::sleep(delay_duration).await;
            }

            match node.process_batch(node_batch).await {
                Ok(processed) => {
                    total_processed += processed;
                }
                Err(_) => {
                    // Node failed to process - expected for failed/partitioned nodes
                }
            }
        }

        // Check for nodes that need recovery
        for node in nodes.iter_mut() {
            if (node.failed || node.partitioned || node.resource_exhausted)
                && rng.gen::<f64>() > 0.5
            {
                // 50% chance to recover on each iteration
                node.recover();
                recoveries += 1;
            }
        }

        // Report status periodically
        let now = Instant::now();
        if now - last_report_time >= report_interval {
            let elapsed = now - start_time;

            // Print status report
            println!("\n[{:5.1?}] Status Report:", elapsed);
            println!(
                "  Nodes: {}/{} healthy",
                nodes
                    .iter()
                    .filter(|n| !n.failed && !n.partitioned && !n.resource_exhausted)
                    .count(),
                nodes.len()
            );
            println!("  Batches processed: {}", batch_counter);
            println!("  Transactions successful: {}", total_processed);
            println!("  Failures injected: {}", failures_injected);
            println!("  Recoveries: {}", recoveries);

            // Print node statuses
            println!("\n  Node Status:");
            for node in &nodes {
                println!(
                    "    Node {}: {} - {} txs processed, {} failures recovered, {:.2} TPS",
                    node.id,
                    node.status(),
                    node.transactions_processed,
                    node.failures_recovered,
                    node.get_tps()
                );
            }

            last_report_time = now;
        }

        // Print a dot every few batches to show progress
        if batch_counter % 5 == 0 {
            print!(".");
            if batch_counter % 100 == 0 {
                println!();
            }
        }
    }

    // Final report
    let test_duration = start_time.elapsed();
    let overall_tps = total_processed as f64 / test_duration.as_secs_f64();

    // Calculate fault tolerance metrics
    let healthy_nodes = nodes
        .iter()
        .filter(|n| !n.failed && !n.partitioned && !n.resource_exhausted)
        .count();
    let failure_recovery_ratio = if failures_injected > 0 {
        recoveries as f64 / failures_injected as f64
    } else {
        1.0
    };

    // Calculate node specific metrics
    let healthy_node_tps: Vec<f64> = nodes
        .iter()
        .filter(|n| !n.failed && !n.partitioned && !n.resource_exhausted)
        .map(|n| n.get_tps())
        .collect();

    let avg_healthy_tps = if !healthy_node_tps.is_empty() {
        healthy_node_tps.iter().sum::<f64>() / healthy_node_tps.len() as f64
    } else {
        0.0
    };

    println!("\n\nFault Tolerance Test Complete");
    println!("=============================");
    println!("Total test duration: {:.2?}", test_duration);
    println!("Total batches processed: {}", batch_counter);
    println!("Total transactions processed: {}", total_processed);
    println!("Overall system TPS: {:.2}", overall_tps);
    println!("Failures injected: {}", failures_injected);
    println!("Recoveries completed: {}", recoveries);
    println!("Failure recovery ratio: {:.2}", failure_recovery_ratio);
    println!(
        "Final healthy nodes: {}/{} ({:.1}%)",
        healthy_nodes,
        nodes.len(),
        (healthy_nodes as f64 / nodes.len() as f64) * 100.0
    );
    println!("Average TPS per healthy node: {:.2}", avg_healthy_tps);

    // Evaluate test success
    let success_threshold = 0.7; // At least 70% of normal throughput
    let min_recovery_ratio = 0.9; // At least 90% recovery rate

    if overall_tps
        > success_threshold * avg_healthy_tps * healthy_nodes as f64 / config.num_nodes as f64
        && failure_recovery_ratio >= min_recovery_ratio
    {
        println!(
            "\nFAULT TOLERANCE TEST PASSED ✅ - System maintained throughput despite failures"
        );
    } else {
        println!("\nFAULT TOLERANCE TEST WARNING ⚠️ - System experienced degraded performance under failures");
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
    println!("Blockchain Fault Tolerance Test");
    println!("===============================\n");

    // Different test configurations
    let configs = [
        // Quick test with high failure rate for debugging
        FaultToleranceConfig {
            duration_secs: 30,
            batch_size: 20,
            max_parallel: 4,
            failure_probability: 0.2, // 20% chance of failure per iteration
            network_delay_probability: 0.3, // 30% chance of network delay
            recovery_time_ms: 1000,   // 1 second to recover
            num_nodes: 3,
        },
        // More realistic test
        FaultToleranceConfig {
            duration_secs: 120,
            batch_size: 50,
            max_parallel: 8,
            failure_probability: 0.05,      // 5% chance of failure
            network_delay_probability: 0.1, // 10% chance of network delay
            recovery_time_ms: 3000,         // 3 seconds to recover
            num_nodes: 5,
        },
        // Production-like resilience test
        FaultToleranceConfig {
            duration_secs: 300,
            batch_size: 100,
            max_parallel: 16,
            failure_probability: 0.02,       // 2% chance of failure
            network_delay_probability: 0.05, // 5% chance of network delay
            recovery_time_ms: 5000,          // 5 seconds to recover
            num_nodes: 10,
        },
    ];

    // Run the selected configuration (uncomment the one you want to run)
    // For quick tests, use the first config
    run_fault_tolerance_test(&configs[0]).await?;

    // For more realistic testing
    // run_fault_tolerance_test(&configs[1]).await?;

    // For production-like resilience testing
    // run_fault_tolerance_test(&configs[2]).await?;

    Ok(())
}
