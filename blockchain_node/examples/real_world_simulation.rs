use anyhow::Result;
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::collections::HashMap;
use std::sync::Arc;
/**
 * Real-World Transaction Simulation for Blockchain Node
 *
 * This benchmark simulates real-world usage patterns including:
 * 1. Mixed transaction types (payments, smart contracts, data storage)
 * 2. Varying transaction sizes and complexity
 * 3. Realistic distribution of transaction priorities and gas prices
 * 4. Smart contract deployment and interaction
 *
 * The goal is to measure performance in conditions similar to production.
 */
use std::time::{Duration, Instant};

use blockchain_node::execution::executor::TransactionExecutor;
use blockchain_node::execution::parallel::{
    ConflictStrategy, ParallelConfig, ParallelExecutionManager,
};
use blockchain_node::ledger::state::storage::StateStorage;
use blockchain_node::ledger::state::StateTree;
use blockchain_node::ledger::transaction::TransactionType;
use blockchain_node::transaction::Transaction;

// Transaction mix configuration
#[derive(Clone)]
struct TransactionMixConfig {
    // Percentage of each transaction type (must sum to 100)
    payment_percent: u8,
    contract_deploy_percent: u8,
    contract_call_percent: u8,
    data_storage_percent: u8,

    // Size configurations
    payment_size_bytes: usize,
    contract_deploy_min_bytes: usize,
    contract_deploy_max_bytes: usize,
    contract_call_min_bytes: usize,
    contract_call_max_bytes: usize,
    data_storage_min_bytes: usize,
    data_storage_max_bytes: usize,

    // Gas price distribution
    min_gas_price: u64,
    max_gas_price: u64,
}

// Default real-world transaction mix
impl Default for TransactionMixConfig {
    fn default() -> Self {
        Self {
            // Transaction type distribution
            payment_percent: 60,
            contract_deploy_percent: 5,
            contract_call_percent: 30,
            data_storage_percent: 5,

            // Size configurations
            payment_size_bytes: 100,
            contract_deploy_min_bytes: 1000,
            contract_deploy_max_bytes: 10000,
            contract_call_min_bytes: 200,
            contract_call_max_bytes: 1000,
            data_storage_min_bytes: 1000,
            data_storage_max_bytes: 50000,

            // Gas price distribution
            min_gas_price: 1,
            max_gas_price: 100,
        }
    }
}

// Real-world simulation configuration
struct SimulationConfig {
    duration_secs: u64,
    batch_size: usize,
    max_parallel: usize,
    num_accounts: usize,
    num_contracts: usize,
    tx_mix: TransactionMixConfig,
}

// Transaction generator for different transaction types
struct TransactionGenerator {
    tx_mix: TransactionMixConfig,
    accounts: Vec<String>,
    contracts: Vec<String>,
    next_id: usize,
}

impl TransactionGenerator {
    fn new(tx_mix: TransactionMixConfig, num_accounts: usize, num_contracts: usize) -> Self {
        // Generate account addresses
        let accounts = (0..num_accounts).map(|i| format!("account{}", i)).collect();

        // Generate contract addresses
        let contracts = (0..num_contracts)
            .map(|i| format!("contract{}", i))
            .collect();

        Self {
            tx_mix,
            accounts,
            contracts,
            next_id: 0,
        }
    }

    // Generate a batch of mixed transactions
    fn generate_batch(&mut self, batch_size: usize) -> Vec<Transaction> {
        let mut transactions = Vec::with_capacity(batch_size);
        let mut rng = thread_rng();

        for _ in 0..batch_size {
            // Determine transaction type based on configured percentages
            let tx_type_roll = rng.gen_range(0..100);
            let tx = if tx_type_roll < self.tx_mix.payment_percent {
                self.generate_payment()
            } else if tx_type_roll
                < self.tx_mix.payment_percent + self.tx_mix.contract_deploy_percent
            {
                self.generate_contract_deploy()
            } else if tx_type_roll
                < self.tx_mix.payment_percent
                    + self.tx_mix.contract_deploy_percent
                    + self.tx_mix.contract_call_percent
            {
                self.generate_contract_call()
            } else {
                self.generate_data_storage()
            };

            transactions.push(tx);
            self.next_id += 1;
        }

        transactions
    }

    // Generate a simple payment transaction
    fn generate_payment(&self) -> Transaction {
        let mut rng = thread_rng();

        // Choose random sender and recipient
        let sender = self.accounts.choose(&mut rng).unwrap().clone();
        let recipient = self.accounts.choose(&mut rng).unwrap().clone();

        // Generate random amount
        let amount = rng.gen_range(1..1000);

        // Generate fixed size data for payment
        let mut data = vec![0u8; self.tx_mix.payment_size_bytes];
        rng.fill(&mut data[..]);

        // Generate gas price within range
        let gas_price = rng.gen_range(self.tx_mix.min_gas_price..self.tx_mix.max_gas_price);

        Transaction {
            tx_type: TransactionType::Transfer,
            sender,
            recipient,
            amount,
            nonce: self.next_id as u64,
            gas_price,
            gas_limit: 21000, // Standard gas limit for transfers
            data,
            signature: Vec::new(), // Empty for test
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            #[cfg(feature = "bls")]
            bls_signature: None,
            status: blockchain_node::ledger::transaction::TransactionStatus::Pending,
        }
    }

    // Generate a contract deployment transaction
    fn generate_contract_deploy(&self) -> Transaction {
        let mut rng = thread_rng();

        // Choose random sender
        let sender = self.accounts.choose(&mut rng).unwrap().clone();

        // Generate contract bytecode of random size
        let size = rng.gen_range(
            self.tx_mix.contract_deploy_min_bytes..self.tx_mix.contract_deploy_max_bytes,
        );
        let mut data = vec![0u8; size];
        rng.fill(&mut data[..]);

        // Add magic bytes to indicate this is contract deployment data
        data[0..4].copy_from_slice(&[0xd3, 0x60, 0x45, 0x93]); // Magic bytes for contract deploy

        // Generate gas price with bias towards higher values for deployments
        let gas_price = rng.gen_range(self.tx_mix.min_gas_price..self.tx_mix.max_gas_price) * 2;

        Transaction {
            tx_type: TransactionType::Deploy,
            sender,
            recipient: "0x0000000000000000000000000000000000000000".to_string(), // Zero address for contract creation
            amount: 0,
            nonce: self.next_id as u64,
            gas_price,
            gas_limit: 1000000, // Higher gas limit for deployments
            data,
            signature: Vec::new(), // Empty for test
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            #[cfg(feature = "bls")]
            bls_signature: None,
            status: blockchain_node::ledger::transaction::TransactionStatus::Pending,
        }
    }

    // Generate a contract call transaction
    fn generate_contract_call(&self) -> Transaction {
        let mut rng = thread_rng();

        // Choose random sender and contract
        let sender = self.accounts.choose(&mut rng).unwrap().clone();
        let contract = self.contracts.choose(&mut rng).unwrap().clone();

        // Generate function call data of random size
        let size =
            rng.gen_range(self.tx_mix.contract_call_min_bytes..self.tx_mix.contract_call_max_bytes);
        let mut data = vec![0u8; size];
        rng.fill(&mut data[..]);

        // Add function selector (first 4 bytes)
        match rng.gen_range(0..4) {
            0 => data[0..4].copy_from_slice(&[0xa9, 0x05, 0x9c, 0xbb]), // transfer(address,uint256)
            1 => data[0..4].copy_from_slice(&[0x09, 0x5e, 0xa7, 0xb3]), // approve(address,uint256)
            2 => data[0..4].copy_from_slice(&[0x70, 0xa0, 0x82, 0x31]), // balanceOf(address)
            _ => data[0..4].copy_from_slice(&[0x18, 0x16, 0x0d, 0xdd]), // totalSupply()
        }

        // Generate gas price
        let gas_price = rng.gen_range(self.tx_mix.min_gas_price..self.tx_mix.max_gas_price);

        Transaction {
            tx_type: TransactionType::Call,
            sender,
            recipient: contract,
            amount: rng.gen_range(0..100),
            nonce: self.next_id as u64,
            gas_price,
            gas_limit: 100000, // Medium gas limit for calls
            data,
            signature: Vec::new(), // Empty for test
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            #[cfg(feature = "bls")]
            bls_signature: None,
            status: blockchain_node::ledger::transaction::TransactionStatus::Pending,
        }
    }

    // Generate a data storage transaction
    fn generate_data_storage(&self) -> Transaction {
        let mut rng = thread_rng();

        // Choose random sender
        let sender = self.accounts.choose(&mut rng).unwrap().clone();

        // Generate large data of random size
        let size =
            rng.gen_range(self.tx_mix.data_storage_min_bytes..self.tx_mix.data_storage_max_bytes);
        let mut data = vec![0u8; size];
        rng.fill(&mut data[..]);

        // Generate gas price with bias towards lower values for storage (cost sensitive)
        let gas_price = rng.gen_range(self.tx_mix.min_gas_price..self.tx_mix.max_gas_price) / 2;

        Transaction {
            tx_type: TransactionType::System, // Using System for data storage
            sender,
            recipient: "0x0000000000000000000000000000000000000001".to_string(), // Data storage address
            amount: 0,
            nonce: self.next_id as u64,
            gas_price,
            gas_limit: 500000, // Higher gas limit for data storage
            data,
            signature: Vec::new(), // Empty for test
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            #[cfg(feature = "bls")]
            bls_signature: None,
            status: blockchain_node::ledger::transaction::TransactionStatus::Pending,
        }
    }
}

// Run the mixed-transaction simulation
async fn run_real_world_simulation(config: &SimulationConfig) -> Result<()> {
    println!("Starting real-world transaction simulation with:");
    println!("  - {} seconds duration", config.duration_secs);
    println!("  - {} batch size", config.batch_size);
    println!("  - {} max parallel executions", config.max_parallel);
    println!("  - {} accounts", config.num_accounts);
    println!("  - {} contracts", config.num_contracts);
    println!("  - Transaction mix: {}% payments, {}% contract deploys, {}% contract calls, {}% data storage",
             config.tx_mix.payment_percent,
             config.tx_mix.contract_deploy_percent,
             config.tx_mix.contract_call_percent,
             config.tx_mix.data_storage_percent);

    // Initialize state
    let _storage = Arc::new(StateStorage::new());
    let state_tree = Arc::new(StateTree::new());

    // Create transaction generator
    let mut tx_generator = TransactionGenerator::new(
        config.tx_mix.clone(),
        config.num_accounts,
        config.num_contracts,
    );

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
    let mut batch_counter = 0;

    // Transaction type statistics
    let mut tx_type_counts = HashMap::new();
    tx_type_counts.insert(TransactionType::Transfer, 0);
    tx_type_counts.insert(TransactionType::Deploy, 0);
    tx_type_counts.insert(TransactionType::Call, 0);
    tx_type_counts.insert(TransactionType::System, 0); // For data storage

    // Timing statistics by transaction type
    let mut tx_type_times = HashMap::new();
    tx_type_times.insert(TransactionType::Transfer, Duration::new(0, 0));
    tx_type_times.insert(TransactionType::Deploy, Duration::new(0, 0));
    tx_type_times.insert(TransactionType::Call, Duration::new(0, 0));
    tx_type_times.insert(TransactionType::System, Duration::new(0, 0)); // For data storage

    // Batch statistics
    let mut total_txs = 0;
    let mut total_batch_time = Duration::new(0, 0);
    let mut total_size_bytes = 0;

    println!("\nRunning simulation...");
    println!("---------------------------------------");

    // Report interval (5 seconds)
    let report_interval = Duration::from_secs(5);
    let mut last_report_time = start_time;
    let mut txs_since_report = 0;

    while Instant::now() < end_time {
        // Generate a batch of mixed transactions
        let transactions = tx_generator.generate_batch(config.batch_size);
        let batch_size = transactions.len();
        batch_counter += 1;

        // Count transaction types and total data size
        for tx in &transactions {
            *tx_type_counts.entry(tx.tx_type).or_insert(0) += 1;
            total_size_bytes += tx.data.len();
        }

        // Process the batch
        let batch_start = Instant::now();
        let _results = execution_manager.process_transactions(transactions).await?;
        let batch_duration = batch_start.elapsed();

        // Update metrics
        total_txs += batch_size;
        total_batch_time += batch_duration;
        txs_since_report += batch_size;

        // Report periodically
        let now = Instant::now();
        if now - last_report_time >= report_interval {
            let elapsed = now - start_time;
            let report_duration = now - last_report_time;
            let report_tps = txs_since_report as f64 / report_duration.as_secs_f64();
            let overall_tps = total_txs as f64 / elapsed.as_secs_f64();

            println!("[{:5.1?}] Processed {:5} txs in last {:3.1?} | Current: {:8.0} TPS | Overall: {:8.0} TPS | Batches: {}",
                     elapsed,
                     txs_since_report,
                     report_duration,
                     report_tps,
                     overall_tps,
                     batch_counter);

            // Reset report metrics
            last_report_time = now;
            txs_since_report = 0;
        }

        // Show progress
        if batch_counter % 10 == 0 {
            print!(".");
            if batch_counter % 200 == 0 {
                println!();
            }
        }
    }

    // Calculate overall statistics
    let test_duration = start_time.elapsed();
    let overall_tps = total_txs as f64 / test_duration.as_secs_f64();
    let avg_batch_time = if batch_counter > 0 {
        total_batch_time.div_f64(batch_counter as f64)
    } else {
        Duration::new(0, 0)
    };

    // Calculate TPS by transaction type
    let mut tx_type_tps = HashMap::new();
    for (tx_type, count) in &tx_type_counts {
        let tx_tps = *count as f64 / test_duration.as_secs_f64();
        tx_type_tps.insert(*tx_type, tx_tps);
    }

    println!("\n\nReal-World Simulation Complete");
    println!("==============================");
    println!("Total test duration: {:.2?}", test_duration);
    println!("Total transactions processed: {}", total_txs);
    println!("Total batches processed: {}", batch_counter);
    println!("Overall TPS: {:.2}", overall_tps);
    println!(
        "Total data size processed: {:.2} MB",
        total_size_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("Average batch processing time: {:.2?}", avg_batch_time);

    // Print transaction type statistics
    println!("\nTransaction Type Statistics:");
    println!("----------------------------");
    for tx_type in &[
        TransactionType::Transfer,
        TransactionType::Deploy,
        TransactionType::Call,
        TransactionType::System,
    ] {
        let count = tx_type_counts.get(tx_type).unwrap_or(&0);
        let percentage = if total_txs > 0 {
            (*count as f64 / total_txs as f64) * 100.0
        } else {
            0.0
        };
        let tps = tx_type_tps.get(tx_type).unwrap_or(&0.0);

        println!(
            "{:?}: {} transactions ({:.1}%) at {:.2} TPS",
            tx_type, count, percentage, tps
        );
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Real-World Transaction Simulation");
    println!("=================================\n");

    // Define transaction mix configurations
    let default_mix = TransactionMixConfig::default();

    // Contract-heavy mix
    let contract_mix = TransactionMixConfig {
        payment_percent: 30,
        contract_deploy_percent: 10,
        contract_call_percent: 55,
        data_storage_percent: 5,
        ..TransactionMixConfig::default()
    };

    // Data-heavy mix
    let data_mix = TransactionMixConfig {
        payment_percent: 40,
        contract_deploy_percent: 5,
        contract_call_percent: 25,
        data_storage_percent: 30,
        ..TransactionMixConfig::default()
    };

    // Define simulation configurations
    let configs = vec![
        // Standard realistic mix (short duration)
        SimulationConfig {
            duration_secs: 30,
            batch_size: 20,
            max_parallel: 8,
            num_accounts: 100,
            num_contracts: 10,
            tx_mix: default_mix,
        },
        // Contract-heavy mix (medium duration)
        SimulationConfig {
            duration_secs: 60,
            batch_size: 50,
            max_parallel: 16,
            num_accounts: 200,
            num_contracts: 50,
            tx_mix: contract_mix,
        },
        // Data-heavy mix (long duration)
        SimulationConfig {
            duration_secs: 120,
            batch_size: 100,
            max_parallel: 32,
            num_accounts: 500,
            num_contracts: 100,
            tx_mix: data_mix,
        },
    ];

    // Run the selected configuration (uncomment the one you want to run)
    run_real_world_simulation(&configs[0]).await?;

    // For more complex tests
    // run_real_world_simulation(&configs[1]).await?;
    // run_real_world_simulation(&configs[2]).await?;

    Ok(())
}
