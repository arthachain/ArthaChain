use anyhow::Result;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashSet;
use std::sync::Arc;
/**
 * Network Layer Benchmark for Blockchain Node
 *
 * This benchmark tests the distributed performance of the blockchain network:
 * 1. Simulates a multi-node network with configurable topology
 * 2. Measures transaction propagation and consensus times
 * 3. Evaluates throughput across the distributed system
 * 4. Calculates network overhead and latency effects
 *
 * NOTE: This is a simulation of distributed performance. For actual multi-machine
 * testing, this code should be deployed to separate physical machines.
 */
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::time::sleep;

use blockchain_node::execution::executor::TransactionExecutor;
use blockchain_node::execution::parallel::{
    ConflictStrategy, ParallelConfig, ParallelExecutionManager,
};
use blockchain_node::ledger::state::StateTree;
use blockchain_node::ledger::transaction::{Transaction, TransactionStatus, TransactionType};

// Network benchmark configuration
#[derive(Clone)]
struct NetworkBenchmarkConfig {
    duration_secs: u64,           // Test duration
    num_nodes: usize,             // Number of nodes in the network
    transactions_per_second: u32, // Network-wide TPS target
    avg_latency_ms: u64,          // Average network latency between nodes
    latency_variance_ms: u64,     // Variance in network latency
    packet_loss_percent: u8,      // Percentage of packet loss to simulate
    topology: NetworkTopology,    // Network topology type
}

// Network topology types
#[derive(Debug, Clone)]
enum NetworkTopology {
    FullMesh,   // Every node connected to every other node
    #[allow(dead_code)]
    Ring,       // Each node connected to two neighbors
    #[allow(dead_code)]
    Star,       // All nodes connected to a central node
    SmallWorld, // Realistic internet-like topology
}

// Message types for network simulation
enum NetworkMessage {
    Transaction(Transaction),
    TransactionAck {
        tx_hash: String,
        node_id: usize,
    },
    Block {
        transactions: Vec<String>,
        node_id: usize,
    }, // Simplified block representation
    #[allow(dead_code)]
    BlockAck {
        block_hash: String,
        node_id: usize,
    },
}

// Node in the simulated network
struct NetworkNode {
    id: usize,
    connections: Vec<usize>, // IDs of connected nodes
    execution_manager: ParallelExecutionManager,
    tx_queue: Vec<Transaction>,           // Pending transactions
    processed_tx_hashes: HashSet<String>, // Transactions this node has processed
    confirmed_tx_hashes: HashSet<String>, // Transactions confirmed in blocks
    blocks: Vec<Vec<String>>,             // Simplified blocks (just tx hashes)
    metrics: NodeMetrics,
    #[allow(dead_code)]
    state_tree: Arc<StateTree>,
}

// Metrics for each node
struct NodeMetrics {
    transactions_received: usize,
    transactions_processed: usize,
    blocks_created: usize,
    blocks_received: usize,
    total_processing_time: Duration,
    network_latency: Duration,
    consensus_times: Vec<Duration>,
}

impl NetworkNode {
    fn new(id: usize, max_parallel: usize) -> Self {
        let state_tree = Arc::new(StateTree::new());
        let executor = Arc::new(TransactionExecutor::new());

        let parallel_config = ParallelConfig {
            max_parallel,
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

        let execution_manager =
            ParallelExecutionManager::new(parallel_config, Arc::clone(&state_tree), executor);

        Self {
            id,
            connections: Vec::new(),
            execution_manager,
            tx_queue: Vec::new(),
            processed_tx_hashes: HashSet::new(),
            confirmed_tx_hashes: HashSet::new(),
            blocks: Vec::new(),
            metrics: NodeMetrics {
                transactions_received: 0,
                transactions_processed: 0,
                blocks_created: 0,
                blocks_received: 0,
                total_processing_time: Duration::new(0, 0),
                network_latency: Duration::new(0, 0),
                consensus_times: Vec::new(),
            },
            state_tree,
        }
    }

    // Process a new transaction that arrived over the network
    async fn process_transaction(&mut self, tx: Transaction) -> Result<()> {
        let tx_hash = format!("{}_{}", tx.sender, tx.nonce); // Simplified hash

        // Skip if already processed
        if self.processed_tx_hashes.contains(&tx_hash) {
            return Ok(());
        }

        // Add to queue
        self.tx_queue.push(tx.clone());
        self.metrics.transactions_received += 1;

        // Process transaction
        let start = Instant::now();
        let _result = self
            .execution_manager
            .process_transactions(vec![tx])
            .await?;
        let processing_time = start.elapsed();

        // Update metrics
        self.metrics.total_processing_time += processing_time;

        // Mark as processed
        self.processed_tx_hashes.insert(tx_hash);
        self.metrics.transactions_processed += 1;

        Ok(())
    }

    // Create a block from pending transactions
    fn create_block(&mut self) -> Vec<String> {
        // Get transaction hashes from queue
        let tx_hashes: Vec<String> = self
            .tx_queue
            .iter()
            .map(|tx| format!("{}_{}", tx.sender, tx.nonce))
            .collect();

        if !tx_hashes.is_empty() {
            // Clear queue
            self.tx_queue.clear();

            // Add to confirmed transactions
            for hash in &tx_hashes {
                self.confirmed_tx_hashes.insert(hash.clone());
            }

            // Store block
            self.blocks.push(tx_hashes.clone());
            self.metrics.blocks_created += 1;

            tx_hashes
        } else {
            Vec::new()
        }
    }

    // Process a block received from another node
    fn process_block(&mut self, tx_hashes: Vec<String>, consensus_time: Duration) {
        if !tx_hashes.is_empty() {
            // Add transactions to confirmed set
            for hash in &tx_hashes {
                self.confirmed_tx_hashes.insert(hash.clone());
            }

            // Remove any matching transactions from queue
            self.tx_queue.retain(|tx| {
                let hash = format!("{}_{}", tx.sender, tx.nonce);
                !tx_hashes.contains(&hash)
            });

            // Store block
            self.blocks.push(tx_hashes);
            self.metrics.blocks_received += 1;
            self.metrics.consensus_times.push(consensus_time);
        }
    }
}

// Simulate network topology creation
fn create_network_topology(num_nodes: usize, topology: &NetworkTopology) -> Vec<Vec<usize>> {
    let mut connections = vec![Vec::new(); num_nodes];
    let mut rng = StdRng::from_entropy();

    match topology {
        NetworkTopology::FullMesh => {
            // Every node connected to every other node
            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    if i != j {
                        connections[i].push(j);
                    }
                }
            }
        }
        NetworkTopology::Ring => {
            // Each node connected to two neighbors
            for i in 0..num_nodes {
                let prev = (i + num_nodes - 1) % num_nodes;
                let next = (i + 1) % num_nodes;
                connections[i].push(prev);
                connections[i].push(next);
            }
        }
        NetworkTopology::Star => {
            // All nodes connected to node 0
            for i in 1..num_nodes {
                connections[0].push(i);
                connections[i].push(0);
            }
        }
        NetworkTopology::SmallWorld => {
            // Start with ring topology
            for i in 0..num_nodes {
                let prev = (i + num_nodes - 1) % num_nodes;
                let next = (i + 1) % num_nodes;
                connections[i].push(prev);
                connections[i].push(next);
            }

            // Add random long-distance connections
            // Each node has 10% chance of connecting to any other non-neighbor
            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    // Skip self, neighbors, and already connected
                    if i == j || connections[i].contains(&j) {
                        continue;
                    }

                    // 10% chance of connection
                    if rng.gen_range(0..10) == 0 {
                        connections[i].push(j);
                        connections[j].push(i); // Bidirectional
                    }
                }
            }
        }
    }

    connections
}

// Run the network benchmark
async fn run_network_benchmark(config: NetworkBenchmarkConfig) -> Result<()> {
    println!("Starting network benchmark with:");
    println!("  - {} seconds duration", config.duration_secs);
    println!("  - {} nodes", config.num_nodes);
    println!(
        "  - {} transactions per second",
        config.transactions_per_second
    );
    println!(
        "  - {} ms average latency (±{} ms variance)",
        config.avg_latency_ms, config.latency_variance_ms
    );
    println!("  - {}% packet loss", config.packet_loss_percent);
    println!("  - {:?} network topology", config.topology);

    // Create nodes
    let mut nodes: Vec<NetworkNode> = (0..config.num_nodes)
        .map(|id| {
            let max_parallel = 4 + id % 5; // Vary parallelism between nodes (4-8)
            NetworkNode::new(id, max_parallel)
        })
        .collect();

    // Set up network topology
    let node_connections = create_network_topology(config.num_nodes, &config.topology);
    for (i, connections) in node_connections.into_iter().enumerate() {
        nodes[i].connections = connections;
    }

    // Create communication channels
    let (tx_global, mut rx_global) = mpsc::channel::<(
        NetworkMessage,
        usize,   // Target node ID
        Instant, // Message send time
    )>(10000); // Buffer size

    // Create transaction generator
    let _transaction_interval =
        Duration::from_micros(1_000_000 / config.transactions_per_second as u64);

    // Shared metrics
    let total_transactions = Arc::new(Mutex::new(0usize));
    let network_messages = Arc::new(Mutex::new(0usize));

    // Start benchmark
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(config.duration_secs);

    // Use tokio::sync::Mutex for thread-safe interaction
    let nodes_ref = Arc::new(Mutex::new(nodes));
    let confirmed_transactions = Arc::new(Mutex::new(0usize));

    // Clone the Arc before moving into async block
    let nodes_ref_for_handler = Arc::clone(&nodes_ref);
    let confirmed_transactions_for_handler = Arc::clone(&confirmed_transactions);
    let network_messages_clone = Arc::clone(&network_messages);

    let tx_global_msg = tx_global.clone();
    let message_handler = tokio::spawn(async move {
        // Create a thread-safe RNG
        let mut rng = StdRng::from_entropy();

        while let Some((message, target_id, sent_time)) = rx_global.recv().await {
            // Update message count
            {
                let mut count = network_messages_clone.lock().await;
                *count += 1;
            }

            // Get latency for this message
            let base_latency = config.avg_latency_ms;
            let variance = config.latency_variance_ms;

            // Fix overflow by using checked operations and avoiding negative values
            let latency = if variance > 0 {
                // Generate a random value between 0 and 2*variance, then subtract variance
                // Use saturating operations to avoid overflow
                let random_variance = rng.gen_range(0..=variance.saturating_mul(2));
                if random_variance > variance {
                    base_latency.saturating_add(random_variance - variance)
                } else {
                    base_latency.saturating_sub(variance - random_variance)
                }
            } else {
                base_latency
            };

            // Apply minimum latency of 1ms
            let latency = std::cmp::max(1, latency);
            let network_latency = Duration::from_millis(latency);

            // Simulate packet loss
            if rng.gen_range(0..100) < config.packet_loss_percent {
                continue; // Message lost
            }

            // Apply network delay
            sleep(network_latency).await;

            // Process the message
            match message {
                NetworkMessage::Transaction(tx) => {
                    let tx_hash: String;
                    let connections: Vec<usize>;

                    {
                        let mut nodes = nodes_ref_for_handler.lock().await;

                        // Process transaction
                        if let Err(e) = nodes[target_id].process_transaction(tx.clone()).await {
                            eprintln!("Error processing transaction: {}", e);
                            continue;
                        }

                        // Get values we need outside the lock
                        tx_hash = format!("{}_{}", tx.sender, tx.nonce);
                        connections = nodes[target_id].connections.clone();
                    }

                    // Propagate to connected nodes
                    for &connected_id in &connections {
                        if connected_id != target_id {
                            if let Err(e) = tx_global_msg
                                .send((
                                    NetworkMessage::Transaction(tx.clone()),
                                    connected_id,
                                    Instant::now(),
                                ))
                                .await
                            {
                                eprintln!("Error sending transaction: {}", e);
                            }
                        }
                    }

                    // Send acknowledgement
                    for &connected_id in &connections {
                        if let Err(e) = tx_global_msg
                            .send((
                                NetworkMessage::TransactionAck {
                                    tx_hash: tx_hash.clone(),
                                    node_id: target_id,
                                },
                                connected_id,
                                Instant::now(),
                            ))
                            .await
                        {
                            eprintln!("Error sending TransactionAck: {}", e);
                        }
                    }
                }
                NetworkMessage::TransactionAck { tx_hash, node_id } => {
                    // No async operations here, can keep the lock
                    let connections;
                    {
                        let nodes = nodes_ref_for_handler.lock().await;
                        connections = nodes[target_id].connections.clone();
                    }

                    for &connected_id in &connections {
                        if connected_id != node_id {
                            if let Err(e) = tx_global_msg
                                .send((
                                    NetworkMessage::TransactionAck {
                                        tx_hash: tx_hash.clone(),
                                        node_id,
                                    },
                                    connected_id,
                                    Instant::now(),
                                ))
                                .await
                            {
                                eprintln!("Error sending TransactionAck: {}", e);
                            }
                        }
                    }
                }
                NetworkMessage::Block {
                    transactions,
                    node_id,
                } => {
                    let consensus_time = Instant::now() - sent_time;
                    let connections;

                    {
                        let mut nodes = nodes_ref_for_handler.lock().await;
                        nodes[target_id].process_block(transactions.clone(), consensus_time);
                        connections = nodes[target_id].connections.clone();
                    }

                    // Update confirmed transactions count - outside of nodes lock
                    {
                        let mut confirmed = confirmed_transactions_for_handler.lock().await;
                        *confirmed += transactions.len();
                    }

                    for &connected_id in &connections {
                        if connected_id != node_id {
                            if let Err(e) = tx_global_msg
                                .send((
                                    NetworkMessage::Block {
                                        transactions: transactions.clone(),
                                        node_id,
                                    },
                                    connected_id,
                                    sent_time,
                                ))
                                .await
                            {
                                eprintln!("Error sending Block: {}", e);
                            }
                        }
                    }
                }
                NetworkMessage::BlockAck {
                    block_hash,
                    node_id,
                } => {
                    // Just propagate the acknowledgment
                    let connections;
                    {
                        let nodes = nodes_ref_for_handler.lock().await;
                        connections = nodes[target_id].connections.clone();
                    }

                    for &connected_id in &connections {
                        if connected_id != node_id {
                            if let Err(e) = tx_global_msg
                                .send((
                                    NetworkMessage::BlockAck {
                                        block_hash: block_hash.clone(),
                                        node_id,
                                    },
                                    connected_id,
                                    Instant::now(),
                                ))
                                .await
                            {
                                eprintln!("Error sending BlockAck: {}", e);
                            }
                        }
                    }
                }
            }

            // Add network latency to node metrics
            {
                let mut nodes = nodes_ref_for_handler.lock().await;
                nodes[target_id].metrics.network_latency += network_latency;
            }
        }
    });

    // Clone references needed for transaction generator
    let tx_global_tx = tx_global.clone();
    let config_clone = config.clone();
    let total_transactions_clone = Arc::clone(&total_transactions);

    let transaction_generator = tokio::spawn(async move {
        let mut next_id = 0;
        let mut last_tx_time = Instant::now();

        // Create a thread-safe RNG
        let mut rng = StdRng::from_entropy();

        while Instant::now() < end_time {
            // Generate transactions based on TPS
            let elapsed = Instant::now() - last_tx_time;
            let tx_count =
                (config_clone.transactions_per_second as f64 * elapsed.as_secs_f64()) as usize;

            for _ in 0..tx_count {
                // Create and send transaction
                let tx = generate_transaction(next_id);
                next_id += 1;

                // Send to random node
                let target_node = rng.gen_range(0..config_clone.num_nodes);
                if let Err(e) = tx_global_tx
                    .send((NetworkMessage::Transaction(tx), target_node, Instant::now()))
                    .await
                {
                    eprintln!("Error sending transaction: {}", e);
                    break; // Channel closed, stop generating
                }

                // Update total transaction count
                {
                    let mut total = total_transactions_clone.lock().await;
                    *total += 1;
                }
            }

            last_tx_time = Instant::now();
            sleep(Duration::from_millis(100)).await;
        }
    });

    // Clone references needed for block creator
    let tx_global_block = tx_global.clone();
    let nodes_ref_for_creator = Arc::clone(&nodes_ref);
    let config_clone = config.clone();

    let block_creator = tokio::spawn(async move {
        // Block interval (1 block every 2 seconds)
        let block_interval = Duration::from_secs(2);
        let mut last_block_time = Instant::now();

        // Create a thread-safe RNG
        let mut rng = StdRng::from_entropy();

        loop {
            // Check if test is done
            if Instant::now() >= end_time {
                break;
            }

            // Wait until next block time
            let elapsed = Instant::now() - last_block_time;
            if elapsed < block_interval {
                sleep(block_interval - elapsed).await;
            }

            // Select a random node to be the block creator
            let block_creator_id = rng.gen_range(0..config_clone.num_nodes);

            // Create block
            let tx_hashes = {
                let mut nodes = nodes_ref_for_creator.lock().await;
                nodes[block_creator_id].create_block()
            };

            if !tx_hashes.is_empty() {
                // Broadcast block to connected nodes
                let connections = {
                    let nodes = nodes_ref_for_creator.lock().await;
                    nodes[block_creator_id].connections.clone()
                };

                for &connected_id in &connections {
                    if let Err(e) = tx_global_block
                        .send((
                            NetworkMessage::Block {
                                transactions: tx_hashes.clone(),
                                node_id: block_creator_id,
                            },
                            connected_id,
                            Instant::now(),
                        ))
                        .await
                    {
                        eprintln!("Error sending block: {}", e);
                        break; // Channel closed, stop creating blocks
                    }
                }
            }

            last_block_time = Instant::now();
        }
    });

    // Start progress reporting task
    let total_transactions_clone = Arc::clone(&total_transactions);
    let confirmed_transactions_clone = Arc::clone(&confirmed_transactions);
    let network_messages_clone = Arc::clone(&network_messages);

    let progress_reporter = tokio::spawn(async move {
        let report_interval = Duration::from_secs(2);

        while Instant::now() < end_time {
            sleep(report_interval).await;

            // Get metrics
            let elapsed = Instant::now() - start_time;
            let total_tx = {
                let total = total_transactions_clone.lock().await;
                *total
            };
            let confirmed_tx = {
                let confirmed = confirmed_transactions_clone.lock().await;
                *confirmed
            };
            let network_msg_count = {
                let count = network_messages_clone.lock().await;
                *count
            };

            // Print progress
            println!(
                "Progress: {:.1}s elapsed, {} tx generated, {} tx confirmed, {} messages",
                elapsed.as_secs_f64(),
                total_tx,
                confirmed_tx,
                network_msg_count
            );
        }
    });

    // Wait for the test duration
    println!("Benchmark running for {} seconds...", config.duration_secs);
    sleep(Duration::from_secs(config.duration_secs)).await;

    // Send end signal to all tasks by dropping the tx channel
    drop(tx_global);

    // Wait for tasks to complete
    // Use tokio::try_join! to handle errors with all tasks
    let task_results = tokio::try_join!(
        message_handler,
        transaction_generator,
        block_creator,
        progress_reporter
    );

    if let Err(e) = task_results {
        eprintln!("Error in tasks: {}", e);
    }

    // Collect final metrics
    println!("\nBenchmark completed. Gathering results...");

    let nodes_final = nodes_ref.lock().await;
    let _total_txs_received: usize = nodes_final
        .iter()
        .map(|node| node.metrics.transactions_received)
        .sum();
    let total_txs_processed: usize = nodes_final
        .iter()
        .map(|node| node.metrics.transactions_processed)
        .sum();
    let total_blocks_created: usize = nodes_final
        .iter()
        .map(|node| node.metrics.blocks_created)
        .sum();
    let total_blocks_received: usize = nodes_final
        .iter()
        .map(|node| node.metrics.blocks_received)
        .sum();

    // Calculate average consensus time
    let mut total_consensus_time = Duration::new(0, 0);
    let mut consensus_count = 0;

    for node in nodes_final.iter() {
        for time in &node.metrics.consensus_times {
            total_consensus_time += *time;
            consensus_count += 1;
        }
    }

    let avg_consensus_time = if consensus_count > 0 {
        total_consensus_time / consensus_count as u32
    } else {
        Duration::new(0, 0)
    };

    // Calculate network efficiency
    let total_txs = {
        let total = total_transactions.lock().await;
        *total
    };

    let confirmed_txs = {
        let confirmed = confirmed_transactions.lock().await;
        *confirmed
    };

    let network_msg_count = {
        let count = network_messages.lock().await;
        *count
    };

    let messages_per_tx = if total_txs > 0 {
        network_msg_count as f64 / total_txs as f64
    } else {
        0.0
    };

    let confirmation_rate = if total_txs > 0 {
        (confirmed_txs as f64 / total_txs as f64) * 100.0
    } else {
        0.0
    };

    // Print final report
    println!("\n\nNetwork Benchmark Complete");
    println!("=========================");
    println!("Test duration: {:.2?}", start_time.elapsed());
    println!(
        "Topology: {:?} with {} nodes",
        config.topology, config.num_nodes
    );
    println!(
        "Network conditions: {}ms latency, {}% packet loss",
        config.avg_latency_ms, config.packet_loss_percent
    );
    println!("\nPerformance:");
    println!("  Transactions generated: {}", total_txs);
    println!("  Transactions processed: {}", total_txs_processed);
    println!("  Transactions confirmed: {}", confirmed_txs);
    println!("  Confirmation rate: {:.2}%", confirmation_rate);
    println!("  Blocks created: {}", total_blocks_created);
    println!("  Blocks propagated: {}", total_blocks_received);
    println!("  Network messages sent: {}", network_msg_count);
    println!("  Messages per transaction: {:.2}", messages_per_tx);
    println!("  Average consensus time: {:.2?}", avg_consensus_time);

    // Print performance by node
    println!("\nPer-Node Statistics:");
    println!("| Node | Txs Received | Txs Processed | Blocks Created | Blocks Received |");
    println!("|------|--------------|---------------|----------------|-----------------|");
    for node in nodes_final.iter() {
        println!(
            "| {:4} | {:12} | {:13} | {:14} | {:15} |",
            node.id,
            node.metrics.transactions_received,
            node.metrics.transactions_processed,
            node.metrics.blocks_created,
            node.metrics.blocks_received
        );
    }

    Ok(())
}

// Generate a random transaction for benchmarking
fn generate_transaction(id: usize) -> Transaction {
    // Use a deterministic RNG seeded with the transaction ID for reproducibility
    let mut rng = StdRng::seed_from_u64(id as u64);

    // Generate random transaction data
    let sender = format!("wallet_{}", rng.gen_range(0..1000));
    let recipient = format!("wallet_{}", rng.gen_range(0..1000));
    let amount = rng.gen_range(1..1000);
    let nonce = id as u64; // Use ID as nonce for uniqueness

    // Create transaction
    Transaction {
        tx_type: TransactionType::Transfer,
        sender,
        recipient,
        amount,
        nonce,
        gas_price: 1,
        gas_limit: 21000,
        data: vec![0; 32],      // Dummy data
        signature: vec![0; 64], // Dummy signature
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        #[cfg(feature = "bls")]
        bls_signature: None,
        status: TransactionStatus::Pending,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Network Benchmark: Simulating Distributed Blockchain");
    println!("===================================================\n");

    // Network benchmark configuration
    let configs = vec![
        NetworkBenchmarkConfig {
            duration_secs: 30,
            num_nodes: 10,
            transactions_per_second: 50,
            avg_latency_ms: 50,
            latency_variance_ms: 20,
            packet_loss_percent: 2,
            topology: NetworkTopology::FullMesh,
        },
        NetworkBenchmarkConfig {
            duration_secs: 30,
            num_nodes: 20,
            transactions_per_second: 100,
            avg_latency_ms: 100,
            latency_variance_ms: 50,
            packet_loss_percent: 5,
            topology: NetworkTopology::SmallWorld,
        },
    ];

    println!(
        "Running network benchmark with {} configurations",
        configs.len()
    );

    // Run the selected configuration
    match run_network_benchmark(configs[0].clone()).await {
        Ok(_) => println!("Network benchmark completed successfully"),
        Err(e) => eprintln!("Network benchmark failed: {}", e),
    }

    // For more complex tests
    // for (i, config) in configs.iter().enumerate() {
    //     println!("\nRunning configuration {}/{}:", i + 1, configs.len());
    //     if let Err(e) = run_network_benchmark(config.clone()).await {
    //         eprintln!("Configuration {} failed: {}", i + 1, e);
    //     }
    // }

    Ok(())
}
