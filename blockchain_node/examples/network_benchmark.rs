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

use std::time::{Instant, Duration};
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet};
use anyhow::{Result, anyhow};
use rand::{thread_rng, Rng};
use tokio::sync::mpsc;
use tokio::time::sleep;

use blockchain_node::transaction::Transaction;
use blockchain_node::ledger::state::StateTree;
use blockchain_node::ledger::state::storage::StateStorage;
use blockchain_node::execution::executor::TransactionExecutor;
use blockchain_node::execution::parallel::{ParallelExecutionManager, ParallelConfig, ConflictStrategy};

// Network benchmark configuration
struct NetworkBenchmarkConfig {
    duration_secs: u64,        // Test duration
    num_nodes: usize,          // Number of nodes in the network
    transactions_per_second: u32, // Network-wide TPS target
    avg_latency_ms: u64,       // Average network latency between nodes
    latency_variance_ms: u64,  // Variance in network latency
    packet_loss_percent: u8,   // Percentage of packet loss to simulate
    topology: NetworkTopology, // Network topology type
}

// Network topology types
enum NetworkTopology {
    FullMesh,      // Every node connected to every other node
    Ring,          // Each node connected to two neighbors
    Star,          // All nodes connected to a central node
    SmallWorld,    // Realistic internet-like topology
}

// Message types for network simulation
enum NetworkMessage {
    Transaction(Transaction),
    TransactionAck { tx_hash: String, node_id: usize },
    Block { transactions: Vec<String>, node_id: usize }, // Simplified block representation
    BlockAck { block_hash: String, node_id: usize },
}

// Node in the simulated network
struct NetworkNode {
    id: usize,
    connections: Vec<usize>,               // IDs of connected nodes
    execution_manager: ParallelExecutionManager,
    tx_queue: Vec<Transaction>,            // Pending transactions
    processed_tx_hashes: HashSet<String>,  // Transactions this node has processed
    confirmed_tx_hashes: HashSet<String>,  // Transactions confirmed in blocks
    blocks: Vec<Vec<String>>,              // Simplified blocks (just tx hashes)
    metrics: NodeMetrics,
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
        };
        
        let execution_manager = ParallelExecutionManager::new(
            parallel_config,
            Arc::clone(&state_tree),
            executor
        );
        
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
        let result = self.execution_manager.process_transactions(vec![tx]).await?;
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
        let tx_hashes: Vec<String> = self.tx_queue.iter()
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
    let mut rng = thread_rng();
    
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
        },
        NetworkTopology::Ring => {
            // Each node connected to two neighbors
            for i in 0..num_nodes {
                let prev = (i + num_nodes - 1) % num_nodes;
                let next = (i + 1) % num_nodes;
                connections[i].push(prev);
                connections[i].push(next);
            }
        },
        NetworkTopology::Star => {
            // All nodes connected to node 0
            for i in 1..num_nodes {
                connections[0].push(i);
                connections[i].push(0);
            }
        },
        NetworkTopology::SmallWorld => {
            // Start with ring topology
            for i in 0..num_nodes {
                let prev = (i + num_nodes - 1) % num_nodes;
                let next = (i + 1) % num_nodes;
                connections[i].push(prev);
                connections[i].push(next);
                
                // Add random long-distance connections (small world property)
                let num_long_connections = rng.gen_range(1..4);
                for _ in 0..num_long_connections {
                    let target = rng.gen_range(0..num_nodes);
                    if target != i && !connections[i].contains(&target) {
                        connections[i].push(target);
                        connections[target].push(i);
                    }
                }
            }
        },
    }
    
    connections
}

// Run the network benchmark
async fn run_network_benchmark(config: &NetworkBenchmarkConfig) -> Result<()> {
    println!("Starting network benchmark with:");
    println!("  - {} seconds duration", config.duration_secs);
    println!("  - {} nodes", config.num_nodes);
    println!("  - {} transactions per second", config.transactions_per_second);
    println!("  - {} ms average latency (±{} ms variance)", 
             config.avg_latency_ms, config.latency_variance_ms);
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
    let (tx_global, mut rx_global) = mpsc::channel::<(NetworkMessage, usize, Instant)>(1000);
    
    // Create transaction generator
    let transaction_interval = Duration::from_micros(1_000_000 / config.transactions_per_second as u64);
    
    // Shared metrics
    let total_transactions = Arc::new(Mutex::new(0usize));
    let confirmed_transactions = Arc::new(Mutex::new(0usize));
    let network_messages = Arc::new(Mutex::new(0usize));
    
    // Start benchmark
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(config.duration_secs);
    
    // Spawn message handler task
    let tx_global_clone = tx_global.clone();
    let config_clone = config.clone();
    let nodes_ref = Arc::new(Mutex::new(nodes));
    let network_messages_clone = Arc::clone(&network_messages);
    
    let message_handler = tokio::spawn(async move {
        let mut rng = thread_rng();
        
        while let Some((message, target_id, sent_time)) = rx_global.recv().await {
            // Update message count
            {
                let mut count = network_messages_clone.lock().unwrap();
                *count += 1;
            }
            
            // Get latency for this message
            let base_latency = config_clone.avg_latency_ms;
            let variance = config_clone.latency_variance_ms;
            let latency = base_latency + if variance > 0 {
                rng.gen_range(0..variance * 2).saturating_sub(variance)
            } else {
                0
            };
            
            // Simulate network latency
            sleep(Duration::from_millis(latency)).await;
            
            // Simulate packet loss
            if rng.gen_range(0..100) < config_clone.packet_loss_percent {
                continue; // Message lost
            }
            
            // Calculate network latency
            let network_latency = sent_time.elapsed();
            
            // Process message
            let mut nodes = nodes_ref.lock().unwrap();
            
            match message {
                NetworkMessage::Transaction(tx) => {
                    if let Err(e) = nodes[target_id].process_transaction(tx.clone()).await {
                        println!("Error processing transaction on node {}: {}", target_id, e);
                    } else {
                        // Propagate to connected nodes
                        for &connected_id in &nodes[target_id].connections {
                            if connected_id != target_id {
                                tx_global_clone.send((
                                    NetworkMessage::Transaction(tx.clone()),
                                    connected_id,
                                    Instant::now()
                                )).await.unwrap();
                            }
                        }
                        
                        // Send acknowledgement
                        let tx_hash = format!("{}_{}", tx.sender, tx.nonce);
                        for &connected_id in &nodes[target_id].connections {
                            tx_global_clone.send((
                                NetworkMessage::TransactionAck { 
                                    tx_hash: tx_hash.clone(), 
                                    node_id: target_id 
                                },
                                connected_id,
                                Instant::now()
                            )).await.unwrap();
                        }
                    }
                },
                NetworkMessage::Block { transactions, node_id } => {
                    // Process block with consensus time
                    nodes[target_id].process_block(transactions.clone(), network_latency);
                    
                    // Propagate to connected nodes
                    for &connected_id in &nodes[target_id].connections {
                        if connected_id != node_id {
                            tx_global_clone.send((
                                NetworkMessage::Block { 
                                    transactions: transactions.clone(),
                                    node_id: target_id
                                },
                                connected_id,
                                Instant::now()
                            )).await.unwrap();
                        }
                    }
                    
                    // Update confirmed transactions
                    {
                        let mut confirmed = confirmed_transactions.lock().unwrap();
                        *confirmed += transactions.len();
                    }
                },
                _ => {
                    // Handle other message types as needed
                }
            }
            
            // Add network latency to node metrics
            nodes[target_id].metrics.network_latency += network_latency;
        }
    });
    
    // Transaction generation task
    let tx_global_clone = tx_global.clone();
    let total_transactions_clone = Arc::clone(&total_transactions);
    
    let transaction_generator = tokio::spawn(async move {
        let mut next_id = 0;
        let mut last_tx_time = Instant::now();
        
        while Instant::now() < end_time {
            // Wait until next transaction time
            let elapsed = last_tx_time.elapsed();
            if elapsed < transaction_interval {
                sleep(transaction_interval - elapsed).await;
            }
            
            // Generate transaction
            let tx = generate_transaction(next_id);
            next_id += 1;
            
            // Send to random node
            let target_node = thread_rng().gen_range(0..config.num_nodes);
            tx_global_clone.send((
                NetworkMessage::Transaction(tx),
                target_node,
                Instant::now()
            )).await.unwrap();
            
            // Update total transaction count
            {
                let mut total = total_transactions_clone.lock().unwrap();
                *total += 1;
            }
            
            last_tx_time = Instant::now();
        }
    });
    
    // Block creation task
    let tx_global_clone = tx_global.clone();
    let nodes_ref_clone = Arc::clone(&nodes_ref);
    
    let block_creator = tokio::spawn(async move {
        // Block interval (1 block every 2 seconds)
        let block_interval = Duration::from_secs(2);
        let mut last_block_time = Instant::now();
        
        while Instant::now() < end_time {
            // Wait until next block time
            let elapsed = last_block_time.elapsed();
            if elapsed < block_interval {
                sleep(block_interval - elapsed).await;
            }
            
            // Choose random node to create block
            let block_creator_id = thread_rng().gen_range(0..config.num_nodes);
            
            // Create block
            let tx_hashes = {
                let mut nodes = nodes_ref_clone.lock().unwrap();
                nodes[block_creator_id].create_block()
            };
            
            if !tx_hashes.is_empty() {
                // Broadcast block to connected nodes
                let mut nodes = nodes_ref_clone.lock().unwrap();
                for &connected_id in &nodes[block_creator_id].connections {
                    tx_global_clone.send((
                        NetworkMessage::Block { 
                            transactions: tx_hashes.clone(),
                            node_id: block_creator_id
                        },
                        connected_id,
                        Instant::now()
                    )).await.unwrap();
                }
            }
            
            last_block_time = Instant::now();
        }
    });
    
    // Wait for test to complete
    let report_interval = Duration::from_secs(2);
    let mut last_report_time = start_time;
    
    while Instant::now() < end_time {
        // Sleep for a bit
        sleep(Duration::from_millis(500)).await;
        
        // Periodic reporting
        if Instant::now() - last_report_time >= report_interval {
            let elapsed = Instant::now() - start_time;
            let total_tx = {
                let total = total_transactions.lock().unwrap();
                *total
            };
            let confirmed_tx = {
                let confirmed = confirmed_transactions.lock().unwrap();
                *confirmed
            };
            let network_msg_count = {
                let count = network_messages.lock().unwrap();
                *count
            };
            
            println!("[{:5.1?}] Generated: {} txs | Confirmed: {} txs | Messages: {} | Confirmation Rate: {:.1}%",
                     elapsed, total_tx, confirmed_tx,
                     network_msg_count,
                     if total_tx > 0 { (confirmed_tx as f64 / total_tx as f64) * 100.0 } else { 0.0 });
            
            last_report_time = Instant::now();
        }
    }
    
    // Wait for tasks to finish
    message_handler.abort();
    transaction_generator.abort();
    block_creator.abort();
    
    // Calculate statistics
    let test_duration = start_time.elapsed();
    
    // Collect node metrics
    let nodes_final = nodes_ref.lock().unwrap();
    let total_txs_received: usize = nodes_final.iter().map(|n| n.metrics.transactions_received).sum();
    let total_txs_processed: usize = nodes_final.iter().map(|n| n.metrics.transactions_processed).sum();
    let total_blocks_created: usize = nodes_final.iter().map(|n| n.metrics.blocks_created).sum();
    let total_blocks_received: usize = nodes_final.iter().map(|n| n.metrics.blocks_received).sum();
    
    // Calculate average consensus time
    let mut all_consensus_times = Vec::new();
    for node in nodes_final.iter() {
        all_consensus_times.extend(&node.metrics.consensus_times);
    }
    
    let avg_consensus_time = if !all_consensus_times.is_empty() {
        all_consensus_times.iter().sum::<Duration>() / all_consensus_times.len() as u32
    } else {
        Duration::new(0, 0)
    };
    
    // Calculate network efficiency
    let total_txs = {
        let total = total_transactions.lock().unwrap();
        *total
    };
    
    let confirmed_txs = {
        let confirmed = confirmed_transactions.lock().unwrap();
        *confirmed
    };
    
    let network_msg_count = {
        let count = network_messages.lock().unwrap();
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
    println!("Test duration: {:.2?}", test_duration);
    println!("Topology: {:?} with {} nodes", config.topology, config.num_nodes);
    println!("Network conditions: {}ms latency, {}% packet loss", config.avg_latency_ms, config.packet_loss_percent);
    println!("\nOverall Statistics:");
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
        println!("| {:4} | {:12} | {:13} | {:14} | {:15} |",
                 node.id, 
                 node.metrics.transactions_received,
                 node.metrics.transactions_processed,
                 node.metrics.blocks_created,
                 node.metrics.blocks_received);
    }
    
    Ok(())
}

fn generate_transaction(id: usize) -> Transaction {
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut rng = thread_rng();
    
    // Create random-looking addresses from the ID
    let sender = format!("sender{}", id % 100);
    let recipient = format!("recipient{}", rng.gen_range(0..100));
    
    // Generate random data based on ID
    let mut data = Vec::new();
    data.extend_from_slice(&id.to_be_bytes());
    data.extend_from_slice(&[id as u8; 32]);
    
    Transaction {
        tx_type: blockchain_node::ledger::transaction::TransactionType::Transfer,
        sender,
        recipient,
        amount: (id as u64) * 100,
        nonce: id as u64,
        gas_price: 1,
        gas_limit: 21000,
        data,
        signature: Vec::new(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        #[cfg(feature = "bls")]
        bls_signature: None,
        status: blockchain_node::ledger::transaction::TransactionStatus::Pending,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Blockchain Network Layer Benchmark");
    println!("=================================\n");
    
    // Define benchmark configurations
    let configs = vec![
        // Small network with ideal conditions (quick test)
        NetworkBenchmarkConfig {
            duration_secs: 30,
            num_nodes: 4,
            transactions_per_second: 50,
            avg_latency_ms: 10,
            latency_variance_ms: 5,
            packet_loss_percent: 0,
            topology: NetworkTopology::FullMesh,
        },
        // Medium network with realistic conditions
        NetworkBenchmarkConfig {
            duration_secs: 60,
            num_nodes: 8,
            transactions_per_second: 100,
            avg_latency_ms: 50,
            latency_variance_ms: 20,
            packet_loss_percent: 2,
            topology: NetworkTopology::SmallWorld,
        },
        // Large network with challenging conditions
        NetworkBenchmarkConfig {
            duration_secs: 120,
            num_nodes: 16,
            transactions_per_second: 200,
            avg_latency_ms: 100,
            latency_variance_ms: 50,
            packet_loss_percent: 5, 
            topology: NetworkTopology::SmallWorld,
        },
    ];
    
    // Run the selected configuration
    run_network_benchmark(&configs[0]).await?;
    
    // For more complex tests
    // run_network_benchmark(&configs[1]).await?;
    // run_network_benchmark(&configs[2]).await?;
    
    Ok(())
} 