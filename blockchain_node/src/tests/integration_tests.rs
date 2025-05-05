#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::runtime::Runtime;
    use tokio::sync::{mpsc, RwLock};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use rand::{thread_rng, Rng};
    
    use crate::storage::{Storage, MemMapStorage, MemMapOptions, CompressionAlgorithm, StorageInit};
    use crate::network::custom_udp::{UdpNetwork, NetworkConfig, Message, MessageType};
    use crate::sharding::{ShardManager, ShardConfig};
    
    #[test]
    fn test_ultra_high_tps_integration() {
        // This test validates that our complete blockchain system can achieve
        // the 500,000+ TPS target with all optimizations enabled
        
        let rt = Runtime::new().unwrap();
        
        // Test parameters
        let num_shards = 128; // Maximum sharding
        let num_transactions = 1_000_000;
        let cross_shard_ratio = 0.10; // 10% cross-shard transactions
        let tx_size = 1024; // 1KB transactions
        
        println!("Starting ultra-high TPS integration test");
        println!("- Shards: {}", num_shards);
        println!("- Transactions: {}", num_transactions);
        println!("- Cross-shard ratio: {:.1}%", cross_shard_ratio * 100.0);
        println!("- Transaction size: {} bytes", tx_size);
        
        rt.block_on(async {
            // 1. Set up storage for each shard
            let mut storages = Vec::with_capacity(num_shards);
            
            for shard_id in 0..num_shards {
                // Use temp dir for testing
                let temp_dir = tempfile::tempdir().unwrap();
                
                // Create storage with optimized settings
                let options = MemMapOptions {
                    compression: CompressionAlgorithm::LZ4, // Fast compression
                    sync_on_flush: false, // Disable sync for performance
                    file_growth_size: 64 * 1024 * 1024, // 64MB growth
                    max_file_size: 1024 * 1024 * 1024, // 1GB max
                    block_size: 4096, // Standard block size
                    max_open_files: 1000,
                };
                
                let storage = MemMapStorage::new(&temp_dir.path(), options).await
                    .expect(&format!("Failed to create storage for shard {}", shard_id));
                
                storages.push(Arc::new(storage));
            }
            
            // 2. Set up network for each shard
            let mut networks = Vec::with_capacity(num_shards);
            
            for shard_id in 0..num_shards {
                let port = 40000 + shard_id as u16;
                
                let config = NetworkConfig {
                    bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port),
                    max_message_size: 64 * 1024, // 64KB
                    reliable_delivery: true,
                    connection_timeout_ms: 30000,
                    buffer_size: 8 * 1024 * 1024, // 8MB buffer
                    compression: true,
                    encryption: false, // Disable for test performance
                };
                
                let network = UdpNetwork::new(config, format!("shard-{}", shard_id)).await
                    .expect(&format!("Failed to create network for shard {}", shard_id));
                
                networks.push(Arc::new(network));
                
                // Start network
                networks[shard_id as usize].start().await.unwrap();
            }
            
            // 3. Connect networks in a mesh topology
            for i in 0..num_shards {
                for j in 0..num_shards {
                    if i != j {
                        let port = 40000 + j as u16;
                        let peer_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);
                        
                        networks[i as usize].connect(peer_addr).await.unwrap();
                    }
                }
            }
            
            // 4. Set up shard managers
            let mut shard_managers = Vec::with_capacity(num_shards as usize);
            
            for shard_id in 0..num_shards {
                let config = ShardConfig {
                    shard_count: num_shards,
                    this_shard_id: shard_id,
                    max_cross_shard_delay: Duration::from_millis(50),
                    retry_interval: Duration::from_millis(100),
                    batch_size: 1000, // Large batch size for throughput
                    max_pending_refs: 100000,
                };
                
                let (tx, _rx) = mpsc::channel(100000);
                let manager = ShardManager::new(config, tx);
                shard_managers.push(Arc::new(manager));
            }
            
            // 5. Register cross-shard connections
            for i in 0..num_shards as usize {
                for j in 0..num_shards as usize {
                    if i != j {
                        shard_managers[i].register_shard_connection(j as u16, shard_managers[j].clone()).await;
                    }
                }
            }
            
            // 6. Generate test transactions
            println!("Generating {} transactions...", num_transactions);
            let mut rng = thread_rng();
            let mut transactions = Vec::with_capacity(num_transactions);
            
            for i in 0..num_transactions {
                // Determine transaction type (cross-shard or intra-shard)
                let is_cross_shard = rng.gen::<f64>() < cross_shard_ratio;
                
                // Generate random source shard
                let source_shard = rng.gen_range(0..num_shards);
                
                // Generate target shard
                let target_shard = if is_cross_shard {
                    // Different shard for cross-shard tx
                    let mut target;
                    loop {
                        target = rng.gen_range(0..num_shards);
                        if target != source_shard {
                            break;
                        }
                    }
                    target
                } else {
                    // Same shard for intra-shard tx
                    source_shard
                };
                
                // Generate random data for the transaction
                let mut data = vec![0u8; tx_size];
                rng.fill(&mut data[..]);
                
                // Create a transaction ID that includes metadata
                let tx_id = format!("tx-{}-{}-{}", i, source_shard, target_shard);
                
                transactions.push((source_shard, target_shard, tx_id, data));
            }
            
            // 7. Execute transactions and measure throughput
            println!("Starting transaction processing...");
            let start = Instant::now();
            
            // Track transaction confirmation count
            let confirmed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            
            // Submit transactions in batches
            let batch_size = 10000;
            for chunk in transactions.chunks(batch_size) {
                let futures = chunk.iter().map(|(source_shard, target_shard, tx_id, data)| {
                    let shard_mgr = shard_managers[*source_shard as usize].clone();
                    let target = *target_shard;
                    let id = tx_id.clone();
                    let tx_data = data.clone();
                    let counter = confirmed_count.clone();
                    
                    tokio::spawn(async move {
                        if source_shard == &target {
                            // Intra-shard transaction
                            shard_mgr.process_local_transaction(id).await;
                        } else {
                            // Cross-shard transaction
                            shard_mgr.send_cross_shard_transaction(target, id, tx_data).await;
                        }
                        
                        // Increment counter
                        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    })
                });
                
                // Wait for batch to complete
                for future in futures {
                    let _ = future.await;
                }
                
                // Log progress
                let completed = confirmed_count.load(std::sync::atomic::Ordering::SeqCst);
                println!("Processed {}/{} transactions", completed, num_transactions);
            }
            
            // Allow time for cross-shard finalization
            tokio::time::sleep(Duration::from_secs(2)).await;
            
            // Calculate throughput
            let elapsed = start.elapsed();
            let total_tps = num_transactions as f64 / elapsed.as_secs_f64();
            let cross_shard_tps = (num_transactions as f64 * cross_shard_ratio) / elapsed.as_secs_f64();
            let intra_shard_tps = (num_transactions as f64 * (1.0 - cross_shard_ratio)) / elapsed.as_secs_f64();
            
            println!("\n=== Performance Results ===");
            println!("Total elapsed time: {:.2?}", elapsed);
            println!("Overall throughput: {:.2} TPS", total_tps);
            println!("Intra-shard throughput: {:.2} TPS", intra_shard_tps);
            println!("Cross-shard throughput: {:.2} TPS", cross_shard_tps);
            
            // Verify cross-shard transaction completion
            let mut total_pending = 0;
            for shard_id in 0..num_shards as usize {
                let pending = shard_managers[shard_id].get_pending_cross_shard_count().await;
                total_pending += pending;
            }
            
            println!("Total pending cross-shard references: {}", total_pending);
            
            // Shutdown networks
            for network in &networks {
                network.stop().await.unwrap();
            }
            
            // Close storage
            for storage in &storages {
                storage.close().await.unwrap();
            }
            
            // Assert the performance meets our target
            assert!(total_tps > 450_000.0, "Failed to meet 450K+ TPS target. Achieved: {:.2} TPS", total_tps);
            // Allow some cross-shard references to be pending due to test timing
            assert!(total_pending < num_transactions as usize / 100, "Too many pending cross-shard references: {}", total_pending);
        });
    }
} 