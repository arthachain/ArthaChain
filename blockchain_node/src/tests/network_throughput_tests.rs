#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use tokio::runtime::Runtime;
    use tokio::sync::{mpsc, RwLock};
    use tokio::time::sleep;
    use rand::{thread_rng, Rng};
    
    use crate::network::custom_udp::{UdpNetwork, NetworkConfig, Message, MessageType};
    
    #[test]
    fn test_network_throughput() {
        // This test will measure the throughput of our custom UDP protocol
        // by setting up a local sender and receiver
        
        let rt = Runtime::new().unwrap();
        
        // Configure network
        let sender_config = NetworkConfig {
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 40001),
            buffer_size: 8 * 1024 * 1024, // 8MB buffer
            broadcast_interval: Duration::from_millis(100),
            retry_interval: Duration::from_millis(100),
            max_packet_size: 64 * 1024, // 64KB
            fragment_size: 16 * 1024,   // 16KB for large messages
        };
        
        let receiver_config = NetworkConfig {
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 40002),
            buffer_size: 8 * 1024 * 1024, // 8MB buffer
            broadcast_interval: Duration::from_millis(100),
            retry_interval: Duration::from_millis(100),
            max_packet_size: 64 * 1024, // 64KB
            fragment_size: 16 * 1024,   // 16KB for large messages
        };
        
        // Test parameters
        let message_sizes = [1024, 10 * 1024, 100 * 1024]; // 1KB, 10KB, 100KB
        let num_messages = 1000;
        
        rt.block_on(async {
            // Create sender and receiver networks
            let sender_network = Arc::new(UdpNetwork::new(sender_config, "sender-node".to_string()).await.unwrap());
            let receiver_network = Arc::new(UdpNetwork::new(receiver_config, "receiver-node".to_string()).await.unwrap());
            
            // Start networks
            sender_network.start().await.unwrap();
            receiver_network.start().await.unwrap();
            
            // Connect the two nodes
            sender_network.add_peer(receiver_config.bind_addr).await.unwrap();
            
            // Register message handler for the receiver
            let (tx, mut rx) = mpsc::channel(num_messages);
            receiver_network.register_handler(MessageType::Data, tx).await.unwrap();
            
            // Wait a bit for connection to establish
            sleep(Duration::from_millis(500)).await;
            
            for &size in &message_sizes {
                println!("\nTesting with message size: {} bytes", size);
                
                // Generate random messages
                let mut messages = Vec::with_capacity(num_messages);
                for _ in 0..num_messages {
                    let mut rng = thread_rng();
                    let data: Vec<u8> = (0..size).map(|_| rng.gen::<u8>()).collect();
                    messages.push(data);
                }
                
                // Send messages and measure throughput
                let start = Instant::now();
                
                for (i, data) in messages.iter().enumerate() {
                    sender_network.broadcast(MessageType::Data, data.clone()).await.unwrap();
                    
                    if (i + 1) % 100 == 0 {
                        println!("Sent {} messages", i + 1);
                    }
                }
                
                // Receive messages
                let mut received = 0;
                let timeout = Duration::from_secs(30); // 30 seconds timeout
                let timeout_instant = Instant::now() + timeout;
                
                while received < num_messages && Instant::now() < timeout_instant {
                    match tokio::time::timeout(Duration::from_millis(100), rx.recv()).await {
                        Ok(Some(_)) => {
                            received += 1;
                            if received % 100 == 0 {
                                println!("Received {} messages", received);
                            }
                        },
                        Ok(None) => break,
                        Err(_) => continue,
                    }
                }
                
                let elapsed = start.elapsed();
                
                // Calculate throughput in MB/s
                let throughput = (received * size) as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);
                
                println!("Received {}/{} messages in {:.2?}", received, num_messages, elapsed);
                println!("Network throughput: {:.2} MB/s", throughput);
                
                // Verify receipt rate and minimum throughput
                let receipt_rate = received as f64 / num_messages as f64;
                assert!(receipt_rate > 0.95, "Receipt rate too low: {:.2}%", receipt_rate * 100.0);
                assert!(throughput > 5.0, "Throughput below minimum requirement: {:.2} MB/s", throughput);
            }
            
            // Shutdown networks
            sender_network.stop().await.unwrap();
            receiver_network.stop().await.unwrap();
        });
    }
} 