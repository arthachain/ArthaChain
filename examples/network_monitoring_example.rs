//! Network Monitoring API Example
//!
//! This example demonstrates how to use the network monitoring functionality
//! to track peer count, mempool size, uptime, and retrieve detailed peer information.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

// Import the necessary modules
use blockchain_node::config::Config;
use blockchain_node::ledger::state::State;
use blockchain_node::ledger::transaction::{Transaction, TransactionStatus, TransactionType};
use blockchain_node::transaction::mempool::{EnhancedMempool, MempoolConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Network Monitoring API Example");

    // Create test components
    let state = create_example_state().await?;
    let mempool = create_example_mempool().await;

    // Add some test transactions to the mempool
    populate_mempool(&mempool).await?;

    println!("✅ Created monitoring service with test data");

    // Demonstrate monitoring functionality
    demonstrate_monitoring_capabilities(&state, &mempool).await?;

    println!("🎉 Network Monitoring Example completed successfully!");
    Ok(())
}

/// Create example state
async fn create_example_state() -> Result<Arc<RwLock<State>>, Box<dyn std::error::Error>> {
    let config = Config::default();
    let state = State::new(&config)?;
    Ok(Arc::new(RwLock::new(state)))
}

/// Create example mempool with configuration
async fn create_example_mempool() -> Arc<EnhancedMempool> {
    let config = MempoolConfig {
        max_size_bytes: 10 * 1024 * 1024, // 10MB
        max_transactions: 10000,
        default_ttl: Duration::from_secs(3600), // 1 hour
        min_gas_price: 1,
        use_quantum_resistant: false,
        cleanup_interval: Duration::from_secs(300), // 5 minutes
        max_txs_per_account: 50,
    };

    Arc::new(EnhancedMempool::new(config))
}

/// Populate mempool with test transactions
async fn populate_mempool(
    mempool: &Arc<EnhancedMempool>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Adding test transactions to mempool...");

    for i in 0..25 {
        let tx = create_example_transaction(i, 10 + (i % 20));
        mempool.add_transaction(tx).await?;
    }

    let stats = mempool.get_stats().await;
    println!(
        "   Added {} transactions ({} bytes)",
        stats.total_transactions, stats.size_bytes
    );

    Ok(())
}

/// Create example transaction
fn create_example_transaction(nonce: u64, gas_price: u64) -> Transaction {
    use std::time::SystemTime;

    Transaction {
        tx_type: TransactionType::Transfer,
        sender: format!("sender_{}", nonce % 5), // 5 different senders
        recipient: format!("receiver_{}", (nonce + 3) % 7), // 7 different receivers
        amount: 100 + (nonce * 10),
        nonce,
        gas_price,
        gas_limit: 21000 + (nonce * 100),
        data: if nonce % 3 == 0 {
            vec![0xde, 0xad, 0xbe, 0xef]
        } else {
            vec![]
        },
        signature: vec![0x01, 0x02, 0x03, 0x04],
        bls_signature: None,
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        status: TransactionStatus::Pending,
    }
}

/// Demonstrate monitoring capabilities
async fn demonstrate_monitoring_capabilities(
    state: &Arc<RwLock<State>>,
    mempool: &Arc<EnhancedMempool>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Demonstrating monitoring capabilities:");

    // Test mempool monitoring
    println!("\n💾 Mempool Information:");
    let stats = mempool.get_stats().await;
    println!("   Transaction count: {}", stats.total_transactions);
    println!("   Size: {} bytes", stats.size_bytes);

    // Calculate utilization percentage
    let utilization = (stats.size_bytes as f64 / (10 * 1024 * 1024) as f64) * 100.0;
    println!("   Utilization: {utilization:.2}%");

    // Determine health status based on utilization
    let health_status = match utilization {
        x if x < 50.0 => "Normal",
        x if x < 75.0 => "Busy",
        x if x < 90.0 => "Congested",
        _ => "Full",
    };
    println!("   Health status: {health_status}");

    // Test state monitoring
    println!("\n🗃️ State Information:");
    let state_guard = state.read().await;
    println!("   State initialized: ✅");
    drop(state_guard);

    // Simulate network monitoring
    println!("\n📡 Network Information:");
    let peer_count = 0; // No actual peers in this example
    let max_peers = 50;
    println!("   Connected peers: {peer_count}");
    println!("   Max peers: {max_peers}");

    let network_health = if peer_count == 0 {
        "Offline"
    } else if peer_count < 5 {
        "Critical"
    } else if peer_count < 15 {
        "Warning"
    } else {
        "Healthy"
    };
    println!("   Network health: {network_health}");

    // Simulate uptime monitoring
    println!("\n⏰ Uptime Information:");
    let start_time = std::time::SystemTime::now();
    tokio::time::sleep(Duration::from_millis(100)).await; // Simulate some runtime
    let uptime = start_time.elapsed().unwrap();
    println!("   Uptime: {:.2} seconds", uptime.as_secs_f64());
    println!("   Start time: {start_time:?}");

    // Overall health assessment
    println!("\n🌐 Overall Health Assessment:");
    let overall_health = match (network_health, health_status) {
        ("Healthy", "Normal") => "Excellent",
        ("Healthy", "Busy") | ("Warning", "Normal") => "Good",
        ("Warning", "Busy") | ("Critical", "Normal") => "Fair",
        ("Critical", "Busy") | ("Offline", "Normal") => "Poor",
        _ => "Critical",
    };
    println!("   Overall health: {overall_health}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_functionality() {
        let state = create_example_state().await.unwrap();
        let mempool = create_example_mempool().await;

        // Add test data
        populate_mempool(&mempool).await.unwrap();

        // Test monitoring
        demonstrate_monitoring_capabilities(&state, &mempool)
            .await
            .unwrap();

        // Verify mempool has transactions
        let stats = mempool.get_stats().await;
        assert!(stats.total_transactions > 0);
        assert!(stats.size_bytes > 0);
    }

    #[test]
    fn test_transaction_creation() {
        let tx = create_example_transaction(1, 20);
        assert_eq!(tx.tx_type, TransactionType::Transfer);
        assert_eq!(tx.nonce, 1);
        assert_eq!(tx.gas_price, 20);
        assert_eq!(tx.amount, 110); // 100 + (1 * 10)
        assert!(tx.sender.starts_with("sender_"));
        assert!(tx.recipient.starts_with("receiver_"));
    }
}
