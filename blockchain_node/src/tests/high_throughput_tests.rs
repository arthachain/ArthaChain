#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};
    use tokio::runtime::Runtime;
    use tokio::sync::mpsc;
    use tokio::task;

    #[test]
    fn test_parallel_processing_capacity() {
        // Set up a test measuring raw parallel processing speed
        let num_threads = num_cpus::get();
        let num_transactions = 1_000_000;
        let tx_size = 1024; // 1KB transactions

        println!(
            "Starting parallel processing test with {} threads",
            num_threads
        );
        println!(
            "Processing {} transactions of {} bytes each",
            num_transactions, tx_size
        );

        let rt = Runtime::new().unwrap();

        let start = Instant::now();

        // Create transactions
        let transactions: Vec<Vec<u8>> = (0..num_transactions)
            .map(|i| {
                let mut data = vec![0u8; tx_size];
                // Add some data to make each transaction unique
                let bytes = i.to_le_bytes();
                data[0..bytes.len()].copy_from_slice(&bytes);
                data
            })
            .collect();

        // Process transactions in parallel
        let counter = Arc::new(AtomicUsize::new(0));

        // Create channels
        let (tx, mut rx) = mpsc::channel(num_transactions);

        // Process transactions using worker threads
        rt.block_on(async {
            // Create workers
            for chunk in transactions.chunks(num_transactions / num_threads) {
                let chunk = chunk.to_vec();
                let tx = tx.clone();
                let counter_clone = counter.clone();

                task::spawn(async move {
                    for tx_data in chunk {
                        // Simulate processing
                        let hash = blake3::hash(&tx_data);
                        counter_clone.fetch_add(1, Ordering::SeqCst);

                        // Send confirmation
                        let _ = tx.send(hash.as_bytes().to_vec()).await;
                    }
                });
            }

            // Drop sender to allow receiver to complete
            drop(tx);

            // Collect responses
            let mut received = 0;
            while let Some(_) = rx.recv().await {
                received += 1;
                if received % 100000 == 0 {
                    println!("Processed {} transactions", received);
                }
            }
        });

        let elapsed = start.elapsed();
        let tps = num_transactions as f64 / elapsed.as_secs_f64();

        println!(
            "Processed {} transactions in {:.2?}",
            num_transactions, elapsed
        );
        println!("Throughput: {:.2} TPS", tps);

        // Assert that we can meet minimum throughput requirements
        assert!(
            tps > 50000.0,
            "Throughput below minimum requirement: {:.2} TPS",
            tps
        );
    }
}
