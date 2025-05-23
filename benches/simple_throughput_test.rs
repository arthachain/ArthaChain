use std::thread;
/**
 * Simple Throughput Test for Blockchain
 *
 * This benchmark focuses solely on transaction processing speed without
 * complex dependencies.
 */
use std::time::{Duration, Instant};

// Create a simple benchmark that processes transactions in parallel
fn main() {
    // Configuration
    let tx_sizes = [100, 1000, 10000]; // Small, medium, large
    let tx_counts = [10000, 100000, 500000]; // Different batch sizes
    let thread_counts = [1, 4, 8, 16, 32]; // Different parallelism levels

    println!("Starting High-TPS Throughput Benchmark");
    println!("======================================");

    for &tx_size in &tx_sizes {
        for &tx_count in &tx_counts {
            println!(
                "\nTesting with {tx_count} transactions of {tx_size} bytes each:"
            );

            // Generate test transactions
            let transactions = generate_test_transactions(tx_count, tx_size);

            for &thread_count in &thread_counts {
                // Run single-threaded test first for baseline
                if thread_count == 1 {
                    let start = Instant::now();
                    let processed = process_transactions_single(&transactions);
                    let duration = start.elapsed();
                    let tps = calculate_tps(processed, duration);

                    println!(
                        "  Single-threaded: {:.2} TPS ({} ms)",
                        tps,
                        duration.as_millis()
                    );
                }

                // Run multi-threaded test
                let start = Instant::now();
                let processed = process_transactions_parallel(&transactions, thread_count);
                let duration = start.elapsed();
                let tps = calculate_tps(processed, duration);

                println!(
                    "  {} threads: {:.2} TPS ({} ms)",
                    thread_count,
                    tps,
                    duration.as_millis()
                );
            }
        }
    }
}

// Generate test transactions with random data
fn generate_test_transactions(count: usize, size: usize) -> Vec<Vec<u8>> {
    let mut transactions = Vec::with_capacity(count);

    for i in 0..count {
        // Create transaction with random data
        let mut tx = Vec::with_capacity(size);

        // Add transaction header (64 bytes)
        tx.extend_from_slice(&i.to_be_bytes()); // Nonce (8 bytes)
        tx.extend_from_slice(&[0u8; 24]); // Sender (24 bytes)
        tx.extend_from_slice(&[1u8; 24]); // Receiver (24 bytes)
        tx.extend_from_slice(&[0u8; 8]); // Amount (8 bytes)

        // Fill the rest with random-like data
        let remaining = size.saturating_sub(64);
        if remaining > 0 {
            tx.extend((0..remaining).map(|j| ((i + j) % 256) as u8));
        }

        transactions.push(tx);
    }

    transactions
}

// Process transactions in a single thread
fn process_transactions_single(transactions: &[Vec<u8>]) -> usize {
    let mut processed = 0;

    for tx in transactions {
        // Simple processing - hash the transaction
        let hash = hash_transaction(tx);
        // Verify the hash is valid (first byte is less than 250)
        if hash[0] < 250 {
            processed += 1;
        }
    }

    processed
}

// Process transactions in parallel
fn process_transactions_parallel(transactions: &[Vec<u8>], threads: usize) -> usize {
    let chunk_size = transactions.len().div_ceil(threads);
    let chunks: Vec<&[Vec<u8>]> = transactions.chunks(chunk_size).collect();

    let mut handles = vec![];

    for chunk in chunks {
        // Clone the chunk data to send to the thread
        let chunk_data: Vec<Vec<u8>> = chunk.iter().cloned().collect();

        let handle = thread::spawn(move || {
            let mut processed = 0;

            for tx in &chunk_data {
                // Simple processing - hash the transaction
                let hash = hash_transaction(tx);
                // Verify the hash is valid (first byte is less than 250)
                if hash[0] < 250 {
                    processed += 1;
                }
            }

            processed
        });

        handles.push(handle);
    }

    // Collect results
    let mut total_processed = 0;
    for handle in handles {
        total_processed += handle.join().unwrap_or(0);
    }

    total_processed
}

// Calculate TPS
fn calculate_tps(processed: usize, duration: Duration) -> f64 {
    processed as f64 / duration.as_secs_f64()
}

// Simple hash function
fn hash_transaction(data: &[u8]) -> [u8; 32] {
    let mut hash = [0u8; 32];

    // Very simple hash function for benchmarking
    for (i, &byte) in data.iter().enumerate() {
        hash[i % 32] = hash[i % 32].wrapping_add(byte);
    }

    hash
}
