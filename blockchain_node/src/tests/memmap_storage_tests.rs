#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;
    use tokio::runtime::Runtime;

    use crate::storage::{CompressionAlgorithm, Hash, MemMapOptions, Storage, StorageInit};

    #[test]
    fn test_memmap_storage_throughput() {
        let rt = Runtime::new().unwrap();

        // Create a temporary directory for the test
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path();

        // Configure memmap options
        let options = MemMapOptions {
            compression: CompressionAlgorithm::LZ4,
            sync_on_flush: false,
            file_growth_size: 64 * 1024 * 1024, // 64MB
            max_file_size: 1024 * 1024 * 1024,  // 1GB
            block_size: 4096,
            max_open_files: 1000,
        };

        // Initialize storage
        let memmap_storage = rt.block_on(async {
            crate::storage::memmap_storage::MemMapStorage::new(path, options)
                .await
                .expect("Failed to create memmap storage")
        });

        // Test parameters
        let data_sizes = [1024, 10 * 1024, 100 * 1024]; // 1KB, 10KB, 100KB
        let num_operations = 10000;

        for &size in &data_sizes {
            println!("\nTesting with data size: {} bytes", size);

            // Generate random data
            let data: Vec<Vec<u8>> = (0..num_operations)
                .map(|_| {
                    let mut rng = thread_rng();
                    (0..size).map(|_| rng.gen::<u8>()).collect()
                })
                .collect();

            // Write test
            let start = Instant::now();
            let hashes = rt.block_on(async {
                let mut hashes = Vec::with_capacity(num_operations);

                // Store data with batch processing
                let mut batch = Vec::with_capacity(100);

                for chunk in data.chunks(100) {
                    batch.clear();
                    for item in chunk {
                        batch.push(item.as_slice());
                    }

                    // Process batch
                    for item in batch.iter() {
                        let hash = memmap_storage
                            .store(item)
                            .await
                            .expect("Failed to store data");
                        hashes.push(hash);
                    }
                }

                hashes
            });

            let write_time = start.elapsed();
            let write_throughput =
                (num_operations * size) as f64 / write_time.as_secs_f64() / (1024.0 * 1024.0);
            println!(
                "Write throughput: {:.2} MB/s ({} operations in {:.2?})",
                write_throughput, num_operations, write_time
            );

            // Read test
            let start = Instant::now();
            rt.block_on(async {
                for hash in &hashes {
                    let retrieved = memmap_storage
                        .retrieve(hash)
                        .await
                        .expect("Failed to retrieve data");
                    assert!(retrieved.is_some(), "Data not found for hash");
                }
            });

            let read_time = start.elapsed();
            let read_throughput =
                (num_operations * size) as f64 / read_time.as_secs_f64() / (1024.0 * 1024.0);
            println!(
                "Read throughput: {:.2} MB/s ({} operations in {:.2?})",
                read_throughput, num_operations, read_time
            );

            // Verify minimum performance requirements
            assert!(
                write_throughput > 50.0,
                "Write throughput below minimum requirement: {:.2} MB/s",
                write_throughput
            );
            assert!(
                read_throughput > 100.0,
                "Read throughput below minimum requirement: {:.2} MB/s",
                read_throughput
            );
        }

        // Close storage
        rt.block_on(async {
            memmap_storage
                .close()
                .await
                .expect("Failed to close storage");
        });
    }
}
