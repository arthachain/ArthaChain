use blockchain_node::ai_engine::data_chunking::{DataChunkingAI, ChunkingConfig, CompressionType};
use blockchain_node::config::Config;
use std::time::Instant;
use std::collections::HashMap;

// Integration test for DataChunkingAI with larger files and real-world scenarios
#[cfg(test)]
mod tests {
    use super::*;
    
    // Helper function to create test data with patterns to simulate real files
    fn create_patterned_data(size: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(size);
        
        // Add repeating patterns (simulating structured data)
        let pattern1 = b"This is some structured data that might appear in a document multiple times. ";
        let pattern2 = b"Another pattern with numbers 12345678901234567890 and some repeating content. ";
        
        // Fill with patterns until we reach approximately target size
        while data.len() < size - 100 {
            if data.len() % 2 == 0 {
                data.extend_from_slice(pattern1);
            } else {
                data.extend_from_slice(pattern2);
            }
            
            // Add some random bytes occasionally
            if data.len() % 1000 == 0 {
                for i in 0..50 {
                    data.push((i % 256) as u8);
                }
            }
        }
        
        // Pad to exact size
        while data.len() < size {
            data.push(0);
        }
        
        data
    }
    
    // Helper function to calculate the hash of data
    fn calculate_hash(data: &[u8]) -> String {
        let hash = blake3::hash(data);
        hex::encode(hash.as_bytes())
    }
    
    // Custom setup for each test
    fn setup_custom_config(max_chunk_size: usize, min_chunk_size: usize, compression: CompressionType) -> (Config, ChunkingConfig) {
        let mut config = Config::default();
        
        let chunking_config = ChunkingConfig {
            max_chunk_size,
            min_chunk_size,
            chunking_threshold: 5 * 1024 * 1024, // 5 MB
            use_content_based_chunking: true,
            enable_deduplication: true,
            compress_chunks: true,
            encrypt_chunks: false,
            default_compression: compression,
            replication_factor: 3,
            chunk_size: 512, // default value
            overlap_size: 50, // default value
            max_chunks: 100, // default value
        };
        
        config.chunking_config = chunking_config.clone();
        
        (config, chunking_config)
    }
    
    #[test]
    fn test_large_file_chunking_performance() {
        // Create a 50MB file
        let data_size = 50 * 1024 * 1024;
        let data = create_patterned_data(data_size);
        
        // Setup with custom config - larger chunks
        let (config, _) = setup_custom_config(
            10 * 1024 * 1024,  // 10 MB max
            1 * 1024 * 1024,   // 1 MB min
            CompressionType::LZ4
        );
        
        let ai = DataChunkingAI::new(&config);
        
        // Measure chunking time
        let start_time = Instant::now();
        let chunks = ai.split_file(
            "performance_test_file",
            "performance_test.dat",
            &data,
            "application/octet-stream"
        ).unwrap();
        let chunking_time = start_time.elapsed();
        
        println!("Chunking time for 50MB file: {:?}", chunking_time);
        println!("Number of chunks created: {}", chunks.len());
        
        // Verify all chunks are within size limits
        for chunk in &chunks {
            assert!(chunk.data.len() <= 10 * 1024 * 1024, "Chunk too large");
            assert!(chunk.data.len() >= 1 * 1024 * 1024 || 
                   chunk.metadata.chunk_index == chunks.len() - 1, 
                   "Chunk too small (except last chunk)");
        }
        
        // Test reconstruction performance
        let file_id = "performance_test_file";
        let original_file_hash = calculate_hash(&data);
        
        let start_time = Instant::now();
        ai.start_file_reconstruction(file_id, "performance_test.dat", chunks.len(), &original_file_hash).unwrap();
        
        // Add all chunks
        for chunk in chunks {
            ai.add_chunk_to_reconstruction(chunk).unwrap();
        }
        
        // Reconstruct
        let reconstructed = ai.reconstruct_file(file_id).unwrap();
        let reconstruction_time = start_time.elapsed();
        
        println!("Reconstruction time for 50MB file: {:?}", reconstruction_time);
        
        // Verify reconstruction
        assert_eq!(reconstructed.len(), data.len());
        assert_eq!(calculate_hash(&reconstructed), original_file_hash);
    }
    
    #[test]
    fn test_compression_effectiveness() {
        // Create a highly compressible file (lots of repeated patterns)
        let data_size = 10 * 1024 * 1024; // 10 MB
        let mut data = Vec::with_capacity(data_size);
        
        // Add highly repetitive data
        for _ in 0..(data_size / 100) {
            data.extend_from_slice(b"The same text repeated over and over again to ensure high compression ratio. ");
        }
        
        // Test different compression types
        let compression_types = vec![
            CompressionType::None,
            CompressionType::GZip,
            CompressionType::ZStd,
            CompressionType::LZ4
        ];
        
        for compression_type in &compression_types {
            // Setup with specific compression
            let (config, _) = setup_custom_config(
                5 * 1024 * 1024,  // 5 MB max
                1 * 1024 * 1024,  // 1 MB min
                compression_type.clone()
            );
            
            let ai = DataChunkingAI::new(&config);
            
            // Measure with timing
            let start_time = Instant::now();
            let chunks = ai.split_file(
                &format!("compression_test_{:?}", compression_type),
                &format!("compression_test_{:?}.dat", compression_type),
                &data,
                "application/octet-stream"
            ).unwrap();
            let processing_time = start_time.elapsed();
            
            // In a real implementation, we would measure the compressed size
            // Here we just print the timing since our simulation doesn't actually compress
            println!("Processing time with {:?}: {:?}", compression_type, processing_time);
            println!("Number of chunks: {}", chunks.len());
        }
        
        // Verify we can correctly reconstruct with each compression type
        // Since our test implementation doesn't actually compress/decompress,
        // we just verify the logic flow works correctly
        let (config, _) = setup_custom_config(
            5 * 1024 * 1024,
            1 * 1024 * 1024,
            CompressionType::ZStd
        );
        
        let ai = DataChunkingAI::new(&config);
        let chunks = ai.split_file(
            "compression_reconstruction_test",
            "compression_reconstruction_test.dat",
            &data,
            "application/octet-stream"
        ).unwrap();
        
        let file_id = "compression_reconstruction_test";
        let original_file_hash = calculate_hash(&data);
        
        ai.start_file_reconstruction(file_id, "compression_reconstruction_test.dat", chunks.len(), &original_file_hash).unwrap();
        
        for chunk in chunks {
            ai.add_chunk_to_reconstruction(chunk).unwrap();
        }
        
        let reconstructed = ai.reconstruct_file(file_id).unwrap();
        
        // Verify reconstruction
        assert_eq!(reconstructed.len(), data.len());
        assert_eq!(calculate_hash(&reconstructed), original_file_hash);
    }
    
    #[test]
    fn test_partial_async_reconstruction() {
        // This test simulates a real-world scenario where chunks arrive out of order
        // and with delays, like in a distributed storage system
        
        let data_size = 20 * 1024 * 1024; // 20 MB
        let data = create_patterned_data(data_size);
        
        let (config, _) = setup_custom_config(
            4 * 1024 * 1024,  // 4 MB max
            1 * 1024 * 1024,  // 1 MB min
            CompressionType::LZ4
        );
        
        let ai = DataChunkingAI::new(&config);
        let file_id = "async_test";
        
        // Split the file
        let chunks = ai.split_file(
            file_id,
            "async_test.dat",
            &data,
            "application/octet-stream"
        ).unwrap();
        
        println!("Number of chunks for async test: {}", chunks.len());
        assert!(chunks.len() >= 5, "Need at least 5 chunks for this test");
        
        let original_file_hash = calculate_hash(&data);
        
        // Start reconstruction
        ai.start_file_reconstruction(file_id, "async_test.dat", chunks.len(), &original_file_hash).unwrap();
        
        // Deliberately add chunks in reverse order to simulate out-of-order arrival
        for i in (0..chunks.len()).rev() {
            let is_complete = ai.add_chunk_to_reconstruction(chunks[i].clone()).unwrap();
            
            if i == 0 {
                // Last chunk (first index) should complete it
                assert!(is_complete);
            } else {
                assert!(!is_complete);
            }
            
            // Check progress after each chunk
            let progress = ai.get_reconstruction_progress(file_id).unwrap();
            let expected_progress = (chunks.len() - i) as f32 / chunks.len() as f32;
            assert!((progress - expected_progress).abs() < 0.001, 
                    "Progress should be {}, got {}", expected_progress, progress);
        }
        
        // Reconstruct and verify
        let reconstructed = ai.reconstruct_file(file_id).unwrap();
        assert_eq!(reconstructed.len(), data.len());
        assert_eq!(calculate_hash(&reconstructed), original_file_hash);
    }
    
    #[test]
    fn test_distribution_plan_balancing() {
        // Test the distribution planning for chunks across nodes
        
        let data_size = 16 * 1024 * 1024; // Increased from 8 MB to 16 MB
        let data = create_patterned_data(data_size);
        
        let (config, _) = setup_custom_config(
            1 * 1024 * 1024,  // Decreased from 2 MB to 1 MB max
            256 * 1024,       // Decreased from 512 KB to 256 KB min
            CompressionType::None
        );
        
        let ai = DataChunkingAI::new(&config);
        
        // Create chunks
        let chunks = ai.split_file(
            "distribution_test",
            "distribution_test.dat",
            &data,
            "application/octet-stream"
        ).unwrap();
        
        println!("Created {} chunks for distribution test", chunks.len());
        assert!(chunks.len() >= 4, "Need at least 4 chunks for this test");
        
        // Test with different node counts
        let node_counts = [5, 10, 20];
        
        for &node_count in &node_counts {
            let plan = ai.generate_distribution_plan(&chunks, node_count).unwrap();
            
            // Should have an entry for each chunk
            assert_eq!(plan.len(), chunks.len());
            
            // Verify replication factor
            for (_, node_ids) in &plan {
                assert_eq!(node_ids.len(), 3); // Default replication factor
                
                // Verify all node IDs are unique
                let mut unique_nodes = node_ids.clone();
                unique_nodes.sort();
                unique_nodes.dedup();
                assert_eq!(unique_nodes.len(), node_ids.len());
            }
            
            // Analyze load balancing
            let mut node_load = HashMap::new();
            
            for (_, node_ids) in &plan {
                for node_id in node_ids {
                    *node_load.entry(node_id.clone()).or_insert(0) += 1;
                }
            }
            
            // Calculate load stats
            let total_assignments: usize = node_load.values().sum();
            let average_load = total_assignments as f32 / node_count as f32;
            let mut max_load = 0;
            let mut min_load = usize::MAX;
            
            for (_, load) in &node_load {
                max_load = std::cmp::max(max_load, *load);
                min_load = std::cmp::min(min_load, *load);
            }
            
            println!("Node count: {}, Avg load: {:.2}, Min: {}, Max: {}", 
                     node_count, average_load, min_load, max_load);
            
            // Verify reasonable load balancing
            if chunks.len() >= node_count {
                assert!(
                    max_load as f32 <= average_load * 2.5,
                    "Load too imbalanced. Max: {}, Avg: {:.2}", max_load, average_load
                );
            } else {
                // When there are fewer chunks than nodes, max load can be at most the replication factor
                assert!(
                    max_load <= 3,
                    "With fewer chunks than nodes, no node should have more than the replication factor (3) chunks. Max: {}",
                    max_load
                );
            }
        }
    }
} 