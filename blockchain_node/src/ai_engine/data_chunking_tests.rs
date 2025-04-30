use super::data_chunking::{DataChunkingAI, DataChunk, ChunkMetadata, ReconstructionStatus, CompressionType};
use crate::config::Config;
use blake3;
use hex;

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create test data
    fn create_test_data(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i % 256) as u8).collect()
    }

    // Helper function to create a test config
    fn create_test_config() -> Config {
        Config::default()
    }
    
    // Helper function to calculate the hash of data similar to DataChunkingAI
    fn calculate_hash(data: &[u8]) -> String {
        let hash = blake3::hash(data);
        hex::encode(hash.as_bytes())
    }

    #[test]
    fn test_split_file() {
        let config = create_test_config();
        let ai = DataChunkingAI::new(&config);
        let data = create_test_data(1500); // Data size of 1500 bytes
        
        let file_id = "test_file_id";
        let filename = "test_file.dat";
        let mime_type = "application/octet-stream";
        
        let chunks = ai.split_file(file_id, filename, &data, mime_type).unwrap();
        
        // Verify we got the expected number of chunks
        assert!(chunks.len() > 0);
        
        // Verify chunk metadata
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.metadata.file_id, file_id);
            assert_eq!(chunk.metadata.chunk_index, i);
            assert_eq!(chunk.metadata.total_chunks, chunks.len());
            assert_eq!(chunk.metadata.original_filename, filename);
            assert_eq!(chunk.metadata.mime_type, mime_type);
            
            // Verify chunk has data
            assert!(!chunk.data.is_empty());
        }
        
        // Verify combined size equals original
        let total_data_size: usize = chunks.iter().map(|c| c.data.len()).sum();
        assert_eq!(total_data_size, data.len());
    }
    
    #[test]
    fn test_file_reconstruction() {
        let config = create_test_config();
        let ai = DataChunkingAI::new(&config);
        let data = create_test_data(1200); // Creates multiple chunks
        
        let file_id = "test_recon_id";
        let filename = "test_recon.dat";
        let mime_type = "application/octet-stream";
        
        // Split the file into chunks
        let chunks = ai.split_file(file_id, filename, &data, mime_type).unwrap();
        
        // Get original file hash
        let original_file_hash = calculate_hash(&data);
        
        // Start reconstruction
        ai.start_file_reconstruction(file_id, filename, chunks.len(), &original_file_hash).unwrap();
        
        // Add all chunks
        for chunk in chunks {
            let is_complete = ai.add_chunk_to_reconstruction(chunk).unwrap();
            if is_complete {
                // The last chunk should complete the reconstruction
                let reconstructed = ai.reconstruct_file(file_id).unwrap();
                assert_eq!(reconstructed, data);
            }
        }
        
        // Verify final status
        let status = ai.get_reconstruction_status(file_id).unwrap();
        assert_eq!(status, ReconstructionStatus::Complete);
    }
    
    #[test]
    fn test_partial_reconstruction() {
        let config = create_test_config();
        let ai = DataChunkingAI::new(&config);
        
        // Create a test data significantly larger than default chunking threshold
        let data = create_test_data(15 * 1024 * 1024); // 15MB, should create multiple chunks
        
        let file_id = "test_partial_id";
        let filename = "test_partial.dat";
        let mime_type = "application/octet-stream";
        
        // Split the file into chunks
        let chunks = ai.split_file(file_id, filename, &data, mime_type).unwrap();
        
        println!("Number of chunks created: {}", chunks.len());
        assert!(chunks.len() >= 2, "Expected at least 2 chunks, but got {} chunks", chunks.len()); // Ensure we have at least 2 chunks for this test
        
        // Get original file hash
        let original_file_hash = calculate_hash(&data);
        
        // Start reconstruction
        ai.start_file_reconstruction(file_id, filename, chunks.len(), &original_file_hash).unwrap();
        
        // Add only the first chunk
        let is_complete = ai.add_chunk_to_reconstruction(chunks[0].clone()).unwrap();
        assert!(!is_complete);
        
        // Verify status is InProgress
        let status = ai.get_reconstruction_status(file_id).unwrap();
        assert_eq!(status, ReconstructionStatus::InProgress);
        
        // Attempting reconstruction should fail as not all chunks present
        let result = ai.reconstruct_file(file_id);
        assert!(result.is_err());
        
        // Add the remaining chunks
        for i in 1..chunks.len() {
            let is_complete = ai.add_chunk_to_reconstruction(chunks[i].clone()).unwrap();
            
            if i == chunks.len() - 1 {
                // Last chunk should complete it
                assert!(is_complete);
            } else {
                assert!(!is_complete);
            }
        }
        
        // Now reconstruction should succeed
        let reconstructed = ai.reconstruct_file(file_id).unwrap();
        assert_eq!(reconstructed, data);
    }
    
    #[test]
    fn test_small_file_no_chunking() {
        let config = create_test_config();
        let ai = DataChunkingAI::new(&config);
        
        // Create a small file that should not be chunked (below chunking threshold)
        let small_data = create_test_data(1 * 1024 * 1024); // 1MB - below default chunking threshold
        
        let file_id = "test_small_file";
        let filename = "small_file.dat";
        let mime_type = "application/octet-stream";
        
        // Split the file
        let chunks = ai.split_file(file_id, filename, &small_data, mime_type).unwrap();
        
        // Should only be a single chunk
        assert_eq!(chunks.len(), 1, "Small file should only produce 1 chunk");
        
        // Verify metadata
        assert_eq!(chunks[0].metadata.file_id, file_id);
        assert_eq!(chunks[0].metadata.total_chunks, 1);
        
        // Data should match original
        assert_eq!(chunks[0].data.len(), small_data.len());
    }
    
    #[test]
    fn test_chunking_patterns() {
        let config = create_test_config();
        let ai = DataChunkingAI::new(&config);
        
        // Create some test data with repeating patterns
        let mut data = Vec::new();
        for _ in 0..20 {
            data.extend_from_slice(&[1, 2, 3, 4, 5]);
            data.extend_from_slice(&[6, 7, 8, 9, 10]);
            data.extend_from_slice(&[11, 12, 13, 14, 15]);
        }
        
        // Add some random data to ensure it's large enough
        data.extend_from_slice(&create_test_data(12 * 1024 * 1024));
        
        let file_id = "test_chunking_patterns";
        let filename = "patterns.dat";
        let mime_type = "application/octet-stream";
        
        // Get chunks using the public API
        let chunks = ai.split_file(file_id, filename, &data, mime_type).unwrap();
        
        // Should have created multiple chunks
        assert!(chunks.len() >= 2, "Expected multiple chunks");
        
        // Verify the chunks cover the entire data
        let total_size: usize = chunks.iter().map(|c| c.data.len()).sum();
        assert_eq!(total_size, data.len(), "Chunks should cover the entire file data");
        
        // Verify chunks are in order and non-overlapping
        let _prev_end = 0;
        let mut all_data = Vec::new();
        
        for chunk in chunks {
            all_data.extend_from_slice(&chunk.data);
        }
        
        assert_eq!(all_data, data, "Reconstructed data should match original");
    }
    
    #[test]
    fn test_distribution_plan() {
        let config = create_test_config();
        let ai = DataChunkingAI::new(&config);
        
        let data = create_test_data(2 * 1024 * 1024); // 2MB
        let file_id = "test_distribution";
        let filename = "distribution.dat";
        let mime_type = "application/octet-stream";
        
        // Create chunks
        let chunks = ai.split_file(file_id, filename, &data, mime_type).unwrap();
        
        // Test with 5 available nodes
        let plan = ai.generate_distribution_plan(&chunks, 5).unwrap();
        
        // Should have an entry for each chunk
        assert_eq!(plan.len(), chunks.len());
        
        // Each chunk should be assigned to multiple nodes based on replication factor
        // Default replication factor in ChunkingConfig is 3
        let expected_replication = 3;
        
        for (_, node_list) in &plan {
            assert_eq!(node_list.len(), expected_replication, 
                       "Each chunk should be replicated on {} nodes", expected_replication);
            
            // Each node ID should be unique
            let mut unique_nodes = node_list.clone();
            unique_nodes.sort();
            unique_nodes.dedup();
            assert_eq!(unique_nodes.len(), node_list.len(), "Node IDs should be unique");
        }
        
        // Test with 0 nodes - should return error
        let error_plan = ai.generate_distribution_plan(&chunks, 0);
        assert!(error_plan.is_err());
    }
    
    #[test]
    fn test_reconstruction_error_handling() {
        let config = create_test_config();
        let ai = DataChunkingAI::new(&config);
        
        // Test getting status for non-existent reconstruction
        let status_result = ai.get_reconstruction_status("nonexistent_file");
        assert!(status_result.is_err());
        
        // Test getting progress for non-existent reconstruction
        let progress_result = ai.get_reconstruction_progress("nonexistent_file");
        assert!(progress_result.is_err());
        
        // Test adding chunk to non-existent reconstruction
        let data = [1, 2, 3, 4, 5];
        let metadata = ChunkMetadata {
            file_id: "nonexistent_file".to_string(),
            chunk_index: 0,
            total_chunks: 1,
            original_filename: "test.dat".to_string(),
            mime_type: "application/octet-stream".to_string(),
            created_at: std::time::SystemTime::now(),
            original_file_hash: "abcdef".to_string(),
        };
        
        let chunk = DataChunk {
            id: "test-chunk-0".to_string(),
            data: data.to_vec(),
            size: 5,
            hash: "abcdef".to_string(),
            metadata,
            compression: CompressionType::None,
            encryption: None,
        };
        
        let add_result = ai.add_chunk_to_reconstruction(chunk);
        assert!(add_result.is_err());
    }
    
    #[test]
    fn test_chunking_large_file() {
        let config = create_test_config();
        let ai = DataChunkingAI::new(&config);
        
        // Test with a large file (considerably larger than chunking threshold)
        let large_data = create_test_data(25 * 1024 * 1024); // 25MB
        let file_id = "test_large_file";
        let filename = "large_file.dat";
        let mime_type = "application/octet-stream";
        
        // Split into chunks
        let chunks = ai.split_file(file_id, filename, &large_data, mime_type).unwrap();
        
        // Should have multiple chunks
        assert!(chunks.len() > 1, "Expected multiple chunks for large file");
        
        // Each chunk should be within reasonable size limits
        // Assuming default ChunkingConfig values
        let max_chunk_size = 5 * 1024 * 1024; // 5 MB from default config
        
        for chunk in &chunks {
            assert!(chunk.data.len() <= max_chunk_size, 
                   "Chunk size {} exceeds maximum {}", chunk.data.len(), max_chunk_size);
        }
        
        // Test reconstruction
        let original_file_hash = calculate_hash(&large_data);
        ai.start_file_reconstruction(file_id, filename, chunks.len(), &original_file_hash).unwrap();
        
        // Add all chunks
        for chunk in chunks {
            ai.add_chunk_to_reconstruction(chunk).unwrap();
        }
        
        // Reconstruct the file
        let reconstructed = ai.reconstruct_file(file_id).unwrap();
        assert_eq!(reconstructed, large_data);
    }
    
    #[test]
    fn test_reconstruction_progress_tracking() {
        let config = create_test_config();
        let ai = DataChunkingAI::new(&config);
        
        // Create a larger file to ensure it gets split into at least 3 chunks
        let data = create_test_data(20 * 1024 * 1024); // 20MB - should create multiple chunks
        
        let file_id = "test_progress";
        let filename = "progress.dat";
        let mime_type = "application/octet-stream";
        
        // Create chunks
        let chunks = ai.split_file(file_id, filename, &data, mime_type).unwrap();
        
        // Ensure we have enough chunks for the test
        println!("Created {} chunks for progress tracking test", chunks.len());
        assert!(chunks.len() >= 3, "Need at least 3 chunks for this test");
        
        let original_file_hash = calculate_hash(&data);
        
        // Start reconstruction
        ai.start_file_reconstruction(file_id, filename, chunks.len(), &original_file_hash).unwrap();
        
        // Check initial progress
        let initial_progress = ai.get_reconstruction_progress(file_id).unwrap();
        assert_eq!(initial_progress, 0.0);
        
        // Add first chunk
        ai.add_chunk_to_reconstruction(chunks[0].clone()).unwrap();
        
        // Check progress after first chunk
        let progress_1 = ai.get_reconstruction_progress(file_id).unwrap();
        let expected_1 = 1.0 / (chunks.len() as f32);
        assert!((progress_1 - expected_1).abs() < 0.001, 
                "Progress should be approximately {}, got {}", expected_1, progress_1);
        
        // Add second chunk
        ai.add_chunk_to_reconstruction(chunks[1].clone()).unwrap();
        
        // Check progress after second chunk
        let progress_2 = ai.get_reconstruction_progress(file_id).unwrap();
        let expected_2 = 2.0 / (chunks.len() as f32);
        assert!((progress_2 - expected_2).abs() < 0.001, 
                "Progress should be approximately {}, got {}", expected_2, progress_2);
        
        // Add all remaining chunks
        for i in 2..chunks.len() {
            ai.add_chunk_to_reconstruction(chunks[i].clone()).unwrap();
        }
        
        // Check final progress
        let final_progress = ai.get_reconstruction_progress(file_id).unwrap();
        assert_eq!(final_progress, 1.0, "Final progress should be 1.0");
    }
} 