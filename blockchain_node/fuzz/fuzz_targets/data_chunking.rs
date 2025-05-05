#![no_main]
use libfuzzer_sys::fuzz_target;
use blockchain_node::ai_engine::data_chunking::{DataChunkingAI, CompressionType};
use blockchain_node::config::Config;

fuzz_target!(|data: &[u8]| {
    // Skip empty inputs
    if data.is_empty() {
        return;
    }

    let mut config = Config::default();
    config.chunking_config.max_chunk_size = 10 * 1024 * 1024; // 10MB
    config.chunking_config.min_chunk_size = 1 * 1024;         // 1KB
    config.chunking_config.default_compression = CompressionType::None;
    
    let ai = DataChunkingAI::new(&config);
    let file_id = "fuzz_test";

    // Try to split the fuzzed data into chunks
    if let Ok(chunks) = ai.split_file(
        file_id,
        "fuzz_test.dat",
        data,
        "application/octet-stream"
    ) {
        // Skip if no chunks were created
        if chunks.is_empty() {
            return;
        }
        
        // Get the original file hash from the first chunk's metadata
        let original_file_hash = chunks[0].metadata.original_file_hash.clone();
        
        // If splitting succeeded, try reconstruction
        // Create a reconstruction from chunks
        if let Ok(_) = ai.start_file_reconstruction(file_id, "fuzz_test.dat", chunks.len(), &original_file_hash) {
            // Add each chunk to the reconstruction
            for chunk in chunks {
                let _ = ai.add_chunk_to_reconstruction(chunk);
            }
            
            // Now reconstruct the file
            if let Ok(reconstructed) = ai.reconstruct_file(file_id) {
                // Verify the reconstruction matches the original
                assert_eq!(data, reconstructed.as_slice(), "Reconstructed data doesn't match original");
            }
        }
    }
}); 