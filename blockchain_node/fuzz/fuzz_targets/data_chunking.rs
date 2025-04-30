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

    // Try to split the fuzzed data into chunks
    if let Ok(chunks) = ai.split_file(
        "fuzz_test",
        "fuzz_test.dat",
        data,
        "application/octet-stream"
    ) {
        // If splitting succeeded, try reconstruction
        if let Ok(reconstructed) = ai.reconstruct_file(&chunks) {
            // Verify the reconstruction matches the original
            assert_eq!(data, reconstructed.as_slice(), "Reconstructed data doesn't match original");
        }
    }
}); 