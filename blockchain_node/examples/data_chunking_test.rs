use blockchain_node::ai_engine::data_chunking::{DataChunkingAI, CompressionType};
use blockchain_node::config::Config;
use std::time::{Instant, Duration};

fn generate_sample_file(size_mb: usize) -> Vec<u8> {
    let data_size = size_mb * 1024 * 1024; // Convert MB to bytes
    vec![42u8; data_size]
}

// Run a test multiple times and return the average duration
fn run_benchmark<F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut() -> Duration,
{
    let mut total = Duration::new(0, 0);
    for _ in 0..iterations {
        total += f();
    }
    total / iterations as u32
}

fn main() {
    println!("Running Data Chunking Test");
    println!("=========================\n");
    
    let file_sizes = vec![1, 5, 10, 20, 50]; // File sizes in MB
    let iterations = 3; // Number of iterations for each test
    
    println!("| File Size (MB) | Chunks | Split Time (avg) | Reconstruction Time (avg) |");
    println!("|---------------|--------|-----------------|--------------------------|");
    
    for size in file_sizes {
        let data = generate_sample_file(size);
        let mut config = Config::default();
        config.chunking_config.max_chunk_size = 10 * 1024 * 1024;
        config.chunking_config.min_chunk_size = 1 * 1024 * 1024;
        config.chunking_config.default_compression = CompressionType::None;
        
        // Prepare the test data once
        let ai = DataChunkingAI::new(&config);
        let file_id = format!("test_file_{}", size);
        let file_name = format!("test_file_{}.dat", size);
        
        // Split the file into chunks
        let split_time = run_benchmark(iterations, || {
            let start = Instant::now();
            let _ = ai.split_file(
                &file_id,
                &file_name,
                &data,
                "application/octet-stream"
            ).unwrap();
            start.elapsed()
        });
        
        // Do one real split to get the chunks for reconstruction
        let chunks = ai.split_file(
            &file_id,
            &file_name,
            &data,
            "application/octet-stream"
        ).unwrap();
        
        let total_chunks = chunks.len();
        let original_hash = chunks[0].metadata.original_file_hash.clone();
        
        // Test reconstruction
        let reconstruction_time = run_benchmark(iterations, || {
            let start = Instant::now();
            
            // Create a new AI instance for reconstruction
            let reconstruction_ai = DataChunkingAI::new(&config);
            let iter_file_id = format!("{}_{}", file_id, Instant::now().elapsed().as_nanos());
            
            // Initialize reconstruction
            reconstruction_ai.start_file_reconstruction(
                &iter_file_id,
                &file_name,
                total_chunks,
                &original_hash
            ).unwrap();
            
            // Add all chunks to reconstruction
            for chunk in &chunks {
                let mut chunk_clone = chunk.clone();
                chunk_clone.metadata.file_id = iter_file_id.clone();
                reconstruction_ai.add_chunk_to_reconstruction(chunk_clone).unwrap();
            }
            
            // Reconstruct the file
            let reconstructed = reconstruction_ai.reconstruct_file(&iter_file_id).unwrap();
            
            // Verify reconstruction
            assert_eq!(reconstructed.len(), data.len(), "Reconstructed file size mismatch");
            assert_eq!(reconstructed, data, "Reconstructed data does not match original");
            
            start.elapsed()
        });
        
        println!("| {:13} | {:6} | {:16?} | {:24?} |", 
                 size, 
                 total_chunks, 
                 split_time, 
                 reconstruction_time);
    }
    
    println!("\nAll tests passed successfully!");
} 