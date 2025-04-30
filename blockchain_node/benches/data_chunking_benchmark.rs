use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use blockchain_node::ai_engine::data_chunking::{DataChunkingAI, ChunkingConfig, CompressionType};
use blockchain_node::config::Config;
use std::time::Duration;
use blake3;
use hex;

// Helper function to create test data of a specific size
fn create_test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

// Helper function to calculate hash using the same method as DataChunkingAI
fn calculate_hash(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    hex::encode(hash.as_bytes())
}

// Benchmark file splitting at different sizes
fn bench_file_splitting(c: &mut Criterion) {
    let config = Config::default();
    let ai = DataChunkingAI::new(&config);

    let mut group = c.benchmark_group("data_chunking_split_file");
    group.measurement_time(Duration::from_secs(10));
    
    // Test with different file sizes
    for size_mb in [1, 5, 10, 20, 50].iter() {
        let size = size_mb * 1024 * 1024; // Convert MB to bytes
        let data = create_test_data(size);
        
        group.bench_with_input(BenchmarkId::from_parameter(size_mb), &size, |b, _| {
            b.iter(|| {
                ai.split_file(
                    black_box("benchmark_file_id"),
                    black_box("benchmark_file.dat"),
                    black_box(&data),
                    black_box("application/octet-stream"),
                )
            });
        });
    }
    
    group.finish();
}

// Benchmark file reconstruction
fn bench_file_reconstruction(c: &mut Criterion) {
    let config = Config::default();
    let ai = DataChunkingAI::new(&config);
    
    let mut group = c.benchmark_group("data_chunking_reconstruction");
    group.measurement_time(Duration::from_secs(10));
    
    // Test with different file sizes
    for size_mb in [1, 5, 10, 20].iter() {
        let size = size_mb * 1024 * 1024; // Convert MB to bytes
        let data = create_test_data(size);
        let file_id = format!("bench_recon_{}", size_mb);
        let filename = format!("bench_{}.dat", size_mb);
        
        // First split the file into chunks
        let chunks = ai.split_file(&file_id, &filename, &data, "application/octet-stream").unwrap();
        
        // Calculate the hash using the same method as DataChunkingAI
        let hash = calculate_hash(&data);
        
        // Setup reconstruction state before benchmarking
        ai.start_file_reconstruction(&file_id, &filename, chunks.len(), &hash).unwrap();
        
        group.bench_with_input(BenchmarkId::from_parameter(size_mb), &size, |b, _| {
            b.iter_with_setup(
                // Setup for each iteration - we need to restart reconstruction
                || {
                    // Clear previous state and start new reconstruction
                    let _ = ai.start_file_reconstruction(&file_id, &filename, chunks.len(), &hash);
                    chunks.clone()
                },
                // Benchmark the reconstruction process
                |chunks_clone| {
                    // Add all chunks
                    for chunk in chunks_clone {
                        black_box(ai.add_chunk_to_reconstruction(chunk).unwrap());
                    }
                    
                    // Reconstruct the file
                    black_box(ai.reconstruct_file(&file_id).unwrap());
                }
            );
        });
    }
    
    group.finish();
}

// Benchmark compression performance with different types
fn bench_compression(c: &mut Criterion) {
    let mut config = Config::default();
    let mut group = c.benchmark_group("data_chunking_compression");
    group.measurement_time(Duration::from_secs(10));
    
    // Create test data with some patterns to make compression effective
    let mut data = Vec::with_capacity(10 * 1024 * 1024);
    for _ in 0..1000 {
        // Add some repeating patterns
        data.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        data.extend_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        data.extend_from_slice(&[100, 90, 80, 70, 60, 50, 40, 30, 20, 10]);
    }
    
    // Extend to reach desired size
    while data.len() < 10 * 1024 * 1024 {
        data.push(0);
    }
    
    // Test with different compression types
    for compression_type in [CompressionType::None, CompressionType::GZip, CompressionType::LZ4].iter() {
        // Create a custom chunking config with the specific compression type
        let chunking_config = ChunkingConfig {
            min_chunk_size: 64 * 1024,         // 64 KB
            max_chunk_size: 5 * 1024 * 1024,   // 5 MB
            chunking_threshold: 1024 * 1024 * 10, // 10 MB
            use_content_based_chunking: true,
            enable_deduplication: true,
            compress_chunks: true,
            encrypt_chunks: false,
            default_compression: compression_type.clone(),
            replication_factor: 3,
            chunk_size: 512,
            overlap_size: 50,
            max_chunks: 100,
        };
        
        // Update the config with the custom chunking config
        config.chunking_config = chunking_config;
        
        // Create a new AI instance with the updated config
        let ai = DataChunkingAI::new(&config);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", compression_type)),
            compression_type,
            |b, _| {
                b.iter(|| {
                    ai.split_file(
                        black_box("benchmark_compression_id"),
                        black_box("benchmark_compression.dat"),
                        black_box(&data),
                        black_box("application/octet-stream"),
                    )
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_file_splitting,
    bench_file_reconstruction,
    bench_compression
);
criterion_main!(benches); 