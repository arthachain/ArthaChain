use blockchain_node::ai_engine::data_chunking::{CompressionType, DataChunkingAI};
use blockchain_node::config::Config;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;

fn generate_sample_file(size_mb: usize) -> Vec<u8> {
    let data_size = size_mb * 1024 * 1024; // Convert MB to bytes
    vec![42u8; data_size]
}

fn bench_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_chunking_split_file");
    let sizes = vec![1, 5, 10, 20, 50]; // File sizes in MB

    for size in sizes {
        let data = generate_sample_file(size);
        let mut config = Config::default();
        config.chunking_config.max_chunk_size = 10 * 1024 * 1024;
        config.chunking_config.min_chunk_size = 1024 * 1024;
        config.chunking_config.default_compression = CompressionType::None;
        let ai = Arc::new(DataChunkingAI::new(&config));

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let _ = ai.split_file(
                    &format!("bench_file_{size}"),
                    &format!("bench_file_{size}.dat"),
                    &data,
                    "application/octet-stream",
                );
            });
        });
    }
    group.finish();
}

fn bench_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_chunking_reconstruction");
    let sizes = vec![1, 5, 10, 20, 50]; // File sizes in MB

    for size in sizes {
        let data = generate_sample_file(size);
        let mut config = Config::default();
        config.chunking_config.max_chunk_size = 10 * 1024 * 1024;
        config.chunking_config.min_chunk_size = 1024 * 1024;
        config.chunking_config.default_compression = CompressionType::None;

        // Pre-generate chunks outside the benchmark loop
        let ai = Arc::new(DataChunkingAI::new(&config));
        let file_id = format!("bench_reconstruction_{size}");
        let file_name = format!("bench_reconstruction_{size}.dat");
        let chunks = ai
            .split_file(&file_id, &file_name, &data, "application/octet-stream")
            .unwrap();

        let original_hash = chunks[0].metadata.original_file_hash.clone();
        let total_chunks = chunks.len();
        let chunks = Arc::new(chunks);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter_batched(
                // Setup: Create a new DataChunkingAI for each iteration with a unique ID
                || {
                    let ai = DataChunkingAI::new(&config);
                    // Generate a unique file_id for this iteration to avoid collisions
                    let iter_file_id = format!(
                        "{}_{}",
                        file_id,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos()
                    );
                    (ai, iter_file_id)
                },
                |(ai, iter_file_id)| {
                    // Initialize reconstruction state
                    ai.start_file_reconstruction(
                        &iter_file_id,
                        &file_name,
                        total_chunks,
                        &original_hash,
                    )
                    .unwrap();

                    // Add chunks to reconstruction in order
                    for chunk in chunks.iter() {
                        let mut chunk_clone = chunk.clone();
                        // Update the file_id in each chunk to match our reconstruction ID
                        chunk_clone.metadata.file_id = iter_file_id.clone();
                        ai.add_chunk_to_reconstruction(chunk_clone).unwrap();
                    }

                    // Reconstruct the file
                    let reconstructed = ai.reconstruct_file(&iter_file_id).unwrap();
                    assert_eq!(reconstructed.len(), data.len());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_chunking, bench_reconstruction);
criterion_main!(benches);
