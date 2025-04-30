#![no_main]
use libfuzzer_sys::fuzz_target;
use blockchain_node::ai_engine::data_chunking::{DataChunkingAI, CompressionType};
use blockchain_node::config::Config;

fuzz_target!(|data: &[u8]| {
    let (config, _) = {
        let mut config = Config::default();
        config.chunking_config.max_chunk_size = 1024 * 1024;
        config.chunking_config.min_chunk_size = 256 * 1024;
        config.chunking_config.default_compression = CompressionType::None;
        (config, config.chunking_config.clone())
    };
    let ai = DataChunkingAI::new(&config);

    // Fuzz split_file
    let _ = ai.split_file(
        "fuzz_file",
        "fuzz_file.dat",
        data,
        "application/octet-stream"
    );
}); 