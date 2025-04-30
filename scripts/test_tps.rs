fn main() {
    // With 1 miner
    let block_time_1 = 7.5;
    let batch_size_1 = 1000;
    let tps_1 = batch_size_1 as f32 / block_time_1;
    
    // With 4 miners
    let block_time_4 = 1.875;
    let batch_size_4 = 4000;
    let tps_4 = batch_size_4 as f32 / block_time_4;
    
    // With 50 miners
    let block_time_50 = 0.15;
    let batch_size_50 = 50000;
    let tps_50 = batch_size_50 as f32 / block_time_50;
    
    println!("TPS with 1 miner: {:.2} (batch_size={} / block_time={}s)", 
             tps_1, batch_size_1, block_time_1);
    println!("TPS with 4 miners: {:.2} (batch_size={} / block_time={}s)", 
             tps_4, batch_size_4, block_time_4);
    println!("TPS with 50 miners: {:.2} (batch_size={} / block_time={}s)", 
             tps_50, batch_size_50, block_time_50);
    
    // Calculate confirmation time (assuming 1 round after block creation)
    let confirmation_time_1 = block_time_1 * 1.0; // Single confirmation
    let confirmation_time_4 = block_time_4 * 1.0; // Single confirmation
    let confirmation_time_50 = block_time_50 * 1.0; // Single confirmation
    
    println!("Estimated confirmation time with 1 miner: {:.2}s", confirmation_time_1);
    println!("Estimated confirmation time with 4 miners: {:.2}s", confirmation_time_4);
    println!("Estimated confirmation time with 50 miners: {:.2}s", confirmation_time_50);
    
    // Throughput scaling factor
    let throughput_multiplier = 2.0;
    let miner_count_1 = 1;
    let miner_count_4 = 4;
    let miner_count_50 = 50;
    
    let scaling_factor_1 = (miner_count_1 as f32 * throughput_multiplier).max(1.0);
    let scaling_factor_4 = (miner_count_4 as f32 * throughput_multiplier).max(1.0);
    let scaling_factor_50 = (miner_count_50 as f32 * throughput_multiplier).max(1.0);
    
    println!("Scaling factor with 1 miner: {:.2}", scaling_factor_1);
    println!("Scaling factor with 4 miners: {:.2}", scaling_factor_4);
    println!("Scaling factor with 50 miners: {:.2}", scaling_factor_50);
    
    // Theoretical maximum TPS (with max batch size)
    let max_batch_size = 10000;
    let min_block_time = 0.5; // Minimum block time allowed
    let max_tps = max_batch_size as f32 / min_block_time;
    
    println!("Theoretical maximum TPS: {:.2} (batch_size={} / min_block_time={}s)",
             max_tps, max_batch_size, min_block_time);
} 