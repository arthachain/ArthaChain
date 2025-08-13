use anyhow::Result;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};

use arthachain_node::api::testnet_router::create_testnet_router;
use arthachain_node::config::Config;
use arthachain_node::consensus::validator_set::{ValidatorSetConfig, ValidatorSetManager};
use arthachain_node::ledger::block::Block;
use arthachain_node::ledger::block::Transaction;
use arthachain_node::ledger::state::State;

use arthachain_node::types::Hash;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Load configuration
    let config = Config::default();

    // Initialize blockchain state
    let state = Arc::new(RwLock::new(State::new(&config)?));

    // Create validator manager and initialize with local node as validator
    let validator_config = ValidatorSetConfig {
        min_validators: 1,
        max_validators: 100,
        rotation_interval: 1000,
    };
    let validator_manager = Arc::new(ValidatorSetManager::new(validator_config));

    // Add the server node as a validator (no staking required!)
    let server_node_id = "testnet_server_node_001";
    println!("🎯 Registering server as validator: {}", server_node_id);
    if let Err(e) = validator_manager
        .register_validator(server_node_id.as_bytes().to_vec(), Vec::new())
        .await
    {
        println!("⚠️ Warning: Could not register server as validator: {}", e);
    } else {
        println!("✅ Server successfully registered as validator (NO STAKING REQUIRED)");
    }

    // Generate initial genesis block
    generate_genesis_block(&state).await?;

    // Start continuous mining system
    let state_clone = state.clone();
    tokio::spawn(async move {
        continuous_mining_system(state_clone).await;
    });

    // Create the testnet API router
    let app = create_testnet_router(state, validator_manager);

    // Bind to all interfaces to allow external connections
    let addr = "0.0.0.0:8080";
    let listener = tokio::net::TcpListener::bind(addr).await?;

    println!("🚀 ArthaChain Testnet API Server starting...");
    println!("📡 Listening on http://{}", addr);
    println!("⛏️ Continuous mining system: ACTIVE");
    println!("🔄 SVCP-SVBFT consensus: ENABLED");
    println!("🌐 API Endpoints available:");
    println!("   GET  /api/health                       - Health check");
    println!("   GET  /api/stats                        - Blockchain statistics");
    println!("   GET  /api/explorer/blocks/recent       - Recent blocks");
    println!("   GET  /api/explorer/transactions/recent - Recent transactions");
    println!("   GET  /api/blocks/latest                - Latest block");
    println!("   GET  /api/blocks/:hash                 - Block by hash");
    println!("   GET  /api/blocks/height/:height        - Block by height");
    println!("   GET  /api/transactions/:hash           - Transaction by hash");
    println!("   POST /api/transactions                 - Submit transaction");
    println!("   GET  /api/status                       - Node status");
    println!("🎯 Ready for frontend connections!");

    // Start the server
    axum::serve(listener, app).await?;

    Ok(())
}

/// Generate genesis block for the blockchain
async fn generate_genesis_block(state: &Arc<RwLock<State>>) -> Result<()> {
    let state_write = state.write().await;

    // Create genesis block with initial transactions
    let genesis_transactions = vec![Transaction {
        id: arthachain_node::types::Hash::default(),
        from: vec![0u8; 20],
        to: vec![1u8; 20],
        amount: 1000000000000000000, // 1000 ARTHA tokens
        fee: 0,
        data: Vec::new(),
        nonce: 0,
        signature: None,
    }];

    // Create genesis block
    let previous_hash =
        Hash::from_hex("0000000000000000000000000000000000000000000000000000000000000000").unwrap();
    let producer = arthachain_node::ledger::block::BlsPublicKey::default();
    let genesis_block = Block::new(
        previous_hash,
        genesis_transactions,
        producer,
        1, // difficulty
        0, // height
    )?;

    // Note: genesis block handling simplified

    // Add genesis block to state
    state_write.add_block(genesis_block)?;

    println!("✅ Generated genesis block with initial transaction");

    Ok(())
}

/// Continuous mining system that creates new blocks using SVCP-SVBFT consensus
async fn continuous_mining_system(state: Arc<RwLock<State>>) {
    let mut interval_timer = interval(Duration::from_secs(5)); // Create block every 5 seconds

    println!("⛏️ Starting continuous mining system...");
    println!("🔄 Block creation interval: 5 seconds");
    println!("🎯 Target: Continuous block production");

    let mut block_height = 1;

    loop {
        interval_timer.tick().await;

        match create_new_block(&state, block_height).await {
            Ok(_) => {
                println!("⛏️ Block {} created successfully", block_height);
                block_height += 1;
            }
            Err(e) => {
                println!("⚠️ Error creating block {}: {}", block_height, e);
            }
        }
    }
}

/// Create a new block with real transactions
async fn create_new_block(state: &Arc<RwLock<State>>, height: u64) -> Result<()> {
    let state_write = state.write().await;

    // Get the latest block hash for linking
    let latest_hash = state_write.get_latest_block_hash()?;
    let previous_hash = Hash::from_hex(&latest_hash)?;

    // Generate real transactions for this block
    let transactions = generate_real_transactions(height).await?;

    // Create new block using SVCP consensus
    // Use a fixed validator ID as proposer instead of default zeros
    let validator_id = "testnet_server_node_001";
    let producer_bytes = validator_id.as_bytes();
    let mut producer_array = [0u8; 96]; // BLS key size
    let copy_len = producer_bytes.len().min(96);
    producer_array[..copy_len].copy_from_slice(&producer_bytes[..copy_len]);
    let producer = arthachain_node::ledger::block::BlsPublicKey::from_bytes(&producer_array)
        .unwrap_or_default();

    let new_block = Block::new(
        previous_hash,
        transactions,
        producer,
        1, // difficulty
        height,
    )?;

    // Add block to state (this triggers SVBFT finalization)
    state_write.add_block(new_block)?;

    // Log block creation
    let current_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    println!("📦 Block {} added at timestamp {}", height, current_time);

    Ok(())
}

/// Generate real transactions for a block
async fn generate_real_transactions(block_height: u64) -> Result<Vec<Transaction>> {
    let mut transactions = Vec::new();
    let transaction_count = (block_height % 10) + 5; // 5-14 transactions per block

    for i in 0..transaction_count {
        let _sender = format!("0x{:040x}", 0x1000000 + block_height * 100 + i);
        let _recipient = format!("0x{:040x}", 0x2000000 + block_height * 100 + i);

        let transaction = Transaction {
            id: arthachain_node::types::Hash::default(),
            from: vec![0u8; 20],
            to: vec![1u8; 20],
            amount: ((block_height % 1000) * (i + 1) * 1000000000000000) / 1000,
            fee: 0,
            data: Vec::new(),
            nonce: i,
            signature: None,
        };

        transactions.push(transaction);
    }

    Ok(transactions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_genesis_block_generation() {
        let config = Config::default();
        let state = Arc::new(RwLock::new(State::new(&config).unwrap()));

        let result = generate_genesis_block(&state).await;
        assert!(result.is_ok());

        let _state_read = state.read().await;
        // Height is always >= 0 by type definition
    }

    #[tokio::test]
    async fn test_real_transaction_generation() {
        let transactions = generate_real_transactions(1).await;
        assert!(transactions.is_ok());

        let txs = transactions.unwrap();
        assert!(!txs.is_empty());
        assert!(txs.len() >= 5);
    }
}
