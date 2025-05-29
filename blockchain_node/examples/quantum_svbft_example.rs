use anyhow::Result;
use blockchain_node::config::Config;
use blockchain_node::consensus::{quantum_svbft::QuantumSVBFTConfig, QuantumSVBFTConsensus};
use blockchain_node::ledger::block::{
    Block, BlockBody, BlockHeader, ConsensusInfo, ConsensusStatus, SocialVerificationData,
};
use blockchain_node::ledger::state::State;
use blockchain_node::ledger::transaction::Transaction;
use blockchain_node::types::Hash;
use blockchain_node::utils::crypto::generate_quantum_resistant_keypair;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, RwLock};

/// Example demonstrating quantum-resistant SVBFT consensus
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::init();

    println!("Starting Quantum-Resistant SVBFT Example");

    // Create configuration with shorter timeouts for demo purposes
    let qsvbft_config = QuantumSVBFTConfig {
        base_timeout_ms: 500,
        view_change_timeout_ms: 2000,
        quantum_resistance_level: 2, // 0-3, higher is more secure but slower
        parallel_validation: true,
        ..QuantumSVBFTConfig::default()
    };

    // Create node configuration
    let config = Config::default();

    // Create shared state
    let state = Arc::new(RwLock::new(State::new(&config)?));

    // Create channels for consensus messages and blocks
    let (message_sender, message_receiver) = mpsc::channel(100);
    let (block_sender, block_receiver) = mpsc::channel(100);
    let (shutdown_sender, shutdown_receiver) = broadcast::channel(1);

    // Generate quantum-resistant keys for our node
    let (public_key, _private_key) = generate_quantum_resistant_keypair(None)?;
    let node_id = hex::encode(&public_key[0..8]); // Use first 8 bytes as node ID

    println!("Node ID: {}", node_id);

    // Create consensus instance
    let mut consensus = QuantumSVBFTConsensus::new(
        config,
        state.clone(),
        message_sender.clone(),
        message_receiver,
        block_receiver,
        shutdown_receiver,
        node_id.clone(),
        Some(qsvbft_config),
    )
    .await?;

    // Start consensus
    let consensus_handle = consensus.start().await?;
    println!("Consensus started");

    // Show initial state
    let view = consensus.get_current_view().await;
    let leader = consensus.get_current_leader().await;
    let phase = consensus.get_current_phase().await;

    println!("Initial view: {}", view);
    println!("Initial leader: {:?}", leader);
    println!("Initial phase: {:?}", phase);

    // Wait a moment
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Create a sample block and propose it if we're the leader
    let is_leader = consensus
        .get_current_leader()
        .await
        .map(|l| l == node_id)
        .unwrap_or(false);

    if is_leader {
        println!("We are the leader, proposing a block");

        // Create sample transactions
        let txs = vec![
            Transaction::new_test_transaction(1),
            Transaction::new_test_transaction(2),
        ];

        // Create a block
        let block = Block {
            header: BlockHeader::new(
                Hash::default(), // previous_hash
                Hash::default(), // merkle_root
                1,               // height
                1000,            // difficulty
                node_id.clone(), // proposer_id
                0,               // shard_id
            ),
            body: BlockBody { transactions: txs },
            consensus: ConsensusInfo {
                status: ConsensusStatus::Proposed,
                validator_signatures: Vec::new(),
                status_timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                sv_data: SocialVerificationData {
                    sv_score: 0.0,
                    compute_score: 0.0,
                    network_score: 0.0,
                    storage_score: 0.0,
                    engagement_score: 0.0,
                    ai_security_score: 0.0,
                    reputation_history: Vec::new(),
                },
                shard_id: 0,
                cross_shard_refs: Vec::new(),
            },
            is_genesis: false,
        };

        // Send block to consensus
        block_sender.send(block).await?;

        println!("Block proposed");
    } else {
        println!("We are not the leader, can't propose a block");
    }

    // Wait for consensus to process the block
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    // Trigger a view change
    if !is_leader {
        println!("Initiating view change from non-leader node");

        // Create view change request message
        let view_change_msg =
            blockchain_node::consensus::quantum_svbft::ConsensusMessage::ViewChangeRequest {
                current_view: view,
                new_view: view + 1,
                node_id: node_id.clone(),
                signature: vec![1, 2, 3, 4], // Simplified for example
                reason: blockchain_node::consensus::quantum_svbft::ViewChangeReason::LeaderTimeout,
            };

        // Send view change request
        message_sender.send(view_change_msg).await?;

        println!("View change requested");
    }

    // Wait for view change to complete
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    // Show final state
    let new_view = consensus.get_current_view().await;
    let new_leader = consensus.get_current_leader().await;
    let new_phase = consensus.get_current_phase().await;

    println!("Final view: {}", new_view);
    println!("Final leader: {:?}", new_leader);
    println!("Final phase: {:?}", new_phase);

    // Shutdown consensus
    println!("Shutting down consensus");
    shutdown_sender.send(())?;

    // Wait for consensus to shutdown
    consensus_handle.await?;

    println!("Quantum-Resistant SVBFT Example completed");

    Ok(())
}
