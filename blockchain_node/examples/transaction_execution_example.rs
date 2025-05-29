//! Transaction Execution Engine Example
//!
//! This example demonstrates how to use the transaction execution engine
//! to process transactions and update state.

use blockchain_node::config::Config;
use blockchain_node::execution::{transaction_engine::TransactionEngineConfig, TransactionEngine};
use blockchain_node::ledger::state::State;
use blockchain_node::ledger::transaction::{Transaction, TransactionType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logger
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    // Create state
    let config = Config::default();
    let state = std::sync::Arc::new(State::new(&config)?);

    // Initialize state with some accounts
    println!("Initializing state with test accounts");
    state.set_balance("alice", 1_000_000)?;
    state.set_balance("bob", 500_000)?;
    state.set_balance("charlie", 250_000)?;

    // Create transaction engine
    let engine_config = TransactionEngineConfig::default();
    let engine = TransactionEngine::new(state.clone(), engine_config)?;

    // Create some test transactions
    println!("Creating test transactions");

    // Alice sends 10,000 to Bob
    let mut tx1 = Transaction::new(
        TransactionType::Transfer,
        "alice".to_string(),
        "bob".to_string(),
        10_000,
        0,
        1,
        21_000,
        vec![],
    );
    tx1.signature = vec![1, 2, 3, 4]; // Dummy signature

    // Bob sends 5,000 to Charlie
    let mut tx2 = Transaction::new(
        TransactionType::Transfer,
        "bob".to_string(),
        "charlie".to_string(),
        5_000,
        0,
        1,
        21_000,
        vec![],
    );
    tx2.signature = vec![1, 2, 3, 4]; // Dummy signature

    // Charlie sends 1,000 to Alice
    let mut tx3 = Transaction::new(
        TransactionType::Transfer,
        "charlie".to_string(),
        "alice".to_string(),
        1_000,
        0,
        1,
        21_000,
        vec![],
    );
    tx3.signature = vec![1, 2, 3, 4]; // Dummy signature

    let mut transactions = vec![tx1, tx2, tx3];

    // Display initial state
    println!("Initial state:");
    println!("  Alice balance: {}", state.get_balance("alice")?);
    println!("  Bob balance: {}", state.get_balance("bob")?);
    println!("  Charlie balance: {}", state.get_balance("charlie")?);

    // Process transactions
    println!("Processing transactions...");
    let results = engine.process_transactions(&mut transactions).await?;

    // Display results
    println!("Transaction results:");
    for (i, result) in results.iter().enumerate() {
        println!("  Transaction {}: {:?}", i, result);
    }

    // Display final state
    println!("Final state:");
    println!("  Alice balance: {}", state.get_balance("alice")?);
    println!("  Bob balance: {}", state.get_balance("bob")?);
    println!("  Charlie balance: {}", state.get_balance("charlie")?);

    // Calculate and display changes
    let alice_change = state.get_balance("alice")? - 1_000_000;
    let bob_change = state.get_balance("bob")? - 500_000;
    let charlie_change = state.get_balance("charlie")? - 250_000;

    println!("Changes:");
    println!("  Alice: {:+}", alice_change);
    println!("  Bob: {:+}", bob_change);
    println!("  Charlie: {:+}", charlie_change);

    Ok(())
}
