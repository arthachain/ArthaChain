use crate::config::Config;
use crate::execution::{ExecutionResult, TransactionEngine, TransactionEngineConfig};
use crate::ledger::state::State;
use crate::ledger::transaction::{Transaction, TransactionStatus, TransactionType};
use crate::wasm::WasmConfig;
use std::sync::Arc;

#[tokio::test]
async fn test_basic_transfer() {
    // Create state
    let config = Config::default();
    let state = Arc::new(State::new(&config).unwrap());
    
    // Initialize accounts
    state.set_balance("sender", 1_000_000).unwrap();
    state.set_balance("recipient", 500_000).unwrap();
    
    // Create engine
    let engine_config = TransactionEngineConfig::default();
    let engine = TransactionEngine::new(state.clone(), engine_config).unwrap();
    
    // Create transaction
    let mut tx = Transaction::new(
        TransactionType::Transfer,
        "sender".to_string(),
        "recipient".to_string(),
        100_000,
        0,
        1,
        21_000,
        vec![],
        vec![1, 2, 3, 4], // Dummy signature
    );
    
    // Process transaction
    let result = engine.process_transaction(&mut tx).await.unwrap();
    
    // Verify success
    match result {
        ExecutionResult::Success => {
            assert_eq!(tx.status, TransactionStatus::Success);
            assert_eq!(state.get_balance("sender").unwrap(), 1_000_000 - 100_000 - 21_000);
            assert_eq!(state.get_balance("recipient").unwrap(), 500_000 + 100_000);
            assert_eq!(state.get_nonce("sender").unwrap(), 1);
        }
        _ => panic!("Transaction failed: {:?}", result),
    }
}

#[tokio::test]
async fn test_insufficient_balance() {
    // Create state
    let config = Config::default();
    let state = Arc::new(State::new(&config).unwrap());
    
    // Initialize accounts with limited funds
    state.set_balance("poor_sender", 10_000).unwrap();
    state.set_balance("recipient", 0).unwrap();
    
    // Create engine
    let engine_config = TransactionEngineConfig::default();
    let engine = TransactionEngine::new(state.clone(), engine_config).unwrap();
    
    // Create transaction with amount larger than available balance
    let mut tx = Transaction::new(
        TransactionType::Transfer,
        "poor_sender".to_string(),
        "recipient".to_string(),
        1_000_000, // More than sender has
        0,
        1,
        21_000,
        vec![],
        vec![1, 2, 3, 4], // Dummy signature
    );
    
    // Process transaction
    let result = engine.process_transaction(&mut tx).await.unwrap();
    
    // Verify failure
    match result {
        ExecutionResult::InsufficientBalance => {
            // Check state remained unchanged
            assert_eq!(state.get_balance("poor_sender").unwrap(), 10_000);
            assert_eq!(state.get_balance("recipient").unwrap(), 0);
            assert_eq!(state.get_nonce("poor_sender").unwrap(), 0);
            
            match tx.status {
                TransactionStatus::Failed(ref reason) => {
                    assert!(reason.contains("Insufficient"));
                }
                _ => panic!("Expected Failed status, got {:?}", tx.status),
            }
        }
        _ => panic!("Expected InsufficientBalance, got {:?}", result),
    }
}

#[tokio::test]
async fn test_nonce_validation() {
    // Create state
    let config = Config::default();
    let state = Arc::new(State::new(&config).unwrap());
    
    // Initialize accounts
    state.set_balance("sender", 1_000_000).unwrap();
    state.set_nonce("sender", 5).unwrap(); // Current nonce is 5
    
    // Create engine
    let engine_config = TransactionEngineConfig::default();
    let engine = TransactionEngine::new(state.clone(), engine_config).unwrap();
    
    // 1. Test with nonce too low
    let mut tx1 = Transaction::new(
        TransactionType::Transfer,
        "sender".to_string(),
        "recipient".to_string(),
        1_000,
        4, // Lower than current nonce
        1,
        21_000,
        vec![],
        vec![1, 2, 3, 4], // Dummy signature
    );
    
    let result1 = engine.process_transaction(&mut tx1).await.unwrap();
    
    match result1 {
        ExecutionResult::InvalidNonce => {
            assert_eq!(state.get_nonce("sender").unwrap(), 5); // Unchanged
        }
        _ => panic!("Expected InvalidNonce, got {:?}", result1),
    }
    
    // 2. Test with nonce too high
    let mut tx2 = Transaction::new(
        TransactionType::Transfer,
        "sender".to_string(),
        "recipient".to_string(),
        1_000,
        6, // Higher than current nonce
        1,
        21_000,
        vec![],
        vec![1, 2, 3, 4], // Dummy signature
    );
    
    let result2 = engine.process_transaction(&mut tx2).await.unwrap();
    
    match result2 {
        ExecutionResult::InvalidNonce => {
            assert_eq!(state.get_nonce("sender").unwrap(), 5); // Unchanged
        }
        _ => panic!("Expected InvalidNonce, got {:?}", result2),
    }
    
    // 3. Test with correct nonce
    let mut tx3 = Transaction::new(
        TransactionType::Transfer,
        "sender".to_string(),
        "recipient".to_string(),
        1_000,
        5, // Correct nonce
        1,
        21_000,
        vec![],
        vec![1, 2, 3, 4], // Dummy signature
    );
    
    let result3 = engine.process_transaction(&mut tx3).await.unwrap();
    
    match result3 {
        ExecutionResult::Success => {
            assert_eq!(state.get_nonce("sender").unwrap(), 6); // Incremented
        }
        _ => panic!("Expected Success, got {:?}", result3),
    }
}

#[tokio::test]
async fn test_batch_processing() {
    // Create state
    let config = Config::default();
    let state = Arc::new(State::new(&config).unwrap());
    
    // Initialize accounts
    state.set_balance("sender", 1_000_000).unwrap();
    
    // Create engine
    let engine_config = TransactionEngineConfig::default();
    let engine = TransactionEngine::new(state.clone(), engine_config).unwrap();
    
    // Create a batch of transactions
    let mut transactions = vec![
        // First transaction (nonce 0)
        Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient1".to_string(),
            10_000,
            0,
            1,
            21_000,
            vec![],
            vec![1, 2, 3, 4], // Dummy signature
        ),
        // Second transaction (nonce 1)
        Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient2".to_string(),
            20_000,
            1,
            1,
            21_000,
            vec![],
            vec![1, 2, 3, 4], // Dummy signature
        ),
        // Third transaction (nonce 2)
        Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient3".to_string(),
            30_000,
            2,
            1,
            21_000,
            vec![],
            vec![1, 2, 3, 4], // Dummy signature
        ),
    ];
    
    // Process transactions
    let results = engine.process_transactions(&mut transactions).await.unwrap();
    
    // Verify all succeeded
    for (i, result) in results.iter().enumerate() {
        match result {
            ExecutionResult::Success => {
                assert_eq!(transactions[i].status, TransactionStatus::Success);
            }
            _ => panic!("Transaction {} failed: {:?}", i, result),
        }
    }
    
    // Verify final state
    assert_eq!(state.get_nonce("sender").unwrap(), 3);
    assert_eq!(
        state.get_balance("sender").unwrap(),
        1_000_000 - 10_000 - 20_000 - 30_000 - (21_000 * 3)
    );
    assert_eq!(state.get_balance("recipient1").unwrap(), 10_000);
    assert_eq!(state.get_balance("recipient2").unwrap(), 20_000);
    assert_eq!(state.get_balance("recipient3").unwrap(), 30_000);
}

#[tokio::test]
async fn test_apply_transactions_to_block() {
    // Create state
    let config = Config::default();
    let state = Arc::new(State::new(&config).unwrap());
    
    // Initialize accounts
    state.set_balance("sender", 1_000_000).unwrap();
    
    // Create engine
    let engine_config = TransactionEngineConfig::default();
    let engine = TransactionEngine::new(state.clone(), engine_config).unwrap();
    
    // Create a batch of transactions
    let mut transactions = vec![
        Transaction::new(
            TransactionType::Transfer,
            "sender".to_string(),
            "recipient".to_string(),
            50_000,
            0,
            1,
            21_000,
            vec![],
            vec![1, 2, 3, 4], // Dummy signature
        ),
    ];
    
    // Apply to block
    engine.apply_transactions_to_block(&mut transactions, 10).await.unwrap();
    
    // Verify block height updated
    assert_eq!(state.get_height().unwrap(), 10);
    
    // Verify transaction applied
    assert_eq!(state.get_balance("sender").unwrap(), 1_000_000 - 50_000 - 21_000);
    assert_eq!(state.get_balance("recipient").unwrap(), 50_000);
}

// Add more tests for other transaction types and edge cases as needed 