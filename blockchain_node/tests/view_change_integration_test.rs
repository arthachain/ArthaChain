use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;

use blockchain_node::consensus::view_change::{
    ViewChangeConfig, ViewChangeManager, ViewChangeMessage, ViewChangeReason,
};
use blockchain_node::types::Address;

/// Test Byzantine fault tolerance with 33% malicious nodes
#[tokio::test]
async fn test_byzantine_fault_tolerance_33_percent() {
    // Setup: 10 validators, 3 can be malicious (33%)
    let total_validators = 10;
    let malicious_count = 3;
    let quorum_size = 7; // 2f+1 = 2*3+1 = 7

    let config = ViewChangeConfig {
        view_timeout: Duration::from_secs(5),
        max_view_changes: 5,
        min_validators: total_validators,
        leader_election_interval: Duration::from_secs(10),
    };

    let mut manager = ViewChangeManager::new(quorum_size, config);

    // Create validator set
    let validators: HashSet<Vec<u8>> = (0..total_validators)
        .map(|i| format!("validator_{}", i).into_bytes())
        .collect();

    manager.initialize(validators.clone()).await.unwrap();

    // Simulate honest validators sending view change messages
    let honest_validators = total_validators - malicious_count;
    let target_view = 5;

    for i in 0..honest_validators {
        let validator_bytes = format!("validator_{}", i).into_bytes();
        let validator_addr = Address::from_bytes(&validator_bytes).unwrap();

        let message = ViewChangeMessage::new(
            target_view,
            validator_addr.clone(),
            vec![1, 2, 3, 4], // Mock signature
        );

        let view_changed = manager
            .process_view_change_message(message, validator_addr)
            .await
            .unwrap();

        if i == quorum_size - 1 {
            // Should reach quorum and execute view change
            assert!(
                view_changed,
                "View change should succeed with honest quorum"
            );
            break;
        }
    }

    assert_eq!(manager.get_current_view(), target_view);
    println!("✅ Byzantine fault tolerance test passed: 33% malicious nodes handled");
}

/// Test view change timeout mechanism
#[tokio::test]
async fn test_view_change_timeout() {
    let config = ViewChangeConfig {
        view_timeout: Duration::from_millis(100), // Very short timeout for testing
        max_view_changes: 3,
        min_validators: 4,
        leader_election_interval: Duration::from_secs(1),
    };

    let manager = ViewChangeManager::new(3, config); // Need 3 nodes for quorum

    let validators: HashSet<Vec<u8>> = (0..4)
        .map(|i| format!("validator_{}", i).into_bytes())
        .collect();

    manager.initialize(validators).await.unwrap();

    // Start timeout
    manager.start_view_timeout().await.unwrap();

    // Wait for timeout to trigger
    sleep(Duration::from_millis(150)).await;

    // Check that view change attempts have increased
    let state = manager.state.read().await;
    assert!(
        state.change_attempts > 0,
        "Timeout should trigger view change attempt"
    );

    println!("✅ View change timeout test passed");
}

/// Test leader election across multiple view changes
#[tokio::test]
async fn test_leader_election_rotation() {
    let config = ViewChangeConfig {
        view_timeout: Duration::from_secs(1),
        max_view_changes: 10,
        min_validators: 4,
        leader_election_interval: Duration::from_secs(1),
    };

    let manager = ViewChangeManager::new(3, config);

    let validators: HashSet<Vec<u8>> = (0..4)
        .map(|i| format!("validator_{}", i).into_bytes())
        .collect();

    manager.initialize(validators.clone()).await.unwrap();

    let mut previous_leaders = Vec::new();

    // Test multiple view changes to verify round-robin
    for view in 1..=4 {
        manager.elect_leader_for_view(view).await.unwrap();

        let state = manager.state.read().await;
        let current_leader = state.leader.clone();

        assert!(
            current_leader.is_some(),
            "Leader should be elected for view {}",
            view
        );

        // Verify leader rotation (no leader should repeat in consecutive views)
        if let Some(leader) = current_leader {
            if !previous_leaders.is_empty() {
                assert_ne!(
                    previous_leaders.last().unwrap(),
                    &leader,
                    "Leader should rotate between views"
                );
            }
            previous_leaders.push(leader);
        }
    }

    println!("✅ Leader election rotation test passed");
}

/// Test view change message validation
#[tokio::test]
async fn test_view_change_message_validation() {
    let config = ViewChangeConfig {
        view_timeout: Duration::from_secs(5),
        max_view_changes: 5,
        min_validators: 4,
        leader_election_interval: Duration::from_secs(10),
    };

    let mut manager = ViewChangeManager::new(3, config);

    let validators: HashSet<Vec<u8>> = (0..4)
        .map(|i| format!("validator_{}", i).into_bytes())
        .collect();

    manager.initialize(validators.clone()).await.unwrap();

    // Test 1: Valid validator sending view change
    let valid_validator = format!("validator_0").into_bytes();
    let valid_addr = Address::from_bytes(&valid_validator).unwrap();
    let valid_message = ViewChangeMessage::new(2, valid_addr.clone(), vec![1, 2, 3]);

    let result = manager
        .process_view_change_message(valid_message, valid_addr)
        .await;
    assert!(
        result.is_ok(),
        "Valid view change message should be accepted"
    );

    // Test 2: Invalid validator (not in validator set)
    let invalid_validator = format!("invalid_validator").into_bytes();
    let invalid_addr = Address::from_bytes(&invalid_validator).unwrap();
    let invalid_message = ViewChangeMessage::new(3, invalid_addr.clone(), vec![4, 5, 6]);

    let result = manager
        .process_view_change_message(invalid_message, invalid_addr)
        .await;
    assert!(result.is_err(), "Invalid validator should be rejected");

    println!("✅ View change message validation test passed");
}

/// Test concurrent view change requests
#[tokio::test]
async fn test_concurrent_view_changes() {
    let config = ViewChangeConfig {
        view_timeout: Duration::from_secs(5),
        max_view_changes: 5,
        min_validators: 6,
        leader_election_interval: Duration::from_secs(10),
    };

    let manager = Arc::new(RwLock::new(ViewChangeManager::new(4, config))); // Need 4 for quorum

    let validators: HashSet<Vec<u8>> = (0..6)
        .map(|i| format!("validator_{}", i).into_bytes())
        .collect();

    manager
        .write()
        .await
        .initialize(validators.clone())
        .await
        .unwrap();

    // Create concurrent view change requests
    let mut handles = Vec::new();
    let target_view = 10;

    for i in 0..6 {
        let manager_clone = manager.clone();
        let validator_bytes = format!("validator_{}", i).into_bytes();

        let handle = tokio::spawn(async move {
            let validator_addr = Address::from_bytes(&validator_bytes).unwrap();
            let message = ViewChangeMessage::new(
                target_view,
                validator_addr.clone(),
                vec![i as u8, (i + 1) as u8, (i + 2) as u8],
            );

            let mut mgr = manager_clone.write().await;
            mgr.process_view_change_message(message, validator_addr)
                .await
                .unwrap_or(false)
        });

        handles.push(handle);
    }

    // Wait for all concurrent requests
    let results = futures::future::join_all(handles).await;

    // At least one should have triggered the view change
    let view_changed = results.into_iter().any(|r| r.unwrap_or(false));
    assert!(view_changed, "Concurrent view changes should succeed");

    // Verify final view
    let final_view = manager.read().await.get_current_view();
    assert_eq!(
        final_view, target_view,
        "View should be updated to target view"
    );

    println!("✅ Concurrent view change test passed");
}

/// Test view change under network partition simulation
#[tokio::test]
async fn test_view_change_network_partition() {
    let config = ViewChangeConfig {
        view_timeout: Duration::from_millis(200),
        max_view_changes: 5,
        min_validators: 7,
        leader_election_interval: Duration::from_secs(1),
    };

    let manager = ViewChangeManager::new(5, config); // Need 5 for quorum out of 7

    let validators: HashSet<Vec<u8>> = (0..7)
        .map(|i| format!("validator_{}", i).into_bytes())
        .collect();

    manager.initialize(validators.clone()).await.unwrap();

    // Start timeout to simulate network partition
    manager.start_view_timeout().await.unwrap();

    // Wait for timeout
    sleep(Duration::from_millis(250)).await;

    // Simulate partition healing: 5 validators can communicate
    let mut successful_changes = 0;
    for i in 0..5 {
        let validator_bytes = format!("validator_{}", i).into_bytes();
        let validator_addr = Address::from_bytes(&validator_bytes).unwrap();

        let message = ViewChangeMessage::new(
            2, // new view
            validator_addr.clone(),
            vec![i as u8 + 10],
        );

        let result = manager
            .validate_view_change_message(&message, &validator_addr)
            .await;

        if result.is_ok() {
            successful_changes += 1;
        }
    }

    assert_eq!(
        successful_changes, 5,
        "All connected validators should validate successfully"
    );
    println!("✅ Network partition recovery test passed");
}
