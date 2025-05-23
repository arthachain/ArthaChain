pub mod block_tests;
pub mod config;
pub mod state_tests;
pub mod transaction_tests;

#[cfg(test)]
mod integration_tests {
    use crate::config::Config;
    use crate::ledger::state::{ShardConfig, State};

    // Mock implementation of ShardConfig for testing
    #[allow(dead_code)]
    struct MockConfig {
        pub shard_id: u64,
    }

    impl ShardConfig for MockConfig {
        fn get_shard_id(&self) -> u64 {
            self.shard_id
        }

        fn get_genesis_config(&self) -> Option<&Config> {
            None
        }

        fn is_sharding_enabled(&self) -> bool {
            false
        }

        fn get_shard_count(&self) -> u32 {
            1
        }

        fn get_primary_shard(&self) -> u32 {
            0
        }
    }

    #[tokio::test]
    async fn test_basic_blockchain_flow() {
        // Create a state
        let config = Config::new();
        let state = State::new(&config).expect("Failed to create state");
        let sender = "0x1234";
        let recipient = "0x5678";
        state.set_balance(sender, 1000).unwrap();
        // Check balances
        assert_eq!(state.get_balance(sender).unwrap(), 1000);
        assert_eq!(state.get_balance(recipient).unwrap(), 0);
        // Set and check nonce
        state.set_nonce(sender, 0).unwrap();
        assert_eq!(state.get_nonce(sender).unwrap(), 0);
        assert_eq!(state.get_next_nonce(sender).unwrap(), 1);
    }
}
