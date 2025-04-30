use crate::ledger::state::{State, ShardConfig};
use crate::config::Config;

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_new_state() {
        let config = Config::new();
        let state = State::new(&config).expect("Failed to create state");
        assert_eq!(state.get_height().unwrap(), 0);
        assert_eq!(state.get_shard_id(), 0);
        assert!(state.get_pending_transactions(10).is_empty());
    }

    #[test]
    fn test_account_operations() {
        let config = Config::new();
        let state = State::new(&config).expect("Failed to create state");
        let addr = "0x1234";
        // Test initial balance is zero
        assert_eq!(state.get_balance(addr).unwrap(), 0);
        // Test updating balance
        state.set_balance(addr, 100).unwrap();
        assert_eq!(state.get_balance(addr).unwrap(), 100);
        // Test nonce operations
        assert_eq!(state.get_next_nonce(addr).unwrap(), 1); // nonce starts at 0, next is 1
        state.set_nonce(addr, 1).unwrap();
        assert_eq!(state.get_next_nonce(addr).unwrap(), 2);
    }
} 