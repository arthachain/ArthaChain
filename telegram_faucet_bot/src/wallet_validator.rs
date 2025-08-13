use regex::Regex;
use std::sync::OnceLock;

/// Wallet address validator
pub struct WalletValidator {
    eth_address_regex: Regex,
}

impl WalletValidator {
    /// Create a new wallet validator
    pub fn new() -> Self {
        let eth_address_regex = Regex::new(r"^0x[a-fA-F0-9]{40}$")
            .expect("Failed to compile Ethereum address regex");

        Self { eth_address_regex }
    }

    /// Validate if an address is a valid Ethereum-compatible address
    pub fn is_valid_address(&self, address: &str) -> bool {
        // Check basic format
        if !self.eth_address_regex.is_match(address) {
            return false;
        }

        // Additional validation
        self.is_valid_ethereum_address(address)
    }

    /// Validate Ethereum address format and checksum
    fn is_valid_ethereum_address(&self, address: &str) -> bool {
        // Remove 0x prefix
        let address = &address[2..];
        
        // Check length
        if address.len() != 40 {
            return false;
        }

        // Check if all characters are valid hex
        address.chars().all(|c| c.is_ascii_hexdigit())
    }

    /// Validate and normalize address (convert to checksum format)
    pub fn normalize_address(&self, address: &str) -> Option<String> {
        if !self.is_valid_address(address) {
            return None;
        }

        // Convert to lowercase and ensure 0x prefix
        let normalized = if address.starts_with("0x") {
            address.to_lowercase()
        } else {
            format!("0x{}", address.to_lowercase())
        };

        Some(normalized)
    }

    /// Check if address looks like a contract address (basic heuristic)
    pub fn looks_like_contract(&self, address: &str) -> bool {
        if !self.is_valid_address(address) {
            return false;
        }

        // Very basic heuristic: contracts often don't end in many zeros
        let address_part = &address[2..];
        let trailing_zeros = address_part.chars().rev()
            .take_while(|&c| c == '0')
            .count();

        // If it ends with more than 6 zeros, it's likely an EOA
        trailing_zeros <= 6
    }

    /// Get address type description
    pub fn get_address_type(&self, address: &str) -> String {
        if !self.is_valid_address(address) {
            return "Invalid".to_string();
        }

        if self.looks_like_contract(address) {
            "Contract (likely)".to_string()
        } else {
            "EOA (Externally Owned Account)".to_string()
        }
    }

    /// Validate multiple addresses
    pub fn validate_addresses(&self, addresses: &[String]) -> Vec<(String, bool)> {
        addresses.iter()
            .map(|addr| (addr.clone(), self.is_valid_address(addr)))
            .collect()
    }

    /// Extract addresses from text (useful for parsing messages)
    pub fn extract_addresses(&self, text: &str) -> Vec<String> {
        self.eth_address_regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }
}

impl Default for WalletValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Static instance for global access
static VALIDATOR: OnceLock<WalletValidator> = OnceLock::new();

/// Get global validator instance
pub fn get_validator() -> &'static WalletValidator {
    VALIDATOR.get_or_init(WalletValidator::new)
}

/// Quick validation function
pub fn is_valid_ethereum_address(address: &str) -> bool {
    get_validator().is_valid_address(address)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_addresses() {
        let validator = WalletValidator::new();

        // Valid addresses
        assert!(validator.is_valid_address("0x742d35Cc6634C0532925a3b844Bc454e4438f44e"));
        assert!(validator.is_valid_address("0x0000000000000000000000000000000000000000"));
        assert!(validator.is_valid_address("0xFFfFfFffFFfffFFfFFfFFFFFffFFFffffFfFFFfF"));

        // Invalid addresses
        assert!(!validator.is_valid_address("742d35Cc6634C0532925a3b844Bc454e4438f44e")); // No 0x
        assert!(!validator.is_valid_address("0x742d35Cc6634C0532925a3b844Bc454e4438f44")); // Too short
        assert!(!validator.is_valid_address("0x742d35Cc6634C0532925a3b844Bc454e4438f44ef")); // Too long
        assert!(!validator.is_valid_address("0x742d35Gc6634C0532925a3b844Bc454e4438f44e")); // Invalid character 'G'
        assert!(!validator.is_valid_address("")); // Empty
        assert!(!validator.is_valid_address("0x")); // Just prefix
    }

    #[test]
    fn test_normalize_address() {
        let validator = WalletValidator::new();

        assert_eq!(
            validator.normalize_address("0x742D35CC6634C0532925A3B844BC454E4438F44E"),
            Some("0x742d35cc6634c0532925a3b844bc454e4438f44e".to_string())
        );

        assert_eq!(
            validator.normalize_address("742d35Cc6634C0532925a3b844Bc454e4438f44e"),
            Some("0x742d35cc6634c0532925a3b844bc454e4438f44e".to_string())
        );

        assert_eq!(validator.normalize_address("invalid"), None);
    }

    #[test]
    fn test_extract_addresses() {
        let validator = WalletValidator::new();
        let text = "Send to 0x742d35Cc6634C0532925a3b844Bc454e4438f44e and also 0x0000000000000000000000000000000000000000";
        
        let addresses = validator.extract_addresses(text);
        assert_eq!(addresses.len(), 2);
        assert!(addresses.contains(&"0x742d35Cc6634C0532925a3b844Bc454e4438f44e".to_string()));
        assert!(addresses.contains(&"0x0000000000000000000000000000000000000000".to_string()));
    }

    #[test]
    fn test_contract_heuristic() {
        let validator = WalletValidator::new();

        // Typical EOA (many trailing zeros)
        assert!(!validator.looks_like_contract("0x742d35Cc6634C0532925a3b844Bc454e4438f44e"));
        
        // Likely contract (fewer trailing zeros)
        assert!(validator.looks_like_contract("0x1234567890123456789012345678901234567890"));
    }
}