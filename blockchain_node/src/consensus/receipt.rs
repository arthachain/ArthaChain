use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};

use crate::sharding::CrossShardStatus;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionReceipt {
    pub tx_hash: Vec<u8>,
    pub status: CrossShardStatus,
    pub execution_result: Vec<u8>,
    pub shard_signatures: HashMap<u64, Vec<u8>>,
    pub merkle_proof: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptChain {
    pub receipts: Vec<TransactionReceipt>,
    pub merkle_root: Vec<u8>,
    pub last_block_hash: Vec<u8>,
}

impl ReceiptChain {
    pub fn new() -> Self {
        Self {
            receipts: Vec::new(),
            merkle_root: Vec::new(),
            last_block_hash: Vec::new(),
        }
    }

    pub fn add_receipt(&mut self, receipt: TransactionReceipt) {
        self.receipts.push(receipt);
        self.update_merkle_root();
    }

    pub fn update_merkle_root(&mut self) {
        let mut hasher = Sha256::new();
        for receipt in &self.receipts {
            hasher.update(&receipt.tx_hash);
            hasher.update(&receipt.merkle_proof);
        }
        self.merkle_root = hasher.finalize().to_vec();
    }

    pub fn verify_receipt(&self, receipt: &TransactionReceipt) -> bool {
        // Verify receipt is in the chain
        self.receipts.iter().any(|r| r.tx_hash == receipt.tx_hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_receipt(tx_hash: Vec<u8>) -> TransactionReceipt {
        TransactionReceipt {
            tx_hash,
            status: CrossShardStatus::Confirmed,
            execution_result: vec![1, 2, 3, 4],
            shard_signatures: HashMap::new(),
            merkle_proof: vec![5, 6, 7, 8],
        }
    }

    #[test]
    fn test_receipt_chain() {
        let mut chain = ReceiptChain::new();
        assert!(chain.receipts.is_empty());
        assert!(chain.merkle_root.is_empty());

        // Add receipt
        let receipt = create_test_receipt(vec![1, 2, 3, 4]);
        chain.add_receipt(receipt.clone());
        
        assert_eq!(chain.receipts.len(), 1);
        assert!(!chain.merkle_root.is_empty());
        assert!(chain.verify_receipt(&receipt));

        // Add another receipt
        let receipt2 = create_test_receipt(vec![5, 6, 7, 8]);
        chain.add_receipt(receipt2.clone());
        
        assert_eq!(chain.receipts.len(), 2);
        assert!(chain.verify_receipt(&receipt2));
    }

    #[test]
    fn test_merkle_root_update() {
        let chain = ReceiptChain::new();
        let initial_root = chain.merkle_root.clone();

        // Add receipt and check root change
        let mut chain = ReceiptChain::new();
        chain.add_receipt(create_test_receipt(vec![1, 2, 3, 4]));
        assert_ne!(chain.merkle_root, initial_root);

        // Add another receipt and check root changes again
        let first_root = chain.merkle_root.clone();
        chain.add_receipt(create_test_receipt(vec![5, 6, 7, 8]));
        assert_ne!(chain.merkle_root, first_root);
    }

    fn create_receipt_with_status(tx_hash: Vec<u8>, status: CrossShardStatus) -> TransactionReceipt {
        TransactionReceipt {
            tx_hash,
            status,
            execution_result: vec![1, 2, 3, 4],
            shard_signatures: HashMap::new(),
            merkle_proof: vec![5, 6, 7, 8],
        }
    }

    #[test]
    fn test_verify_nonexistent_receipt() {
        let chain = ReceiptChain::new();
        let receipt = create_test_receipt(vec![1, 2, 3, 4]);
        assert!(!chain.verify_receipt(&receipt));
    }

    #[test]
    fn test_multiple_receipts_same_hash() {
        let mut chain = ReceiptChain::new();
        let hash = vec![1, 2, 3, 4];
        
        // Add first receipt
        let receipt1 = create_receipt_with_status(hash.clone(), CrossShardStatus::Pending);
        chain.add_receipt(receipt1.clone());
        
        // Add second receipt with same hash but different status
        let receipt2 = create_receipt_with_status(hash.clone(), CrossShardStatus::Confirmed);
        chain.add_receipt(receipt2.clone());
        
        assert_eq!(chain.receipts.len(), 2);
        assert!(chain.verify_receipt(&receipt1));
        assert!(chain.verify_receipt(&receipt2));
    }

    #[test]
    fn test_receipt_with_signatures() {
        let mut chain = ReceiptChain::new();
        let mut receipt = create_test_receipt(vec![1, 2, 3, 4]);
        
        // Add signatures from different shards
        receipt.shard_signatures.insert(1, vec![10, 11, 12]);
        receipt.shard_signatures.insert(2, vec![20, 21, 22]);
        
        chain.add_receipt(receipt.clone());
        assert!(chain.verify_receipt(&receipt));
        assert_eq!(receipt.shard_signatures.len(), 2);
    }

    #[test]
    fn test_empty_receipt_fields() {
        let mut chain = ReceiptChain::new();
        let receipt = TransactionReceipt {
            tx_hash: Vec::new(),
            status: CrossShardStatus::Pending,
            execution_result: Vec::new(),
            shard_signatures: HashMap::new(),
            merkle_proof: Vec::new(),
        };
        
        chain.add_receipt(receipt.clone());
        assert!(chain.verify_receipt(&receipt));
        assert!(!chain.merkle_root.is_empty());
    }

    #[test]
    fn test_merkle_root_consistency() {
        let mut chain1 = ReceiptChain::new();
        let mut chain2 = ReceiptChain::new();
        
        // Add same receipts in same order
        let receipt1 = create_test_receipt(vec![1, 2, 3, 4]);
        let receipt2 = create_test_receipt(vec![5, 6, 7, 8]);
        
        chain1.add_receipt(receipt1.clone());
        chain1.add_receipt(receipt2.clone());
        
        chain2.add_receipt(receipt1);
        chain2.add_receipt(receipt2);
        
        // Merkle roots should be identical
        assert_eq!(chain1.merkle_root, chain2.merkle_root);
    }
} 