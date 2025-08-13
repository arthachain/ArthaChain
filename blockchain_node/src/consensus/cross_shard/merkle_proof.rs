use anyhow::{anyhow, Result};
use log::{debug, info};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Merkle proof for cross-shard transaction validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    /// Transaction hash being proven
    pub tx_hash: Vec<u8>,
    /// Merkle root hash
    pub root_hash: Vec<u8>,
    /// Proof path (sibling hashes)
    pub proof_path: Vec<Vec<u8>>,
    /// Direction indicators (true = right, false = left)
    pub directions: Vec<bool>,
    /// Block height where transaction was included
    pub block_height: u64,
    /// Shard ID where transaction originated
    pub shard_id: u32,
}

impl MerkleProof {
    /// Create a new Merkle proof
    pub fn new(
        tx_hash: Vec<u8>,
        root_hash: Vec<u8>,
        proof_path: Vec<Vec<u8>>,
        directions: Vec<bool>,
        block_height: u64,
        shard_id: u32,
    ) -> Self {
        Self {
            tx_hash,
            root_hash,
            proof_path,
            directions,
            block_height,
            shard_id,
        }
    }

    /// Verify the Merkle proof
    pub fn verify(&self) -> Result<bool> {
        if self.proof_path.len() != self.directions.len() {
            return Err(anyhow!("Proof path and directions length mismatch"));
        }

        let mut current_hash = self.tx_hash.clone();

        // Reconstruct the root hash using the proof path
        for (sibling_hash, is_right) in self.proof_path.iter().zip(self.directions.iter()) {
            current_hash = if *is_right {
                // Current hash is on the left, sibling on the right
                Self::hash_pair(&current_hash, sibling_hash)
            } else {
                // Current hash is on the right, sibling on the left
                Self::hash_pair(sibling_hash, &current_hash)
            };
        }

        // Check if reconstructed root matches expected root
        let verified = current_hash == self.root_hash;

        debug!(
            "Merkle proof verification for tx {:?}: {}",
            hex::encode(&self.tx_hash),
            if verified { "VALID" } else { "INVALID" }
        );

        Ok(verified)
    }

    /// Hash a pair of values
    fn hash_pair(left: &[u8], right: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().to_vec()
    }
}

/// Merkle tree for generating proofs
pub struct MerkleTree {
    /// Tree levels (level 0 is leaves, root is at highest level)
    levels: Vec<Vec<Vec<u8>>>,
    /// Mapping from transaction hash to leaf index
    tx_to_index: HashMap<Vec<u8>, usize>,
}

impl MerkleTree {
    /// Build a Merkle tree from transaction hashes
    pub fn build(transaction_hashes: Vec<Vec<u8>>) -> Result<Self> {
        if transaction_hashes.is_empty() {
            return Err(anyhow!(
                "Cannot build Merkle tree from empty transaction list"
            ));
        }

        // Create mapping from tx hash to index
        let tx_to_index: HashMap<Vec<u8>, usize> = transaction_hashes
            .iter()
            .enumerate()
            .map(|(i, hash)| (hash.clone(), i))
            .collect();

        let mut levels = vec![transaction_hashes.clone()];
        let mut current_level = transaction_hashes;

        // Build tree bottom-up
        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            // Process pairs
            for chunk in current_level.chunks(2) {
                let hash = if chunk.len() == 2 {
                    // Hash pair
                    MerkleProof::hash_pair(&chunk[0], &chunk[1])
                } else {
                    // Odd number, hash with itself
                    MerkleProof::hash_pair(&chunk[0], &chunk[0])
                };
                next_level.push(hash);
            }

            levels.push(next_level.clone());
            current_level = next_level;
        }

        Ok(Self {
            levels,
            tx_to_index,
        })
    }

    /// Get the root hash
    pub fn root_hash(&self) -> Result<Vec<u8>> {
        if let Some(root_level) = self.levels.last() {
            if let Some(root) = root_level.first() {
                Ok(root.clone())
            } else {
                Err(anyhow!("Empty root level"))
            }
        } else {
            Err(anyhow!("No levels in tree"))
        }
    }

    /// Generate a Merkle proof for a transaction
    pub fn generate_proof(
        &self,
        tx_hash: &[u8],
        block_height: u64,
        shard_id: u32,
    ) -> Result<MerkleProof> {
        // Find the transaction index
        let leaf_index = self
            .tx_to_index
            .get(tx_hash)
            .ok_or_else(|| anyhow!("Transaction not found in tree"))?;

        let mut proof_path = Vec::new();
        let mut directions = Vec::new();
        let mut current_index = *leaf_index;

        // Generate proof path from leaf to root
        for level in 0..(self.levels.len() - 1) {
            let current_level = &self.levels[level];
            let sibling_index = if current_index % 2 == 0 {
                // Current is left child, sibling is right
                current_index + 1
            } else {
                // Current is right child, sibling is left
                current_index - 1
            };

            // Get sibling hash (or duplicate if odd number of nodes)
            let sibling_hash = if sibling_index < current_level.len() {
                current_level[sibling_index].clone()
            } else {
                current_level[current_index].clone() // Duplicate for odd number
            };

            proof_path.push(sibling_hash);
            directions.push(current_index % 2 == 0); // true if current is left child

            // Move to parent index
            current_index /= 2;
        }

        let root_hash = self.root_hash()?;

        Ok(MerkleProof::new(
            tx_hash.to_vec(),
            root_hash,
            proof_path,
            directions,
            block_height,
            shard_id,
        ))
    }
}

/// Cross-shard transaction with Merkle proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenTransaction {
    /// Transaction data
    pub transaction_data: Vec<u8>,
    /// Merkle proof of inclusion
    pub proof: MerkleProof,
    /// Source shard ID
    pub source_shard: u32,
    /// Target shard ID
    pub target_shard: u32,
    /// Transaction timestamp
    pub timestamp: u64,
}

impl ProvenTransaction {
    /// Create a new proven transaction
    pub fn new(
        transaction_data: Vec<u8>,
        proof: MerkleProof,
        source_shard: u32,
        target_shard: u32,
        timestamp: u64,
    ) -> Self {
        Self {
            transaction_data,
            proof,
            source_shard,
            target_shard,
            timestamp,
        }
    }

    /// Verify the transaction proof
    pub fn verify(&self) -> Result<bool> {
        // Verify the Merkle proof
        let proof_valid = self.proof.verify()?;

        if !proof_valid {
            return Ok(false);
        }

        // Verify transaction hash matches proof
        let tx_hash = Self::hash_transaction(&self.transaction_data);
        if tx_hash != self.proof.tx_hash {
            return Ok(false);
        }

        // Verify shard consistency
        if self.source_shard != self.proof.shard_id {
            return Ok(false);
        }

        info!(
            "Cross-shard transaction verified: {} -> {}",
            self.source_shard, self.target_shard
        );

        Ok(true)
    }

    /// Hash transaction data
    fn hash_transaction(data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }
}

/// Proof cache for efficient verification
pub struct ProofCache {
    /// Cached proofs by transaction hash
    cache: HashMap<Vec<u8>, MerkleProof>,
    /// Maximum cache size
    max_size: usize,
}

impl ProofCache {
    /// Create a new proof cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }

    /// Store a proof in cache
    pub fn store(&mut self, tx_hash: Vec<u8>, proof: MerkleProof) {
        if self.cache.len() >= self.max_size {
            // Remove oldest entry (simple LRU approximation)
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }
        self.cache.insert(tx_hash, proof);
    }

    /// Retrieve a proof from cache
    pub fn get(&self, tx_hash: &[u8]) -> Option<&MerkleProof> {
        self.cache.get(tx_hash)
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// Get all cached transaction hashes
    pub fn get_cached_hashes(&self) -> Vec<Vec<u8>> {
        self.cache.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_single_tx() {
        let tx_hashes = vec![vec![1, 2, 3, 4]];
        let tree = MerkleTree::build(tx_hashes.clone()).unwrap();

        // Root should be the single transaction hash
        let root = tree.root_hash().unwrap();
        assert_eq!(root, MerkleProof::hash_pair(&tx_hashes[0], &tx_hashes[0]));
    }

    #[test]
    fn test_merkle_tree_two_tx() {
        let tx_hashes = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        let tree = MerkleTree::build(tx_hashes.clone()).unwrap();

        // Root should be hash of the two transactions
        let expected_root = MerkleProof::hash_pair(&tx_hashes[0], &tx_hashes[1]);
        let root = tree.root_hash().unwrap();
        assert_eq!(root, expected_root);
    }

    #[test]
    fn test_merkle_proof_generation_and_verification() {
        let tx_hashes = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ];

        let tree = MerkleTree::build(tx_hashes.clone()).unwrap();

        // Generate proof for first transaction
        let proof = tree.generate_proof(&tx_hashes[0], 100, 1).unwrap();

        // Verify the proof
        assert!(proof.verify().unwrap());

        // Verify proof contains correct transaction hash
        assert_eq!(proof.tx_hash, tx_hashes[0]);
        assert_eq!(proof.block_height, 100);
        assert_eq!(proof.shard_id, 1);
    }

    #[test]
    fn test_proven_transaction_verification() {
        let tx_data = vec![1, 2, 3, 4, 5];
        let tx_hash = ProvenTransaction::hash_transaction(&tx_data);

        let tx_hashes = vec![tx_hash.clone(), vec![6, 7, 8, 9]];
        let tree = MerkleTree::build(tx_hashes).unwrap();

        let proof = tree.generate_proof(&tx_hash, 100, 1).unwrap();
        let proven_tx = ProvenTransaction::new(tx_data, proof, 1, 2, 1234567890);

        assert!(proven_tx.verify().unwrap());
    }

    #[test]
    fn test_proof_cache() {
        let mut cache = ProofCache::new(2);

        let tx_hash1 = vec![1, 2, 3];
        let tx_hash2 = vec![4, 5, 6];
        let tx_hash3 = vec![7, 8, 9];

        let proof1 = MerkleProof::new(tx_hash1.clone(), vec![], vec![], vec![], 100, 1);
        let proof2 = MerkleProof::new(tx_hash2.clone(), vec![], vec![], vec![], 101, 1);
        let proof3 = MerkleProof::new(tx_hash3.clone(), vec![], vec![], vec![], 102, 1);

        cache.store(tx_hash1.clone(), proof1);
        cache.store(tx_hash2.clone(), proof2);

        assert!(cache.get(&tx_hash1).is_some());
        assert!(cache.get(&tx_hash2).is_some());

        // Adding third proof should evict first one (LRU)
        cache.store(tx_hash3.clone(), proof3);
        assert!(cache.get(&tx_hash3).is_some());
        assert_eq!(cache.cache.len(), 2); // Max size maintained
    }
}
