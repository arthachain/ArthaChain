use crate::utils::crypto::quantum_resistant_hash;
use anyhow::Result;
use std::collections::VecDeque;

/// Node in a quantum-resistant Merkle tree
#[derive(Clone, Debug)]
pub struct MerkleNode {
    /// Hash of this node
    pub hash: Vec<u8>,
    /// Left child (None for leaf nodes)
    pub left: Option<Box<MerkleNode>>,
    /// Right child (None for leaf nodes)
    pub right: Option<Box<MerkleNode>>,
    /// Whether this is a leaf node
    pub is_leaf: bool,
    /// Original data for leaf nodes
    pub data: Option<Vec<u8>>,
}

impl MerkleNode {
    /// Create a new leaf node
    pub fn new_leaf(data: Vec<u8>) -> Result<Self> {
        let hash = quantum_resistant_hash(&data)?;

        Ok(Self {
            hash,
            left: None,
            right: None,
            is_leaf: true,
            data: Some(data),
        })
    }

    /// Create a new internal node
    pub fn new_internal(left: MerkleNode, right: MerkleNode) -> Result<Self> {
        let mut combined = Vec::with_capacity(left.hash.len() + right.hash.len());
        combined.extend_from_slice(&left.hash);
        combined.extend_from_slice(&right.hash);

        let hash = quantum_resistant_hash(&combined)?;

        Ok(Self {
            hash,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            is_leaf: false,
            data: None,
        })
    }
}

/// Quantum-resistant Merkle tree implementation
#[derive(Default)]
pub struct QuantumMerkleTree {
    /// Root of the Merkle tree
    pub root: Option<MerkleNode>,
    /// Number of leaves
    pub leaf_count: usize,
    /// Whether the tree has been built
    pub is_built: bool,
    /// Leaves of the tree
    pub leaves: Vec<MerkleNode>,
}

impl QuantumMerkleTree {
    /// Create a new, empty Merkle tree
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a Merkle tree from a list of data items
    pub fn build_from_data(data_list: &[Vec<u8>]) -> Result<Self> {
        if data_list.is_empty() {
            return Ok(Self::new());
        }

        // Create leaf nodes
        let mut queue: VecDeque<MerkleNode> = VecDeque::new();
        for data in data_list {
            queue.push_back(MerkleNode::new_leaf(data.clone())?);
        }

        // If odd number of leaves, duplicate the last one
        if queue.len() % 2 != 0 {
            let last = queue.back().unwrap().clone();
            queue.push_back(last);
        }

        // Build tree bottom-up
        while queue.len() > 1 {
            let left = queue.pop_front().unwrap();
            let right = queue.pop_front().unwrap();

            let parent = MerkleNode::new_internal(left, right)?;
            queue.push_back(parent);
        }

        let root = queue.pop_front();
        let leaf_count = data_list.len();

        Ok(Self {
            root,
            leaf_count,
            is_built: true,
            leaves: queue.into_iter().collect(),
        })
    }

    /// Get the root hash of the Merkle tree
    pub fn root_hash(&self) -> Option<Vec<u8>> {
        self.root.as_ref().map(|node| node.hash.clone())
    }

    /// Generate a Merkle proof for a data item
    pub fn generate_proof(&self, data: &[u8]) -> Result<Vec<Vec<u8>>> {
        let target_hash = quantum_resistant_hash(data)?;

        if let Some(root) = &self.root {
            let mut proof = Vec::new();
            if self.generate_proof_recursive(root, &target_hash, &mut proof) {
                Ok(proof)
            } else {
                Err(anyhow::anyhow!("Data not found in tree"))
            }
        } else {
            Err(anyhow::anyhow!("Tree is empty"))
        }
    }

    /// Generate proof recursively
    fn generate_proof_recursive(
        &self,
        node: &MerkleNode,
        target_hash: &[u8],
        proof: &mut Vec<Vec<u8>>,
    ) -> bool {
        if node.is_leaf {
            return node.hash == *target_hash;
        }

        if let Some(left) = &node.left {
            if self.generate_proof_recursive(left, target_hash, proof) {
                if let Some(right) = &node.right {
                    proof.push(right.hash.clone());
                }
                return true;
            }
        }

        if let Some(right) = &node.right {
            if self.generate_proof_recursive(right, target_hash, proof) {
                if let Some(left) = &node.left {
                    proof.push(left.hash.clone());
                }
                return true;
            }
        }

        false
    }

    /// Verify a Merkle proof for a data item
    pub fn verify_proof(data: &[u8], proof: &[Vec<u8>], root_hash: &[u8]) -> Result<bool> {
        let mut current_hash = quantum_resistant_hash(data)?;

        for sibling_hash in proof {
            let mut combined = Vec::with_capacity(current_hash.len() + sibling_hash.len());

            // Order by hash value to ensure consistent combination
            if current_hash <= *sibling_hash {
                combined.extend_from_slice(&current_hash);
                combined.extend_from_slice(sibling_hash);
            } else {
                combined.extend_from_slice(sibling_hash);
                combined.extend_from_slice(&current_hash);
            }

            current_hash = quantum_resistant_hash(&combined)?;
        }

        Ok(current_hash == *root_hash)
    }

    /// Add a leaf to the tree and rebuild it
    pub fn add_leaf(&mut self, data: &[u8]) -> Result<MerkleProof> {
        // Create new leaf node
        let leaf = MerkleNode::new_leaf(data.to_vec())?;

        // Get existing leaves
        let mut leaves = self.collect_leaves();

        // Add the new leaf
        leaves.push(leaf);
        self.leaf_count += 1;

        // Rebuild tree
        self.rebuild_from_leaves(&leaves)?;

        // Generate proof for the new leaf
        let root_hash = self
            .root_hash()
            .ok_or_else(|| anyhow::anyhow!("Tree is empty"))?;
        let proof_items = self.generate_proof(data)?;

        Ok(MerkleProof {
            data: data.to_vec(),
            proof_items,
            root_hash,
        })
    }

    /// Collect all leaf nodes in the tree
    fn collect_leaves(&self) -> Vec<MerkleNode> {
        let mut leaves = Vec::new();
        if let Some(root) = &self.root {
            self.collect_leaves_recursive(root, &mut leaves);
        }
        leaves
    }

    /// Recursively collect leaf nodes
    fn collect_leaves_recursive(&self, node: &MerkleNode, leaves: &mut Vec<MerkleNode>) {
        if node.is_leaf {
            leaves.push(node.clone());
            return;
        }

        if let Some(left) = &node.left {
            self.collect_leaves_recursive(left, leaves);
        }

        if let Some(right) = &node.right {
            self.collect_leaves_recursive(right, leaves);
        }
    }

    /// Rebuild tree from leaf nodes
    fn rebuild_from_leaves(&mut self, leaves: &[MerkleNode]) -> Result<()> {
        if leaves.is_empty() {
            self.root = None;
            self.is_built = false;
            return Ok(());
        }

        // Create queue with all leaves
        let mut queue: VecDeque<MerkleNode> = leaves.iter().cloned().collect();

        // If odd number of leaves, duplicate the last one
        if queue.len() % 2 != 0 {
            let last = queue.back().unwrap().clone();
            queue.push_back(last);
        }

        // Build tree bottom-up
        while queue.len() > 1 {
            let left = queue.pop_front().unwrap();
            let right = queue.pop_front().unwrap();

            let parent = MerkleNode::new_internal(left, right)?;
            queue.push_back(parent);
        }

        self.root = queue.pop_front();
        self.is_built = true;

        Ok(())
    }

    /// Get the root hash bytes
    pub fn root(&self) -> Vec<u8> {
        self.root_hash().unwrap_or_default()
    }
}

/// Efficient Merkle proof generator for large datasets
pub struct MerkleProofGenerator {
    /// The tree used for generating proofs
    tree: QuantumMerkleTree,
}

impl MerkleProofGenerator {
    /// Create a new Merkle proof generator
    pub fn new(data_list: &[Vec<u8>]) -> Result<Self> {
        let tree = QuantumMerkleTree::build_from_data(data_list)?;

        Ok(Self { tree })
    }

    /// Generate a proof for a specific data item
    pub fn generate_proof(&self, data: &[u8]) -> Result<MerkleProof> {
        let root_hash = self
            .tree
            .root_hash()
            .ok_or_else(|| anyhow::anyhow!("Tree is empty"))?;
        let proof_items = self.tree.generate_proof(data)?;

        Ok(MerkleProof {
            data: data.to_vec(),
            proof_items,
            root_hash,
        })
    }

    /// Get the root hash of the tree
    pub fn root_hash(&self) -> Option<Vec<u8>> {
        self.tree.root_hash()
    }
}

/// Merkle proof for verifying data inclusion
#[derive(Clone, Debug)]
pub struct MerkleProof {
    /// The data being proven
    pub data: Vec<u8>,
    /// The proof items (sibling hashes)
    pub proof_items: Vec<Vec<u8>>,
    /// The root hash of the tree
    pub root_hash: Vec<u8>,
}

impl MerkleProof {
    /// Verify this proof
    pub fn verify(&self) -> Result<bool> {
        QuantumMerkleTree::verify_proof(&self.data, &self.proof_items, &self.root_hash)
    }

    /// Serialize the proof for transmission
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut result = Vec::new();

        // Add data length and data
        result.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        result.extend_from_slice(&self.data);

        // Add number of proof items
        result.extend_from_slice(&(self.proof_items.len() as u32).to_le_bytes());

        // Add each proof item
        for item in &self.proof_items {
            result.extend_from_slice(&(item.len() as u32).to_le_bytes());
            result.extend_from_slice(item);
        }

        // Add root hash
        result.extend_from_slice(&(self.root_hash.len() as u32).to_le_bytes());
        result.extend_from_slice(&self.root_hash);

        Ok(result)
    }

    /// Deserialize a proof from binary
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        let mut offset = 0;

        // Read data length
        if offset + 4 > bytes.len() {
            return Err(anyhow::anyhow!("Insufficient data for data length"));
        }
        let data_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        // Read data
        if offset + data_len > bytes.len() {
            return Err(anyhow::anyhow!("Insufficient data for data content"));
        }
        let data = bytes[offset..offset + data_len].to_vec();
        offset += data_len;

        // Read number of proof items
        if offset + 4 > bytes.len() {
            return Err(anyhow::anyhow!("Insufficient data for proof items count"));
        }
        let proof_items_count = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        // Read proof items
        let mut proof_items = Vec::with_capacity(proof_items_count);
        for _ in 0..proof_items_count {
            if offset + 4 > bytes.len() {
                return Err(anyhow::anyhow!("Insufficient data for proof item length"));
            }
            let item_len = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + item_len > bytes.len() {
                return Err(anyhow::anyhow!("Insufficient data for proof item content"));
            }
            let item = bytes[offset..offset + item_len].to_vec();
            offset += item_len;

            proof_items.push(item);
        }

        // Read root hash
        if offset + 4 > bytes.len() {
            return Err(anyhow::anyhow!("Insufficient data for root hash length"));
        }
        let root_hash_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + root_hash_len > bytes.len() {
            return Err(anyhow::anyhow!("Insufficient data for root hash content"));
        }
        let root_hash = bytes[offset..offset + root_hash_len].to_vec();

        Ok(Self {
            data,
            proof_items,
            root_hash,
        })
    }
}

/// Light client for verifying Merkle proofs
pub struct LightClientVerifier {
    /// Known valid root hashes
    trusted_roots: Vec<Vec<u8>>,
}

impl LightClientVerifier {
    /// Create a new light client verifier
    pub fn new(trusted_roots: Vec<Vec<u8>>) -> Self {
        Self { trusted_roots }
    }

    /// Add a trusted root hash
    pub fn add_trusted_root(&mut self, root_hash: Vec<u8>) {
        self.trusted_roots.push(root_hash);
    }

    /// Verify a proof against a list of trusted roots
    pub fn verify_proof(&self, proof: &MerkleProof) -> Result<bool> {
        // First, verify the proof is internally consistent
        let is_valid = proof.verify()?;

        if !is_valid {
            return Ok(false);
        }

        // Then check if the root hash is trusted
        for trusted_root in &self.trusted_roots {
            if *trusted_root == proof.root_hash {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> Vec<Vec<u8>> {
        vec![
            b"transaction1".to_vec(),
            b"transaction2".to_vec(),
            b"transaction3".to_vec(),
            b"transaction4".to_vec(),
        ]
    }

    #[test]
    fn test_build_tree() -> Result<()> {
        let data = create_test_data();
        let tree = QuantumMerkleTree::build_from_data(&data)?;

        assert!(tree.is_built);
        assert_eq!(tree.leaf_count, 4);
        assert!(tree.root.is_some());

        Ok(())
    }

    #[test]
    fn test_generate_and_verify_proof() -> Result<()> {
        let data = create_test_data();
        let tree = QuantumMerkleTree::build_from_data(&data)?;

        let root_hash = tree.root_hash().unwrap();

        // Generate proof for transaction2
        let target_data = b"transaction2".to_vec();
        let proof = tree.generate_proof(&target_data)?;

        // Verify the proof
        let is_valid = QuantumMerkleTree::verify_proof(&target_data, &proof, &root_hash)?;
        assert!(is_valid);

        // Try with invalid data
        let invalid_data = b"invalid_transaction".to_vec();
        let is_valid = QuantumMerkleTree::verify_proof(&invalid_data, &proof, &root_hash)?;
        assert!(!is_valid);

        Ok(())
    }

    #[test]
    fn test_proof_generator() -> Result<()> {
        let data = create_test_data();
        let generator = MerkleProofGenerator::new(&data)?;

        let target_data = b"transaction3".to_vec();
        let proof = generator.generate_proof(&target_data)?;

        // Verify the proof
        let is_valid = proof.verify()?;
        assert!(is_valid);

        Ok(())
    }

    #[test]
    fn test_serialization() -> Result<()> {
        let data = create_test_data();
        let generator = MerkleProofGenerator::new(&data)?;

        let target_data = b"transaction1".to_vec();
        let proof = generator.generate_proof(&target_data)?;

        // Serialize and deserialize
        let serialized = proof.serialize()?;
        let deserialized = MerkleProof::deserialize(&serialized)?;

        // Verify deserialized proof
        let is_valid = deserialized.verify()?;
        assert!(is_valid);
        assert_eq!(deserialized.data, target_data);
        assert_eq!(deserialized.root_hash, proof.root_hash);

        Ok(())
    }

    #[test]
    fn test_light_client() -> Result<()> {
        let data = create_test_data();
        let generator = MerkleProofGenerator::new(&data)?;

        let root_hash = generator.root_hash().unwrap();

        // Create a light client with the trusted root
        let mut verifier = LightClientVerifier::new(vec![]);
        verifier.add_trusted_root(root_hash);

        // Generate and verify proof
        let target_data = b"transaction4".to_vec();
        let proof = generator.generate_proof(&target_data)?;

        // Verify with light client
        let is_valid = verifier.verify_proof(&proof)?;
        assert!(is_valid);

        // Try with untrusted root
        let untrusted_root = vec![0; 32]; // Some random root
        let mut untrusted_proof = proof.clone();
        untrusted_proof.root_hash = untrusted_root;

        let is_valid = verifier.verify_proof(&untrusted_proof)?;
        assert!(!is_valid);

        Ok(())
    }
}
