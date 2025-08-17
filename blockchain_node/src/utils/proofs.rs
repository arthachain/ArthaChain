use crate::utils::crypto::Hash;
use std::result::Result;

/// Verify a merkle proof
pub fn verify_proof(
    prev_root: &Hash,
    new_root: &Hash,
    proof: &[u8],
) -> Result<bool, Box<dyn std::error::Error>> {
    use blake3::Hasher;

    // Parse the proof - it should contain state transition elements
    if proof.is_empty() {
        return Err("Empty proof".into());
    }

    // Simple proof verification for state transitions
    // In a full implementation, this would verify a Merkle proof of state changes

    // For now, implement basic validation:
    // 1. Proof should be at least 64 bytes (two hashes)
    if proof.len() < 64 {
        return Err("Proof too short".into());
    }

    // 2. Extract intermediate hash from proof
    let intermediate_hash = &proof[0..32];
    let transition_hash = &proof[32..64];

    // 3. Verify the state transition
    let mut hasher = Hasher::new();
    hasher.update(prev_root.as_ref());
    hasher.update(intermediate_hash);
    let computed_transition = hasher.finalize();

    // 4. Check if transition hash matches
    if computed_transition.as_bytes() != transition_hash {
        return Ok(false);
    }

    // 5. Verify final state
    let mut final_hasher = Hasher::new();
    final_hasher.update(transition_hash);
    final_hasher.update(&proof[64..]); // Additional proof data
    let computed_final = final_hasher.finalize();

    // 6. Compare with new root
    Ok(computed_final.as_bytes() == new_root.as_ref())
}

/// Verify a standard Merkle proof for inclusion
pub fn verify_merkle_inclusion(
    root: &[u8],
    leaf: &[u8],
    proof: &[u8],
    index: usize,
) -> Result<bool, Box<dyn std::error::Error>> {
    use blake3::Hasher;

    if proof.is_empty() {
        return Ok(root == leaf); // Single element tree
    }

    // Each proof element is 32 bytes
    if proof.len() % 32 != 0 {
        return Err("Invalid proof length - must be multiple of 32".into());
    }

    let mut current_hash = leaf.to_vec();
    let mut current_index = index;

    // Process each sibling hash in the proof
    for chunk in proof.chunks(32) {
        let mut hasher = Hasher::new();

        if current_index % 2 == 0 {
            // Current node is left child
            hasher.update(&current_hash);
            hasher.update(chunk);
        } else {
            // Current node is right child
            hasher.update(chunk);
            hasher.update(&current_hash);
        }

        current_hash = hasher.finalize().as_bytes().to_vec();
        current_index /= 2;
    }

    Ok(current_hash == root)
}
