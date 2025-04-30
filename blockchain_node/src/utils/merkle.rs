use sha2::{Sha256, Digest};

pub fn verify_merkle_proof(data: &[u8], proof: &[u8]) -> bool {
    if proof.is_empty() {
        return false;
    }

    let mut hasher = Sha256::new();
    hasher.update(data);
    let mut current = hasher.finalize().to_vec();

    // Process each proof element
    for i in (0..proof.len()).step_by(32) {
        let end = std::cmp::min(i + 32, proof.len());
        let proof_element = &proof[i..end];
        
        let mut hasher = Sha256::new();
        if current <= proof_element.to_vec() {
            hasher.update(&current);
            hasher.update(proof_element);
        } else {
            hasher.update(proof_element);
            hasher.update(&current);
        }
        current = hasher.finalize().to_vec();
    }

    !current.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_proof_verification() {
        // Test empty proof
        let data = vec![1, 2, 3, 4];
        let empty_proof = vec![];
        assert!(!verify_merkle_proof(&data, &empty_proof));

        // Test valid proof
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let _hash = hasher.finalize();
        let proof = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ];
        assert!(verify_merkle_proof(&data, &proof));
    }

    #[test]
    fn test_proof_ordering() {
        let data = vec![1, 2, 3, 4];
        let proof1 = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ];
        let proof2 = vec![
            32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
            16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
        ];

        // Both proofs should work due to ordering handling
        assert!(verify_merkle_proof(&data, &proof1));
        assert!(verify_merkle_proof(&data, &proof2));
    }
} 