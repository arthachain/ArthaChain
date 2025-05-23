/// Propose a new block
pub async fn propose_block(&self, block_data: Vec<u8>, height: u64) -> Result<Vec<u8>> {
    // Generate a placeholder block hash
    let mut rng = rand::thread_rng();
    let mut block_hash = Vec::with_capacity(32);
    for _ in 0..32 {
        // Use gen_range instead of gen() to avoid syntax issues
        let random_byte = rng.gen_range(0..=255);
        block_hash.push(random_byte);
    }

    // Log the proposal
    info!(
        "Proposing block at height {} with hash {:?}",
        height, block_hash
    );

    // Create propose message
    let propose = ConsensusMessageType::Propose {
        block_data,
        height,
        block_hash: block_hash.clone(),
    };

    // Create new round
    let mut rounds = self.active_rounds.write().await;
    rounds.insert(
        block_hash.clone(),
        ConsensusRound {
            block_hash: block_hash.clone(),
            height,
            status: ConsensusStatus::Initial,
            start_time: Instant::now(),
            pre_votes: HashMap::new(),
            pre_commits: HashMap::new(),
            commits: HashMap::new(),
        },
    );

    // Broadcast proposal to all validators
    for &validator in self.validators.read().await.iter() {
        if let Err(e) = self.tx_sender.send((propose.clone(), validator)).await {
            error!("Failed to send proposal: {}", e);
        }
    }

    Ok(block_hash)
} 