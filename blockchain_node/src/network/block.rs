use crate::types::{Block, Hash, BlockHeader};
use crate::network::error::NetworkError;
use crate::consensus::ConsensusMessage;

impl BlockManager {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            headers: HashMap::new(),
            pending: HashSet::new(),
        }
    }

    pub fn process_block(&mut self, block: Block) -> Result<(), NetworkError> {
        let hash = block.hash()?;
        
        // Validate block header
        if !self.validate_header(&block.header) {
            return Err(NetworkError::InvalidBlockHeader);
        }

        // Check if we already have this block
        if self.blocks.contains_key(&hash) {
            return Ok(());
        }

        // Store block and header
        self.headers.insert(hash, block.header.clone());
        self.blocks.insert(hash, block);
        self.pending.remove(&hash);

        Ok(())
    }

    pub fn validate_header(&self, header: &BlockHeader) -> bool {
        // Basic header validation
        if header.timestamp > SystemTime::now() {
            return false;
        }

        // Check parent exists unless genesis
        if header.height > 0 && !self.headers.contains_key(&header.parent_hash) {
            return false;
        }

        true
    }

    pub fn get_block(&self, hash: &Hash) -> Option<&Block> {
        self.blocks.get(hash)
    }

    pub fn get_header(&self, hash: &Hash) -> Option<&BlockHeader> {
        self.headers.get(hash)
    }
} 