use crate::ledger::block::Block;
use crate::network::types::NodeId;
use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

/// A vertex in the directed acyclic graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagVertex {
    /// Hash of the block
    pub hash: Vec<u8>,
    /// Creator of the block
    pub creator: NodeId,
    /// Height in the DAG
    pub height: u64,
    /// Timestamp of creation
    pub timestamp: u64,
    /// Parents of this vertex (hashes)
    pub parents: Vec<Vec<u8>>,
    /// Children of this vertex (hashes)
    pub children: Vec<Vec<u8>>,
    /// Whether this vertex is finalized
    pub finalized: bool,
    /// References to transactions included
    pub tx_refs: Vec<Vec<u8>>,
    /// Block data (optional, may be stored elsewhere)
    pub block: Option<Block>,
}

impl DagVertex {
    /// Create a new DAG vertex
    pub fn new(
        hash: Vec<u8>,
        creator: NodeId,
        height: u64,
        timestamp: u64,
        parents: Vec<Vec<u8>>,
        tx_refs: Vec<Vec<u8>>,
        block: Option<Block>,
    ) -> Self {
        Self {
            hash,
            creator,
            height,
            timestamp,
            parents,
            children: Vec::new(),
            finalized: false,
            tx_refs,
            block,
        }
    }
}

/// Configuration for the DAG Manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagConfig {
    /// Maximum number of parents for a vertex
    pub max_parents: usize,
    /// Maximum number of unfinalized vertices
    pub max_unfinalized: usize,
    /// Finality threshold (number of confirmations)
    pub finality_threshold: usize,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Pruning interval in seconds
    pub pruning_interval_secs: u64,
    /// Sync verification level
    pub sync_verification_level: SyncVerificationLevel,
    /// Enable parallel processing
    pub parallel_processing: bool,
}

impl Default for DagConfig {
    fn default() -> Self {
        Self {
            max_parents: 3,
            max_unfinalized: 1000,
            finality_threshold: 6,
            max_memory_mb: 1024,
            pruning_interval_secs: 300,
            sync_verification_level: SyncVerificationLevel::Full,
            parallel_processing: true,
        }
    }
}

/// Verification level for DAG sync
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncVerificationLevel {
    /// No verification
    None,
    /// Light verification (headers only)
    Light,
    /// Full verification
    Full,
}

/// DAG Manager to handle directed acyclic graph operations
pub struct DagManager {
    /// Configuration
    config: RwLock<DagConfig>,
    /// Vertices by hash
    vertices: RwLock<HashMap<Vec<u8>, DagVertex>>,
    /// Finalized vertices by height
    finalized_by_height: RwLock<HashMap<u64, Vec<Vec<u8>>>>,
    /// Unfinalized vertices by height
    unfinalized_by_height: RwLock<HashMap<u64, Vec<Vec<u8>>>>,
    /// Tips of the DAG (vertices with no children)
    tips: RwLock<HashSet<Vec<u8>>>,
    /// Genesis vertex hash
    genesis_hash: Vec<u8>,
    /// Current highest finalized height
    highest_finalized_height: RwLock<u64>,
    /// Running flag
    running: RwLock<bool>,
}

impl DagManager {
    /// Create a new DAG manager
    pub fn new(config: DagConfig, genesis_block: Block) -> Result<Self> {
        let genesis_hash = genesis_block.hash.clone();

        // Create genesis vertex
        let genesis_vertex = DagVertex::new(
            genesis_hash.clone(),
            "genesis".to_string(),
            0,
            genesis_block.timestamp.unwrap_or(0),
            Vec::new(),
            genesis_block.txs.iter().map(|tx| tx.hash.clone()).collect(),
            Some(genesis_block),
        );

        let mut vertices = HashMap::new();
        vertices.insert(genesis_hash.clone(), genesis_vertex);

        let mut finalized_by_height = HashMap::new();
        finalized_by_height.insert(0, vec![genesis_hash.clone()]);

        let mut tips = HashSet::new();
        tips.insert(genesis_hash.clone());

        Ok(Self {
            config: RwLock::new(config),
            vertices: RwLock::new(vertices),
            finalized_by_height: RwLock::new(finalized_by_height),
            unfinalized_by_height: RwLock::new(HashMap::new()),
            tips: RwLock::new(tips),
            genesis_hash,
            highest_finalized_height: RwLock::new(0),
            running: RwLock::new(false),
        })
    }

    /// Start the DAG manager
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(anyhow!("DAG manager already running"));
        }

        *running = true;

        // Start background tasks
        self.start_pruning_task();

        info!("DAG manager started");
        Ok(())
    }

    /// Stop the DAG manager
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if !*running {
            return Err(anyhow!("DAG manager is not running"));
        }

        *running = false;
        info!("DAG manager stopped");
        Ok(())
    }

    /// Start the background pruning task
    fn start_pruning_task(&self) {
        let self_clone = Arc::new(self.clone());

        tokio::spawn(async move {
            let interval_secs = {
                let config = self_clone.config.read().await;
                config.pruning_interval_secs
            };

            let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));

            loop {
                interval.tick().await;

                let is_running = {
                    let running = self_clone.running.read().await;
                    *running
                };

                if !is_running {
                    break;
                }

                if let Err(e) = self_clone.prune_old_vertices().await {
                    warn!("Error during DAG pruning: {}", e);
                }
            }
        });
    }

    /// Add a new vertex to the DAG
    pub async fn add_vertex(&self, vertex: DagVertex) -> Result<()> {
        let vertex_hash = vertex.hash.clone();

        // Check if we already have this vertex
        {
            let vertices = self.vertices.read().await;
            if vertices.contains_key(&vertex_hash) {
                return Ok(()); // Already exists
            }
        }

        // Check parent validity
        self.validate_parents(&vertex).await?;

        // Add vertex to the DAG
        {
            let mut vertices = self.vertices.write().await;
            let mut tips = self.tips.write().await;
            let mut unfinalized_by_height = self.unfinalized_by_height.write().await;

            // Update parent-child relationships
            for parent_hash in &vertex.parents {
                if let Some(parent) = vertices.get_mut(parent_hash) {
                    parent.children.push(vertex_hash.clone());

                    // Remove parent from tips if it's there
                    tips.remove(parent_hash);
                }
            }

            // Add to unfinalized by height
            let height_vertices = unfinalized_by_height
                .entry(vertex.height)
                .or_insert_with(Vec::new);
            height_vertices.push(vertex_hash.clone());

            // Add to tips
            tips.insert(vertex_hash.clone());

            // Add to vertices
            vertices.insert(vertex_hash.clone(), vertex.clone());
        }

        // Try to finalize vertices
        self.try_finalize().await?;

        debug!(
            "Added vertex {} at height {}",
            hex::encode(&vertex_hash),
            vertex.height
        );
        Ok(())
    }

    /// Validate parents of a new vertex
    async fn validate_parents(&self, vertex: &DagVertex) -> Result<()> {
        let vertices = self.vertices.read().await;
        let config = self.config.read().await;

        // Check number of parents
        if vertex.parents.len() > config.max_parents {
            return Err(anyhow!(
                "Too many parents: {} > {}",
                vertex.parents.len(),
                config.max_parents
            ));
        }

        // Check that parents exist
        for parent_hash in &vertex.parents {
            if !vertices.contains_key(parent_hash) {
                return Err(anyhow!("Parent {} not found", hex::encode(parent_hash)));
            }
        }

        // Check parent heights
        for parent_hash in &vertex.parents {
            let parent = vertices.get(parent_hash).unwrap();
            if parent.height >= vertex.height {
                return Err(anyhow!(
                    "Invalid parent height: parent {} has height {} >= vertex height {}",
                    hex::encode(parent_hash),
                    parent.height,
                    vertex.height
                ));
            }
        }

        Ok(())
    }

    /// Try to finalize vertices
    async fn try_finalize(&self) -> Result<()> {
        let config = self.config.read().await;
        let finality_threshold = config.finality_threshold;

        let mut heights_to_check = Vec::new();
        {
            let unfinalized_by_height = self.unfinalized_by_height.read().await;
            heights_to_check = unfinalized_by_height.keys().cloned().collect();
            heights_to_check.sort();
        }

        for height in heights_to_check {
            let highest_finalized = *self.highest_finalized_height.read().await;

            // Only try to finalize heights above the highest finalized height and up to
            // the finality threshold
            if height <= highest_finalized || height > highest_finalized + finality_threshold as u64
            {
                continue;
            }

            // Check if we can finalize this height
            let confirmations = {
                let vertices = self.vertices.read().await;
                let unfinalized_by_height = self.unfinalized_by_height.read().await;

                if let Some(unfinalized) = unfinalized_by_height.get(&height) {
                    // Count confirmations for each unfinalized vertex
                    let mut min_confirmations = std::usize::MAX;

                    for vertex_hash in unfinalized {
                        let confirmations = self.count_confirmations(vertex_hash, &vertices)?;
                        min_confirmations = min_confirmations.min(confirmations);
                    }

                    min_confirmations
                } else {
                    0
                }
            };

            // If all vertices at this height have enough confirmations, finalize them
            if confirmations >= finality_threshold {
                self.finalize_height(height).await?;
            }
        }

        Ok(())
    }

    /// Count confirmations for a vertex
    fn count_confirmations(
        &self,
        vertex_hash: &[u8],
        vertices: &HashMap<Vec<u8>, DagVertex>,
    ) -> Result<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut max_distance = 0;

        // Start from the vertex's children
        if let Some(vertex) = vertices.get(vertex_hash) {
            for child_hash in &vertex.children {
                queue.push_back((child_hash.clone(), 1));
            }
        } else {
            return Err(anyhow!("Vertex {} not found", hex::encode(vertex_hash)));
        }

        // BFS to find the maximum distance
        while let Some((hash, distance)) = queue.pop_front() {
            if visited.contains(&hash) {
                continue;
            }

            visited.insert(hash.clone());
            max_distance = max_distance.max(distance);

            if let Some(vertex) = vertices.get(&hash) {
                for child_hash in &vertex.children {
                    if !visited.contains(child_hash) {
                        queue.push_back((child_hash.clone(), distance + 1));
                    }
                }
            }
        }

        Ok(max_distance)
    }

    /// Finalize all vertices at a given height
    async fn finalize_height(&self, height: u64) -> Result<()> {
        let mut vertices = self.vertices.write().await;
        let mut unfinalized_by_height = self.unfinalized_by_height.write().await;
        let mut finalized_by_height = self.finalized_by_height.write().await;
        let mut highest_finalized_height = self.highest_finalized_height.write().await;

        // Get vertices to finalize
        let vertex_hashes = if let Some(unfinalized) = unfinalized_by_height.remove(&height) {
            unfinalized
        } else {
            return Ok(()); // No vertices to finalize
        };

        // Update vertices to finalized
        for hash in &vertex_hashes {
            if let Some(vertex) = vertices.get_mut(hash) {
                vertex.finalized = true;
            }
        }

        // Add to finalized by height
        finalized_by_height.insert(height, vertex_hashes.clone());

        // Update highest finalized height
        if height > *highest_finalized_height {
            *highest_finalized_height = height;
        }

        info!(
            "Finalized {} vertices at height {}",
            vertex_hashes.len(),
            height
        );
        Ok(())
    }

    /// Prune old vertices to save memory
    async fn prune_old_vertices(&self) -> Result<()> {
        let config = self.config.read().await;
        let highest_finalized_height = *self.highest_finalized_height.read().await;

        // Only keep a certain number of finalized heights
        let prune_below_height = if highest_finalized_height > 100 {
            highest_finalized_height - 100
        } else {
            0
        };

        // Collect vertices to prune
        let vertex_hashes_to_prune = {
            let mut to_prune = Vec::new();
            let finalized_by_height = self.finalized_by_height.read().await;

            for (height, hashes) in finalized_by_height.iter() {
                if *height < prune_below_height {
                    to_prune.extend(hashes.clone());
                }
            }

            to_prune
        };

        // Prune vertices
        if !vertex_hashes_to_prune.is_empty() {
            let mut vertices = self.vertices.write().await;
            let mut finalized_by_height = self.finalized_by_height.write().await;

            // Remove vertices
            for hash in &vertex_hashes_to_prune {
                vertices.remove(hash);
            }

            // Remove heights
            let heights_to_remove: Vec<u64> = finalized_by_height
                .iter()
                .filter(|(h, _)| **h < prune_below_height)
                .map(|(h, _)| *h)
                .collect();

            for height in heights_to_remove {
                finalized_by_height.remove(&height);
            }

            info!(
                "Pruned {} vertices below height {}",
                vertex_hashes_to_prune.len(),
                prune_below_height
            );
        }

        Ok(())
    }

    /// Get a vertex by hash
    pub async fn get_vertex(&self, hash: &[u8]) -> Option<DagVertex> {
        let vertices = self.vertices.read().await;
        vertices.get(hash).cloned()
    }

    /// Get vertices at a specific height
    pub async fn get_vertices_at_height(&self, height: u64) -> Vec<DagVertex> {
        let mut result = Vec::new();
        let vertices = self.vertices.read().await;
        let finalized_by_height = self.finalized_by_height.read().await;
        let unfinalized_by_height = self.unfinalized_by_height.read().await;

        // Add finalized vertices
        if let Some(hashes) = finalized_by_height.get(&height) {
            for hash in hashes {
                if let Some(vertex) = vertices.get(hash) {
                    result.push(vertex.clone());
                }
            }
        }

        // Add unfinalized vertices
        if let Some(hashes) = unfinalized_by_height.get(&height) {
            for hash in hashes {
                if let Some(vertex) = vertices.get(hash) {
                    result.push(vertex.clone());
                }
            }
        }

        result
    }

    /// Get tips of the DAG
    pub async fn get_tips(&self) -> Vec<DagVertex> {
        let tips = self.tips.read().await;
        let vertices = self.vertices.read().await;

        let mut result = Vec::new();
        for hash in tips.iter() {
            if let Some(vertex) = vertices.get(hash) {
                result.push(vertex.clone());
            }
        }

        result
    }

    /// Get the best tip for creating a new vertex
    pub async fn get_best_tip(&self) -> Option<DagVertex> {
        let tips = self.get_tips().await;

        // Find the tip with the highest height
        tips.into_iter().max_by_key(|v| v.height)
    }

    /// Get the path from a vertex to the genesis
    pub async fn get_path_to_genesis(&self, hash: &[u8]) -> Result<Vec<DagVertex>> {
        let vertices = self.vertices.read().await;

        let mut path = Vec::new();
        let mut current_hash = hash.to_vec();

        while current_hash != self.genesis_hash {
            let vertex = if let Some(v) = vertices.get(&current_hash) {
                v.clone()
            } else {
                return Err(anyhow!("Vertex {} not found", hex::encode(&current_hash)));
            };

            path.push(vertex.clone());

            // Get the highest parent
            if let Some(parent_hash) = vertex
                .parents
                .iter()
                .max_by_key(|&p| vertices.get(p).map(|v| v.height).unwrap_or(0))
            {
                current_hash = parent_hash.clone();
            } else {
                return Err(anyhow!(
                    "Vertex {} has no parents",
                    hex::encode(&current_hash)
                ));
            }
        }

        // Add genesis
        if let Some(genesis) = vertices.get(&self.genesis_hash) {
            path.push(genesis.clone());
        }

        // Reverse to get genesis-to-tip order
        path.reverse();

        Ok(path)
    }

    /// Create a new vertex
    pub async fn create_vertex(
        &self,
        creator: NodeId,
        tx_refs: Vec<Vec<u8>>,
        block: Option<Block>,
    ) -> Result<DagVertex> {
        // Get the best tips to use as parents
        let tips = self.get_tips().await;
        let config = self.config.read().await;

        // Select parents (up to max_parents)
        let mut parents = Vec::new();
        let mut selected_tips = tips.into_iter().collect::<Vec<_>>();

        // Sort by height (descending)
        selected_tips.sort_by(|a, b| b.height.cmp(&a.height));

        // Take up to max_parents
        let max_parents = config.max_parents;
        for tip in selected_tips.iter().take(max_parents) {
            parents.push(tip.hash.clone());
        }

        if parents.is_empty() {
            // If no parents (should not happen except for genesis), use genesis
            parents.push(self.genesis_hash.clone());
        }

        // Calculate height (max parent height + 1)
        let mut height = 0;
        let vertices = self.vertices.read().await;
        for parent_hash in &parents {
            if let Some(parent) = vertices.get(parent_hash) {
                height = height.max(parent.height + 1);
            }
        }

        // Create hash
        let mut hasher = sha2::Sha256::new();
        use sha2::Digest;

        hasher.update(&creator.as_ref());
        for parent in &parents {
            hasher.update(parent);
        }
        for tx_ref in &tx_refs {
            hasher.update(tx_ref);
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        hasher.update(&timestamp.to_be_bytes());

        let hash = hasher.finalize().to_vec();

        // Create the vertex
        Ok(DagVertex::new(
            hash, creator, height, timestamp, parents, tx_refs, block,
        ))
    }

    /// Get all vertices
    pub async fn get_all_vertices(&self) -> Vec<DagVertex> {
        let vertices = self.vertices.read().await;
        vertices.values().cloned().collect()
    }

    /// Get statistics about the DAG
    pub async fn get_stats(&self) -> DagStats {
        let vertices = self.vertices.read().await;
        let tips = self.tips.read().await;
        let highest_finalized_height = *self.highest_finalized_height.read().await;

        let mut max_height = 0;
        let mut finalized_count = 0;
        let mut vertices_by_creator = HashMap::new();

        for vertex in vertices.values() {
            max_height = max_height.max(vertex.height);

            if vertex.finalized {
                finalized_count += 1;
            }

            *vertices_by_creator
                .entry(vertex.creator.clone())
                .or_insert(0) += 1;
        }

        DagStats {
            total_vertices: vertices.len(),
            finalized_vertices: finalized_count,
            unfinalized_vertices: vertices.len() - finalized_count,
            max_height,
            tips_count: tips.len(),
            highest_finalized_height,
            vertices_by_creator,
        }
    }

    /// Update the configuration
    pub async fn update_config(&self, config: DagConfig) {
        let mut cfg = self.config.write().await;
        *cfg = config;
    }
}

impl Clone for DagManager {
    fn clone(&self) -> Self {
        // This is a partial clone that shouldn't be used for regular operation
        // but is useful for certain interfaces
        Self {
            config: RwLock::new(self.config.try_read().unwrap_or_default().clone()),
            vertices: RwLock::new(HashMap::new()),
            finalized_by_height: RwLock::new(HashMap::new()),
            unfinalized_by_height: RwLock::new(HashMap::new()),
            tips: RwLock::new(HashSet::new()),
            genesis_hash: self.genesis_hash.clone(),
            highest_finalized_height: RwLock::new(0),
            running: RwLock::new(false),
        }
    }
}

/// Statistics about the DAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagStats {
    /// Total number of vertices
    pub total_vertices: usize,
    /// Number of finalized vertices
    pub finalized_vertices: usize,
    /// Number of unfinalized vertices
    pub unfinalized_vertices: usize,
    /// Maximum height
    pub max_height: u64,
    /// Number of tips
    pub tips_count: usize,
    /// Highest finalized height
    pub highest_finalized_height: u64,
    /// Vertices by creator
    pub vertices_by_creator: HashMap<NodeId, usize>,
}
