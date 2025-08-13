use anyhow::Result;
use log::{debug, info, warn};
use rand::SeedableRng;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::types::{AccountId, BlockHeight};
use crate::utils::crypto::quantum_resistant_hash;
use rand::seq::SliceRandom;

/// Cache entry with metadata and expiration
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// The actual value being cached
    pub value: T,
    /// When this entry was last accessed
    pub last_accessed: Instant,
    /// When this entry expires
    pub expires_at: Option<Instant>,
    /// How many times this entry has been accessed
    pub access_count: u64,
    /// Size of the entry in bytes (estimated)
    pub size_bytes: usize,
    /// Hash of the entry for integrity checking
    pub integrity_hash: Vec<u8>,
}

impl<T: Clone + serde::Serialize> CacheEntry<T> {
    /// Create a new cache entry
    pub fn new(
        value: T,
        ttl: Option<Duration>,
        size_bytes: usize,
        use_quantum_hash: bool,
    ) -> Result<Self> {
        let now = Instant::now();
        let expires_at = ttl.map(|ttl| now + ttl);

        // Serialize the value for hashing
        let serialized = bincode::serialize(&value)?;

        // Generate hash for integrity checking
        let integrity_hash = if use_quantum_hash {
            quantum_resistant_hash(&serialized)?
        } else {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&serialized);
            hasher.finalize().to_vec()
        };

        Ok(Self {
            value,
            last_accessed: now,
            expires_at,
            access_count: 1,
            size_bytes,
            integrity_hash,
        })
    }

    /// Check if entry has expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Instant::now() > expires_at
        } else {
            false
        }
    }

    /// Update last accessed time and increment access count
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Extend TTL by the specified duration
    pub fn extend_ttl(&mut self, duration: Duration) {
        let now = Instant::now();
        self.expires_at = Some(self.expires_at.map_or(now + duration, |exp| exp + duration));
    }

    /// Verify integrity of the entry
    pub fn verify_integrity(&self, use_quantum_hash: bool) -> Result<bool> {
        // Serialize the value for hashing
        let serialized = bincode::serialize(&self.value)?;

        // Generate hash for comparison
        let hash = if use_quantum_hash {
            quantum_resistant_hash(&serialized)?
        } else {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&serialized);
            hasher.finalize().to_vec()
        };

        Ok(hash == self.integrity_hash)
    }
}

/// Cache eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random Replacement
    Random,
    /// Time-aware Least Recently Used
    TLRU,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum size in bytes
    pub max_size_bytes: usize,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Default TTL for entries
    pub default_ttl: Option<Duration>,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Whether to use quantum-resistant hashing
    pub use_quantum_hash: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Whether to verify integrity on read
    pub verify_integrity: bool,
    /// How often to refresh frequently accessed items
    pub refresh_interval: Option<Duration>,
    /// Threshold for considering an item as frequently accessed
    pub hot_access_threshold: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 100 * 1024 * 1024, // 100MB
            max_entries: 10_000,
            default_ttl: Some(Duration::from_secs(3600)), // 1 hour
            eviction_policy: EvictionPolicy::LRU,
            use_quantum_hash: true,
            cleanup_interval: Duration::from_secs(60),
            verify_integrity: true,
            refresh_interval: Some(Duration::from_secs(300)), // 5 minutes
            hot_access_threshold: 10,
        }
    }
}

/// Quantum-resistant caching system for blockchain state
pub struct QuantumCache<K, V>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
    V: Clone + serde::Serialize + Send + Sync,
{
    /// Cache configuration
    config: CacheConfig,
    /// Main storage of cached items
    cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    /// Access order for LRU eviction
    lru_queue: Arc<RwLock<VecDeque<K>>>,
    /// Insertion order for FIFO eviction
    fifo_queue: Arc<RwLock<VecDeque<K>>>,
    /// Set of frequently accessed keys
    hot_keys: Arc<RwLock<HashSet<K>>>,
    /// Current size in bytes
    current_size: Arc<RwLock<usize>>,
    /// Last cleanup time
    last_cleanup: Arc<RwLock<Instant>>,
    /// Last time hot keys were refreshed
    last_refresh: Arc<RwLock<Instant>>,
    /// Statistics for cache performance
    stats: Arc<RwLock<CacheStats>>,
}

impl<K, V> QuantumCache<K, V>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + serde::Serialize + Send + Sync + 'static,
{
    /// Create a new cache
    pub fn new(config: CacheConfig) -> Self {
        let now = Instant::now();

        Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            lru_queue: Arc::new(RwLock::new(VecDeque::new())),
            fifo_queue: Arc::new(RwLock::new(VecDeque::new())),
            hot_keys: Arc::new(RwLock::new(HashSet::new())),
            current_size: Arc::new(RwLock::new(0)),
            last_cleanup: Arc::new(RwLock::new(now)),
            last_refresh: Arc::new(RwLock::new(now)),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Put value in cache
    pub async fn put(&self, key: K, value: V, size_bytes: Option<usize>) -> Result<()> {
        // Estimate size if not provided
        let size = size_bytes
            .unwrap_or_else(|| bincode::serialize(&value).map(|v| v.len()).unwrap_or(1024));

        // Create cache entry
        let entry = CacheEntry::new(
            value,
            self.config.default_ttl,
            size,
            self.config.use_quantum_hash,
        )?;

        // Check if we need to perform eviction
        self.maybe_evict(size).await?;

        // Insert the entry
        {
            let mut cache = self.cache.write().await;
            let mut current_size = self.current_size.write().await;

            // If key already exists, update size calculation
            if let Some(existing) = cache.get(&key) {
                *current_size = current_size.saturating_sub(existing.size_bytes);
            }

            // Update bookkeeping for eviction policies
            match self.config.eviction_policy {
                EvictionPolicy::LRU => {
                    let mut lru_queue = self.lru_queue.write().await;
                    // Remove if exists to avoid duplicates
                    lru_queue.retain(|k| k != &key);
                    // Add to end of queue (most recently used)
                    lru_queue.push_back(key.clone());
                }
                EvictionPolicy::FIFO => {
                    let mut fifo_queue = self.fifo_queue.write().await;
                    // Only add if not already in queue
                    if !fifo_queue.contains(&key) {
                        fifo_queue.push_back(key.clone());
                    }
                }
                EvictionPolicy::TLRU => {
                    let mut lru_queue = self.lru_queue.write().await;
                    // Remove if exists to avoid duplicates
                    lru_queue.retain(|k| k != &key);
                    // Add to end of queue (most recently used)
                    lru_queue.push_back(key.clone());
                }
                _ => {}
            }

            // Insert entry and update size
            cache.insert(key, entry);
            *current_size += size;

            // Update stats
            let mut stats = self.stats.write().await;
            stats.puts += 1;
            stats.current_entries = cache.len();
            stats.current_size_bytes = *current_size;
        }

        // Maybe run cleanup
        self.maybe_cleanup().await?;

        Ok(())
    }

    /// Get value from cache
    pub async fn get(&self, key: &K) -> Option<V> {
        let result = {
            let mut cache = self.cache.write().await;

            if let Some(entry) = cache.get_mut(key) {
                // Check if expired
                if entry.is_expired() {
                    // Update stats for miss due to expiration
                    let mut stats = self.stats.write().await;
                    stats.misses += 1;
                    stats.expired_misses += 1;

                    return None;
                }

                // Verify integrity if configured
                if self.config.verify_integrity {
                    match entry.verify_integrity(self.config.use_quantum_hash) {
                        Ok(valid) => {
                            if !valid {
                                // Integrity check failed
                                warn!("Cache integrity check failed for key");

                                // Update stats
                                let mut stats = self.stats.write().await;
                                stats.misses += 1;
                                stats.integrity_failures += 1;

                                return None;
                            }
                        }
                        Err(_) => {
                            // Error during integrity check
                            warn!("Error during cache integrity check");

                            // Update stats
                            let mut stats = self.stats.write().await;
                            stats.misses += 1;
                            stats.integrity_failures += 1;

                            return None;
                        }
                    }
                }

                // Update access metadata
                entry.touch();

                // Check if this entry should be marked as hot
                if entry.access_count >= self.config.hot_access_threshold {
                    let mut hot_keys = self.hot_keys.write().await;
                    hot_keys.insert(key.clone());
                }

                // Update LRU queue if using LRU policy
                if self.config.eviction_policy == EvictionPolicy::LRU
                    || self.config.eviction_policy == EvictionPolicy::TLRU
                {
                    let mut lru_queue = self.lru_queue.write().await;
                    lru_queue.retain(|k| k != key);
                    lru_queue.push_back(key.clone());
                }

                // Update stats
                let mut stats = self.stats.write().await;
                stats.hits += 1;

                // Return cloned value
                Some(entry.value.clone())
            } else {
                // Not found
                let mut stats = self.stats.write().await;
                stats.misses += 1;

                None
            }
        };

        // Maybe refresh hot entries
        self.maybe_refresh_hot_entries().await;

        result
    }

    /// Remove an entry from the cache
    pub async fn remove(&self, key: &K) -> bool {
        let result = {
            let mut cache = self.cache.write().await;
            let mut current_size = self.current_size.write().await;

            if let Some(entry) = cache.remove(key) {
                // Update size
                *current_size = current_size.saturating_sub(entry.size_bytes);

                // Update eviction queues
                match self.config.eviction_policy {
                    EvictionPolicy::LRU | EvictionPolicy::TLRU => {
                        let mut lru_queue = self.lru_queue.write().await;
                        lru_queue.retain(|k| k != key);
                    }
                    EvictionPolicy::FIFO => {
                        let mut fifo_queue = self.fifo_queue.write().await;
                        fifo_queue.retain(|k| k != key);
                    }
                    _ => {}
                }

                // Remove from hot keys
                let mut hot_keys = self.hot_keys.write().await;
                hot_keys.remove(key);

                // Update stats
                let mut stats = self.stats.write().await;
                stats.removes += 1;
                stats.current_entries = cache.len();
                stats.current_size_bytes = *current_size;

                true
            } else {
                false
            }
        };

        result
    }

    /// Check if key exists in cache
    pub async fn contains(&self, key: &K) -> bool {
        let cache = self.cache.read().await;
        if let Some(entry) = cache.get(key) {
            !entry.is_expired()
        } else {
            false
        }
    }

    /// Clear all entries from the cache
    pub async fn clear(&self) {
        // Clear all data structures
        {
            let mut cache = self.cache.write().await;
            cache.clear();
        }

        {
            let mut lru_queue = self.lru_queue.write().await;
            lru_queue.clear();
        }

        {
            let mut fifo_queue = self.fifo_queue.write().await;
            fifo_queue.clear();
        }

        {
            let mut hot_keys = self.hot_keys.write().await;
            hot_keys.clear();
        }

        {
            let mut current_size = self.current_size.write().await;
            *current_size = 0;
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.clears += 1;
            stats.current_entries = 0;
            stats.current_size_bytes = 0;
        }

        info!("Cache cleared");
    }

    /// Evict entries based on policy
    async fn evict(&self, needed_space: usize) -> Result<usize> {
        let mut evicted = 0;
        let mut space_freed = 0;

        match self.config.eviction_policy {
            EvictionPolicy::LRU => {
                let mut to_remove = Vec::new();

                // Identify items to remove using LRU queue
                {
                    let lru_queue = self.lru_queue.read().await;
                    let cache = self.cache.read().await;

                    for key in lru_queue.iter() {
                        if let Some(entry) = cache.get(key) {
                            to_remove.push((key.clone(), entry.size_bytes));
                            space_freed += entry.size_bytes;

                            if space_freed >= needed_space {
                                break;
                            }
                        }
                    }
                }

                // Remove the identified items
                for (key, _) in to_remove {
                    self.remove(&key).await;
                    evicted += 1;
                }
            }
            EvictionPolicy::LFU => {
                let mut to_remove = Vec::new();

                // Find least frequently used items
                {
                    let cache = self.cache.read().await;

                    // Sort by access count
                    let mut entries: Vec<_> = cache.iter().collect();
                    entries.sort_by_key(|(_, entry)| entry.access_count);

                    for (key, entry) in entries {
                        to_remove.push((key.clone(), entry.size_bytes));
                        space_freed += entry.size_bytes;

                        if space_freed >= needed_space {
                            break;
                        }
                    }
                }

                // Remove the identified items
                for (key, _) in to_remove {
                    self.remove(&key).await;
                    evicted += 1;
                }
            }
            EvictionPolicy::FIFO => {
                let mut to_remove = Vec::new();

                // Identify items to remove using FIFO queue
                {
                    let fifo_queue = self.fifo_queue.read().await;
                    let cache = self.cache.read().await;

                    for key in fifo_queue.iter() {
                        if let Some(entry) = cache.get(key) {
                            to_remove.push((key.clone(), entry.size_bytes));
                            space_freed += entry.size_bytes;

                            if space_freed >= needed_space {
                                break;
                            }
                        }
                    }
                }

                // Remove the identified items
                for (key, _) in to_remove {
                    self.remove(&key).await;
                    evicted += 1;
                }
            }
            EvictionPolicy::Random => {
                let mut to_remove = Vec::new();

                // Randomly select items to remove
                {
                    let cache = self.cache.read().await;
                    let mut entries: Vec<_> = cache.iter().collect();

                    // Simple shuffle
                    use std::time::SystemTime;
                    let seed = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();

                    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                    use rand::seq::SliceRandom;
                    entries.shuffle(&mut rng);

                    for (key, entry) in entries {
                        to_remove.push((key.clone(), entry.size_bytes));
                        space_freed += entry.size_bytes;

                        if space_freed >= needed_space {
                            break;
                        }
                    }
                }

                // Remove the identified items
                for (key, _) in to_remove {
                    self.remove(&key).await;
                    evicted += 1;
                }
            }
            EvictionPolicy::TLRU => {
                // Time-aware LRU evicts items based on a combination of
                // recency and TTL status
                let mut to_remove = Vec::new();

                {
                    let cache = self.cache.read().await;
                    let mut entries: Vec<_> = cache.iter().collect();

                    // Sort by TTL status (expired first) and then by last accessed
                    entries.sort_by(|(_, a), (_, b)| match (a.is_expired(), b.is_expired()) {
                        (true, false) => std::cmp::Ordering::Less,
                        (false, true) => std::cmp::Ordering::Greater,
                        _ => a.last_accessed.cmp(&b.last_accessed),
                    });

                    for (key, entry) in entries {
                        to_remove.push((key.clone(), entry.size_bytes));
                        space_freed += entry.size_bytes;

                        if space_freed >= needed_space {
                            break;
                        }
                    }
                }

                // Remove the identified items
                for (key, _) in to_remove {
                    self.remove(&key).await;
                    evicted += 1;
                }
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.evictions += evicted as u64;
        }

        if evicted > 0 {
            debug!("Evicted {} entries, freed {} bytes", evicted, space_freed);
        }

        Ok(evicted)
    }

    /// Check if eviction is needed and perform it
    async fn maybe_evict(&self, needed_space: usize) -> Result<()> {
        let current_size = *self.current_size.read().await;
        let cache_len = self.cache.read().await.len();

        if current_size + needed_space > self.config.max_size_bytes
            || cache_len >= self.config.max_entries
        {
            // Calculate how much space we need to free
            let space_to_free = if current_size + needed_space > self.config.max_size_bytes {
                (current_size + needed_space) - self.config.max_size_bytes
            } else {
                needed_space
            };

            self.evict(space_to_free).await?;
        }

        Ok(())
    }

    /// Clean up expired entries
    async fn cleanup_expired(&self) -> Result<usize> {
        let mut to_remove = Vec::new();

        // Find expired entries
        {
            let cache = self.cache.read().await;

            for (key, entry) in cache.iter() {
                if entry.is_expired() {
                    to_remove.push(key.clone());
                }
            }
        }

        // Remove expired entries
        let mut removed = 0usize;
        for key in to_remove {
            if self.remove(&key).await {
                removed += 1;
            }
        }

        // Update last cleanup time
        *self.last_cleanup.write().await = Instant::now();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.expired_cleanups += removed as u64;
        }

        if removed > 0 {
            debug!("Cleaned up {} expired entries", removed);
        }

        Ok(removed)
    }

    /// Check if it's time to clean up and do it
    async fn maybe_cleanup(&self) -> Result<()> {
        let last = *self.last_cleanup.read().await;
        let now = Instant::now();

        if now.duration_since(last) >= self.config.cleanup_interval {
            self.cleanup_expired().await?;
        }

        Ok(())
    }

    /// Refresh hot entries (frequently accessed)
    async fn refresh_hot_entries(&self) -> Result<usize> {
        let mut hot_entries = Vec::new();

        // Collect hot entries
        {
            let hot_keys = self.hot_keys.read().await;
            let cache = self.cache.read().await;

            for key in hot_keys.iter() {
                if let Some(entry) = cache.get(key) {
                    if !entry.is_expired() {
                        hot_entries.push((key.clone(), entry.clone()));
                    }
                }
            }
        }

        // Extend TTL for hot entries
        let mut refreshed = 0usize;
        for (key, mut entry) in hot_entries {
            if let Some(ttl) = self.config.default_ttl {
                entry.extend_ttl(ttl);

                // Update the entry
                let mut cache = self.cache.write().await;
                if let Some(existing) = cache.get_mut(&key) {
                    existing.expires_at = entry.expires_at;
                    refreshed += 1;
                }
            }
        }

        // Update last refresh time
        *self.last_refresh.write().await = Instant::now();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.refreshes += refreshed as u64;
        }

        if refreshed > 0 {
            debug!("Refreshed {} hot entries", refreshed);
        }

        Ok(refreshed)
    }

    /// Check if it's time to refresh hot entries
    async fn maybe_refresh_hot_entries(&self) {
        if let Some(interval) = self.config.refresh_interval {
            let last = *self.last_refresh.read().await;
            let now = Instant::now();

            if now.duration_since(last) >= interval {
                if let Err(e) = self.refresh_hot_entries().await {
                    warn!("Error refreshing hot entries: {}", e);
                }
            }
        }
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
}

/// Account state cache for the blockchain
pub struct AccountStateCache {
    /// Inner cache implementation
    inner: QuantumCache<AccountId, AccountState>,
}

impl AccountStateCache {
    /// Create a new account state cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            inner: QuantumCache::new(config),
        }
    }

    /// Get account state
    pub async fn get_account_state(&self, account_id: &AccountId) -> Option<AccountState> {
        self.inner.get(account_id).await
    }

    /// Update account state
    pub async fn update_account_state(
        &self,
        account_id: AccountId,
        state: AccountState,
    ) -> Result<()> {
        self.inner.put(account_id, state, None).await
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.inner.get_stats().await
    }
}

/// Block cache for the blockchain
pub struct BlockCache {
    /// Inner cache implementation
    inner: QuantumCache<BlockHeight, BlockData>,
}

impl BlockCache {
    /// Create a new block cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            inner: QuantumCache::new(config),
        }
    }

    /// Get block data
    pub async fn get_block(&self, height: &BlockHeight) -> Option<BlockData> {
        self.inner.get(height).await
    }

    /// Cache block data
    pub async fn cache_block(&self, height: BlockHeight, data: BlockData) -> Result<()> {
        self.inner.put(height, data, None).await
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        self.inner.get_stats().await
    }
}

/// Generic state for an account
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AccountState {
    /// Account balance
    pub balance: u64,
    /// Account nonce
    pub nonce: u64,
    /// Account storage (key-value pairs)
    pub storage: HashMap<String, Vec<u8>>,
    /// Account code (for smart contracts)
    pub code: Option<Vec<u8>>,
    /// Last updated block height
    pub last_updated: BlockHeight,
}

/// Block data for caching
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockData {
    /// Block height
    pub height: BlockHeight,
    /// Block hash
    pub hash: Vec<u8>,
    /// Block timestamp
    pub timestamp: u64,
    /// Whether block is finalized
    pub finalized: bool,
    /// Transaction count
    pub transaction_count: usize,
    /// Block size in bytes
    pub size_bytes: usize,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of puts
    pub puts: u64,
    /// Number of removes
    pub removes: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Number of expired entries cleaned up
    pub expired_cleanups: u64,
    /// Number of integrity check failures
    pub integrity_failures: u64,
    /// Number of cache clears
    pub clears: u64,
    /// Number of expired misses
    pub expired_misses: u64,
    /// Number of hot entries refreshed
    pub refreshes: u64,
    /// Current number of entries
    pub current_entries: usize,
    /// Current size in bytes
    pub current_size_bytes: usize,
}

impl CacheStats {
    /// Calculate hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_cache_operations() {
        let config = CacheConfig::default();
        let cache: QuantumCache<String, String> = QuantumCache::new(config);

        // Put a value
        cache
            .put("key1".to_string(), "value1".to_string(), None)
            .await
            .unwrap();

        // Get the value
        let value = cache.get(&"key1".to_string()).await.unwrap();
        assert_eq!(value, "value1");

        // Check contains
        assert!(cache.contains(&"key1".to_string()).await);

        // Remove the value
        assert!(cache.remove(&"key1".to_string()).await);

        // Verify it's gone
        assert!(!cache.contains(&"key1".to_string()).await);
        assert_eq!(cache.get(&"key1".to_string()).await, None);
    }

    #[tokio::test]
    async fn test_ttl_expiration() {
        let config = CacheConfig {
            default_ttl: Some(Duration::from_millis(10)),
            ..CacheConfig::default()
        };
        let cache: QuantumCache<String, String> = QuantumCache::new(config);

        // Put a value
        cache
            .put("key1".to_string(), "value1".to_string(), None)
            .await
            .unwrap();

        // Verify it exists immediately
        assert!(cache.contains(&"key1".to_string()).await);

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Verify it's expired
        assert!(!cache.contains(&"key1".to_string()).await);
        assert_eq!(cache.get(&"key1".to_string()).await, None);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let mut config = CacheConfig::default();
        config.max_entries = 2;
        config.eviction_policy = EvictionPolicy::LRU;
        let cache: QuantumCache<String, String> = QuantumCache::new(config);

        // Add three entries, which should evict the first one
        cache
            .put("key1".to_string(), "value1".to_string(), None)
            .await
            .unwrap();
        cache
            .put("key2".to_string(), "value2".to_string(), None)
            .await
            .unwrap();

        // Access key1 to make it most recently used
        cache.get(&"key1".to_string()).await;

        // Add key3, which should evict key2 (least recently used)
        cache
            .put("key3".to_string(), "value3".to_string(), None)
            .await
            .unwrap();

        // Verify key1 and key3 exist, but key2 is gone
        assert!(cache.contains(&"key1".to_string()).await);
        assert!(!cache.contains(&"key2".to_string()).await);
        assert!(cache.contains(&"key3".to_string()).await);
    }

    #[tokio::test]
    async fn test_account_state_cache() {
        let config = CacheConfig::default();
        let cache = AccountStateCache::new(config);

        // Create account state
        let account_id = crate::types::AccountId::from("account1");
        let state = AccountState {
            balance: 1000,
            nonce: 5,
            storage: HashMap::new(),
            code: None,
            last_updated: 100,
        };

        // Update cache
        cache
            .update_account_state(account_id.clone(), state)
            .await
            .unwrap();

        // Retrieve state
        let retrieved = cache.get_account_state(&account_id).await.unwrap();
        assert_eq!(retrieved.balance, 1000);
        assert_eq!(retrieved.nonce, 5);

        // Check stats
        let stats = cache.get_stats().await;
        assert_eq!(stats.puts, 1);
        assert_eq!(stats.hits, 1);
    }

    #[tokio::test]
    async fn test_block_cache() {
        let config = CacheConfig::default();
        let cache = BlockCache::new(config);

        // Create block data
        let height = 100;
        let data = BlockData {
            height,
            hash: vec![1, 2, 3, 4],
            timestamp: 12345,
            finalized: true,
            transaction_count: 10,
            size_bytes: 1024,
        };

        // Cache block
        cache.cache_block(height, data).await.unwrap();

        // Retrieve block
        let retrieved = cache.get_block(&height).await.unwrap();
        assert_eq!(retrieved.height, height);
        assert_eq!(retrieved.transaction_count, 10);

        // Check stats
        let stats = cache.get_stats().await;
        assert_eq!(stats.puts, 1);
        assert_eq!(stats.hits, 1);
    }
}
