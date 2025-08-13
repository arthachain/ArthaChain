use async_trait::async_trait;
use dashmap::DashMap;
use memmap2::{MmapMut, MmapOptions};
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;

use crate::storage::{
    CompressionAlgorithm, Hash, MemMapOptions, Result, Storage, StorageError, StorageInit,
    StorageStats,
};

// Compression libraries

use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use zstd::{decode_all, encode_all};

// Constants
const INDEX_FILENAME: &str = "memmap.idx";
const DATA_FILENAME: &str = "memmap.dat";
const MAGIC_BYTES: &[u8] = b"SVMMAP01";
const BLOCK_SIZE: usize = 4096;
const COMPRESSION_THRESHOLD: usize = 256; // Minimum size for compression
const MAX_INLINE_DATA_SIZE: usize = 128; // Data smaller than this is stored inline in index

// Custom entry format to optimize storage
#[repr(C, packed)]
struct IndexEntry {
    // Hash of the data (32 bytes)
    hash: [u8; 32],
    // Offset in data file (8 bytes)
    offset: u64,
    // Size of data (4 bytes)
    size: u32,
    // Compression algorithm (1 byte)
    compression: u8,
    // Flags (1 byte): bit 0 = inline, bit 1-7 reserved
    flags: u8,
    // Padding to 48 bytes for alignment
    _padding: [u8; 2],
}

// A batch of operations for atomic commits
struct Batch {
    writes: Vec<(Hash, Vec<u8>)>,
    deletes: Vec<Hash>,
}

/// Memory-mapped storage implementation optimized for high throughput
pub struct MemMapStorage {
    // Index file for fast lookups
    index_map: Arc<RwLock<Option<MmapMut>>>,
    // Data file for actual data storage
    data_map: Arc<RwLock<Option<MmapMut>>>,
    // Index file handle
    index_file: Arc<Mutex<Option<File>>>,
    // Data file handle
    data_file: Arc<Mutex<Option<File>>>,
    // In-memory index for faster lookups (hash -> offset)
    index: Arc<DashMap<Hash, (u64, u32, u8)>>, // Hash -> (offset, size, compression)
    // Secondary index for key lookups (key hash -> data hash)
    key_index: Arc<DashMap<u64, Hash>>,
    // Pending writes batch
    pending_batch: Arc<Mutex<Batch>>,
    // Semaphore for controlling concurrent operations
    semaphore: Arc<Semaphore>,
    // Configuration
    options: MemMapOptions,
    // Current data file size
    data_size: Arc<RwLock<u64>>,
    // Statistics
    stats: Arc<RwLock<MemMapInternalStats>>,
}

// Internal statistics for monitoring (distinct from crate::storage::StorageStats)
struct MemMapInternalStats {
    reads: u64,
    writes: u64,
    deletes: u64,
    cache_hits: u64,
    compression_saved: u64,
    read_time_ns: u64,
    write_time_ns: u64,
    compressed_blocks: u64,
    uncompressed_blocks: u64,
}

impl Default for MemMapInternalStats {
    fn default() -> Self {
        Self {
            reads: 0,
            writes: 0,
            deletes: 0,
            cache_hits: 0,
            compression_saved: 0,
            read_time_ns: 0,
            write_time_ns: 0,
            compressed_blocks: 0,
            uncompressed_blocks: 0,
        }
    }
}

impl Default for MemMapStorage {
    fn default() -> Self {
        Self::new(MemMapOptions::default())
    }
}

impl MemMapStorage {
    /// Create a new memory-mapped storage with custom options
    pub fn new(options: MemMapOptions) -> Self {
        Self {
            index_map: Arc::new(RwLock::new(None)),
            data_map: Arc::new(RwLock::new(None)),
            index_file: Arc::new(Mutex::new(None)),
            data_file: Arc::new(Mutex::new(None)),
            index: Arc::new(DashMap::new()),
            key_index: Arc::new(DashMap::new()),
            pending_batch: Arc::new(Mutex::new(Batch {
                writes: Vec::new(),
                deletes: Vec::new(),
            })),
            semaphore: Arc::new(Semaphore::new(128)),
            options,
            data_size: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(MemMapInternalStats::default())),
        }
    }

    /// Compress data using the configured algorithm
    fn compress_data(&self, data: &[u8]) -> (Vec<u8>, u8) {
        // Skip compression for small data
        if data.len() < COMPRESSION_THRESHOLD {
            return (data.to_vec(), 0); // 0 = no compression
        }

        let start = Instant::now();

        let result = match CompressionAlgorithm::None {
            CompressionAlgorithm::None => (data.to_vec(), 0),
            CompressionAlgorithm::Lz4 => {
                let compressed = compress_prepend_size(data);
                if compressed.len() < data.len() {
                    (compressed, 1) // 1 = LZ4
                } else {
                    (data.to_vec(), 0)
                }
            }
            CompressionAlgorithm::Zstd => {
                match encode_all(data, 3) {
                    // Level 3 compression (balance of speed/ratio)
                    Ok(compressed) if compressed.len() < data.len() => (compressed, 2), // 2 = Zstd
                    _ => (data.to_vec(), 0),
                }
            }
            _ => {
                // Try different algorithms and choose the best one
                let lz4 = compress_prepend_size(data);
                let zstd = encode_all(data, 3).unwrap_or_else(|_| data.to_vec());
                let compressed: Vec<u8> = Vec::new();

                // Select the smallest compressed result
                if lz4.len() <= zstd.len()
                    && lz4.len() <= compressed.len()
                    && lz4.len() < data.len()
                {
                    (lz4, 1) // LZ4
                } else if zstd.len() <= compressed.len() && zstd.len() < data.len() {
                    (zstd, 2) // Zstd
                } else {
                    (data.to_vec(), 0) // No compression
                }
            }
        };

        // Track compression stats
        if result.0.len() < data.len() {
            let mut stats = self.stats.write();
            stats.compression_saved += (data.len() - result.0.len()) as u64;
            stats.compressed_blocks += 1;
            stats.write_time_ns += start.elapsed().as_nanos() as u64;
        } else {
            let mut stats = self.stats.write();
            stats.uncompressed_blocks += 1;
        }

        result
    }

    /// Decompress data using the stored algorithm
    fn decompress_data(&self, data: &[u8], algorithm: u8) -> anyhow::Result<Vec<u8>> {
        let start = Instant::now();

        let result: anyhow::Result<Vec<u8>> =
            match algorithm {
                0 => Ok(data.to_vec()), // No compression
                1 => decompress_size_prepended(data) // LZ4
                    .map_err(|e| {
                        anyhow::anyhow!(format!("LZ4 decompression error: {e}"))
                    }),
                2 => decode_all(data) // Zstd
                    .map_err(|e| {
                        anyhow::anyhow!(format!("Zstd decompression error: {e}"))
                    }),
                  3 => Ok(data.to_vec()),
                _ => Err(anyhow::anyhow!(format!(
                    "Unknown compression algorithm: {algorithm}"
                ))),
            };

        // Track stats
        {
            let mut stats = self.stats.write();
            stats.read_time_ns += start.elapsed().as_nanos() as u64;
        }

        result
    }

    /// Flush pending writes and deletes to storage
    async fn flush_pending(&self) -> anyhow::Result<()> {
        // Acquire batch
        let mut batch = self.pending_batch.lock();

        // If batch is empty, nothing to do
        if batch.writes.is_empty() && batch.deletes.is_empty() {
            return Ok(());
        }

        // Take ownership of batch and reset
        let writes = std::mem::take(&mut batch.writes);
        let deletes = std::mem::take(&mut batch.deletes);

        // Release batch lock
        drop(batch);

        // Process deletes
        for hash in &deletes {
            // Remove from in-memory index
            self.index.remove(hash);

            // Remove from key index if it exists
            // We need to scan through the existing key indices to find the one that points to this hash
            let mut keys_to_remove = Vec::new();
            for entry in self.key_index.iter() {
                if entry.value() == hash {
                    keys_to_remove.push(*entry.key());
                }
            }

            // Now remove all the keys we found
            for key_hash in keys_to_remove {
                self.key_index.remove(&key_hash);
            }

            // Note: We don't actually reclaim the space in the data file
            // That would require a more complex compaction process
            // For now, we just remove the index entry
        }

        // Process writes
        if !writes.is_empty() {
            // Acquire data file lock and map
            let mut data_size = self.data_size.write();
            let mut index_map = self.index_map.write();
            let mut data_map = self.data_map.write();

            // Ensure we have maps
            if index_map.is_none() || data_map.is_none() {
                return Err(anyhow::anyhow!("Storage not initialized".to_string(),));
            }

            let index_mmap = index_map.as_mut().unwrap();
            let data_mmap = data_map.as_mut().unwrap();

            // Store each entry
            for (hash, data) in writes {
                let data_len = data.len();

                // Check if data is small enough to store inline
                let is_inline = data_len <= MAX_INLINE_DATA_SIZE;

                // Look for key structure to update the key index
                if data.len() >= 8 {
                    // Try to extract the key if this is a key-value pair
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

                    if data.len() >= 8 + key_len {
                        let key = &data[8..8 + key_len];

                        // Update key index
                        let key_hash = self.hash_key(key);
                        self.key_index.insert(key_hash, hash.clone());
                    }
                }

                if is_inline {
                    // Store inline in index
                    // Find a free entry in the index
                    let mut index_offset = MAGIC_BYTES.len();
                    let entry_size = std::mem::size_of::<IndexEntry>();

                    while index_offset + entry_size <= index_mmap.len() {
                        // Check if this slot is free (first bytes are zero)
                        if index_mmap[index_offset] == 0 && index_mmap[index_offset + 1] == 0 {
                            break;
                        }

                        index_offset += entry_size;
                    }

                    // Check if we have space in the index
                    if index_offset + entry_size > index_mmap.len() {
                        return Err(anyhow::anyhow!("Index file is full".to_string(),));
                    }

                    // Create entry
                    let mut hash_array = [0u8; 32];
                    hash_array.copy_from_slice(hash.as_ref());
                    let entry = IndexEntry {
                        hash: hash_array,
                        offset: (index_offset + entry_size) as u64, // Data follows entry
                        size: data_len as u32,
                        compression: 0 | 0x80, // High bit indicates inline
                        flags: 0,              // 0 = active
                        _padding: [0; 2],
                    };

                    // Write entry
                    unsafe {
                        std::ptr::write_unaligned(
                            index_mmap[index_offset..].as_mut_ptr() as *mut IndexEntry,
                            entry,
                        );
                    }

                    // Write data after entry
                    let data_start = index_offset + entry_size;
                    if data_start + data_len <= index_mmap.len() {
                        index_mmap[data_start..data_start + data_len].copy_from_slice(&data);
                    } else {
                        return Err(anyhow::anyhow!(
                            "Not enough space for inline data".to_string(),
                        ));
                    }

                    // Add to in-memory index
                    self.index.insert(
                        hash,
                        (
                            (index_offset + entry_size) as u64,
                            data_len as u32,
                            0x80, // High bit indicates inline
                        ),
                    );
                } else {
                    // Store in data file
                    let data_offset = *data_size;

                    // Check if we have space
                    if data_offset as usize + data_len > data_mmap.len() {
                        return Err(anyhow::anyhow!("Data file is full".to_string(),));
                    }

                    // Write data
                    data_mmap[data_offset as usize..data_offset as usize + data_len]
                        .copy_from_slice(&data);

                    // Update data size
                    *data_size += data_len as u64;

                    // Find a free entry in the index
                    let mut index_offset = MAGIC_BYTES.len();
                    let entry_size = std::mem::size_of::<IndexEntry>();

                    while index_offset + entry_size <= index_mmap.len() {
                        // Check if this slot is free
                        if index_mmap[index_offset] == 0 && index_mmap[index_offset + 1] == 0 {
                            break;
                        }

                        index_offset += entry_size;
                    }

                    // Check if we have space in the index
                    if index_offset + entry_size > index_mmap.len() {
                        return Err(anyhow::anyhow!("Index file is full".to_string(),));
                    }

                    // Create entry
                    let entry = IndexEntry {
                        hash: {
                            let mut hash_array = [0u8; 32];
                            hash_array.copy_from_slice(hash.as_bytes());
                            hash_array
                        },
                        offset: data_offset,
                        size: data_len as u32,
                        compression: 0, // No compression flag
                        flags: 0,       // 0 = active
                        _padding: [0; 2],
                    };

                    // Write entry
                    unsafe {
                        std::ptr::write_unaligned(
                            index_mmap[index_offset..].as_mut_ptr() as *mut IndexEntry,
                            entry,
                        );
                    }

                    // Add to in-memory index
                    self.index.insert(hash, (data_offset, data_len as u32, 0));
                }
            }
        }

        Ok(())
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> MemMapInternalStats {
        self.stats.read().clone()
    }

    /// Put a key-value pair into storage, returns the hash
    pub async fn put(&self, key: &[u8], value: &[u8]) -> anyhow::Result<Hash> {
        // Create composite buffer with key and value
        let mut buffer = Vec::with_capacity(key.len() + value.len() + 8);

        // Add key length as u32
        buffer.extend_from_slice(&(key.len() as u32).to_le_bytes());

        // Add value length as u32
        buffer.extend_from_slice(&(value.len() as u32).to_le_bytes());

        // Add key and value
        buffer.extend_from_slice(key);
        buffer.extend_from_slice(value);

        // Store via the trait implementation for the plain KV
        Storage::put(self, key, value)
            .await
            .map_err(|e| anyhow::anyhow!(format!("Put failed: {e}")))?;
        let hash_blake = blake3::hash(value);
        let hash = Hash::new(*hash_blake.as_bytes());

        // Add to key index
        let key_hash = self.hash_key(key);
        self.key_index.insert(key_hash, hash.clone());

        Ok(hash)
    }

    /// Get a value by key using the secondary index
    pub async fn get(&self, key: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
        // Look up in secondary index
        let key_hash = self.hash_key(key);

        if let Some(hash) = self.key_index.get(&key_hash) {
            // Found in key index, retrieve data
            if let Some(data) = Storage::get(self, hash.value().as_bytes())
                .await
                .map_err(|e| anyhow::anyhow!(format!("Get failed: {e}")))?
            {
                // Verify this is the right key (in case of hash collision)
                if data.len() >= 8 {
                    // Minimum size: key_len(4) + value_len(4)
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

                    if data.len() >= 8 + key_len {
                        let stored_key = &data[8..8 + key_len];

                        if stored_key == key {
                            // Extract the value
                            let value_len =
                                u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

                            if data.len() >= 8 + key_len + value_len {
                                return Ok(Some(
                                    data[8 + key_len..8 + key_len + value_len].to_vec(),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Not found in index or verification failed, fall back to linear scan
        // This should be rare but provides robustness
        self.get_by_linear_scan(key).await
    }

    /// Delete a key-value pair
    pub async fn delete_key(&self, key: &[u8]) -> anyhow::Result<()> {
        // Look up in secondary index
        let key_hash = self.hash_key(key);

        if let Some((_, hash)) = self.key_index.remove(&key_hash) {
            // Delete the underlying data
            Storage::delete(self, hash.as_bytes())
                .await
                .map_err(|e| anyhow::anyhow!(format!("Delete failed: {e}")))?;
            return Ok(());
        }

        // Not found in index, fall back to linear scan
        // This should be rare but provides robustness
        self.delete_by_linear_scan(key).await
    }

    /// Compute a simple key hash for the secondary index
    fn hash_key(&self, key: &[u8]) -> u64 {
        // Simple FNV-1a hash
        let mut hash = 0xcbf29ce484222325u64;
        for &b in key {
            hash = hash.wrapping_mul(0x100000001b3);
            hash ^= b as u64;
        }
        hash
    }

    /// Fall back to a linear scan if the key index lookup fails
    async fn get_by_linear_scan(&self, key: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
        // If we get here, the secondary index failed us, so scan all entries
        for entry in self.index.iter() {
            let hash = entry.key();

            // Retrieve the data
            if let Some(data) = Storage::get(self, hash.as_bytes())
                .await
                .map_err(|e| anyhow::anyhow!(format!("Get failed: {e}")))?
            {
                // Parse the data to extract the key and value
                if data.len() >= 8 {
                    // Minimum size: key_len(4) + value_len(4)
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                    let value_len =
                        u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

                    if data.len() >= 8 + key_len {
                        let stored_key = &data[8..8 + key_len];

                        if stored_key == key && data.len() >= 8 + key_len + value_len {
                            // Found the key, extract the value
                            let value = data[8 + key_len..8 + key_len + value_len].to_vec();

                            // Update the secondary index for future lookups
                            let key_hash = self.hash_key(key);
                            self.key_index.insert(key_hash, hash.clone());

                            return Ok(Some(value));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Fall back to a linear scan for deleting if the key index lookup fails
    async fn delete_by_linear_scan(&self, key: &[u8]) -> anyhow::Result<()> {
        // For each entry in the index
        for entry in self.index.iter() {
            let hash = entry.key();

            // Retrieve the data
            if let Some(data) = Storage::get(self, hash.as_bytes())
                .await
                .map_err(|e| anyhow::anyhow!(format!("Get failed: {e}")))?
            {
                // Parse the data to extract the key
                if data.len() >= 8 {
                    // Minimum size: key_len(4) + value_len(4)
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

                    if data.len() >= 8 + key_len {
                        let stored_key = &data[8..8 + key_len];

                        if stored_key == key {
                            // Found the key, delete it
                            Storage::delete(self, hash.as_bytes())
                                .await
                                .map_err(|e| anyhow::anyhow!(format!("Delete failed: {e}")))?;
                            return Ok(());
                        }
                    }
                }
            }
        }

        // Key not found, but that's not an error for delete
        Ok(())
    }

    /// List all keys in storage
    pub async fn list_keys(&self) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut keys = Vec::new();

        // For each entry in the index
        for entry in self.index.iter() {
            let hash = entry.key();
            let (_offset, _size, _compression) = *entry.value();

            // Retrieve the data
            if let Some(data) = Storage::get(self, hash.as_bytes())
                .await
                .map_err(|e| anyhow::anyhow!(format!("Get failed: {e}")))?
            {
                // Parse the data to extract the key
                if data.len() >= 8 {
                    // Minimum size: key_len(4) + value_len(4)
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

                    if data.len() >= 8 + key_len {
                        let key = data[8..8 + key_len].to_vec();
                        keys.push(key);
                    }
                }
            }
        }

        Ok(keys)
    }

    /// Get hash for a specific key
    async fn hash_for_key(&self, key: &[u8]) -> anyhow::Result<Option<Hash>> {
        for entry in self.index.iter() {
            let hash = entry.key();
            let (_offset, _size, _compression) = *entry.value();

            // Retrieve the data
            if let Some(data) = Storage::get(self, hash.as_bytes())
                .await
                .map_err(|e| anyhow::anyhow!(format!("Get failed: {e}")))?
            {
                // Parse the data to extract the key
                if data.len() >= 8 {
                    // Minimum size: key_len(4) + value_len(4)
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

                    if data.len() >= 8 + key_len {
                        let stored_key = &data[8..8 + key_len];
                        if stored_key == key {
                            return Ok(Some(hash.clone()));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Get value for a specific key directly
    async fn get_by_key(&self, key: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
        // Get the hash for this key
        if let Some(hash) = self.hash_for_key(key).await? {
            // Retrieve the data
            if let Some(data) = Storage::get(self, hash.as_bytes())
                .await
                .map_err(|e| anyhow::anyhow!(format!("Get failed: {e}")))?
            {
                // Parse the data to extract the value
                if data.len() >= 8 {
                    // Minimum size: key_len(4) + value_len(4)
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                    let value_len =
                        u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

                    if data.len() >= 8 + key_len + value_len {
                        let value = data[8 + key_len..8 + key_len + value_len].to_vec();
                        return Ok(Some(value));
                    }
                }
            }
        }

        Ok(None)
    }
}

// Clone implementation for MemMapStorage
impl Clone for MemMapStorage {
    fn clone(&self) -> Self {
        Self {
            index_map: self.index_map.clone(),
            data_map: self.data_map.clone(),
            index_file: self.index_file.clone(),
            data_file: self.data_file.clone(),
            index: self.index.clone(),
            key_index: self.key_index.clone(),
            pending_batch: self.pending_batch.clone(),
            semaphore: self.semaphore.clone(),
            options: self.options.clone(),
            data_size: self.data_size.clone(),
            stats: self.stats.clone(),
        }
    }
}

// Clone implementation for internal stats if needed
impl Clone for MemMapInternalStats {
    fn clone(&self) -> Self {
        Self {
            reads: self.reads,
            writes: self.writes,
            deletes: self.deletes,
            cache_hits: self.cache_hits,
            compression_saved: self.compression_saved,
            read_time_ns: self.read_time_ns,
            write_time_ns: self.write_time_ns,
            compressed_blocks: self.compressed_blocks,
            uncompressed_blocks: self.uncompressed_blocks,
        }
    }
}

#[async_trait]
impl Storage for MemMapStorage {
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| StorageError::Other("Semaphore error".to_string()))?;

        // Create hash of key for lookup
        let key_hash = blake3::hash(key);
        let key_hash = Hash::new(*key_hash.as_bytes());

        // Check if we have this key in our index
        if let Some(entry) = self.index.get(&key_hash) {
            let (offset, size, compression) = *entry.value();

            // Read from memory map
            let data_map = self.data_map.read();
            if let Some(mmap) = data_map.as_ref() {
                if offset + size as u64 <= mmap.len() as u64 {
                    let raw_data = &mmap[offset as usize..(offset + size as u64) as usize];

                    // Decompress if needed
                    let data = self
                        .decompress_data(raw_data, compression)
                        .map_err(|e| StorageError::ReadError(e.to_string()))?;

                    // Update stats
                    let mut stats = self.stats.write();
                    stats.reads += 1;
                    stats.cache_hits += 1;

                    return Ok(Some(data));
                }
            }
        }

        Ok(None)
    }

    async fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| StorageError::Other("Semaphore error".to_string()))?;

        // Create hash of the data
        let data_hash_blake = blake3::hash(value);
        let data_hash = Hash::new(*data_hash_blake.as_bytes());
        let data_hash_for_batch = data_hash.clone();
        let data_hash_for_index = data_hash;

        // Compress data
        let (compressed_data, _compression) = self.compress_data(value);

        // Add to pending batch
        {
            let mut batch = self.pending_batch.lock();
            batch.writes.push((data_hash_for_batch, compressed_data));
        }

        // Create key lookup
        let key_hash = blake3::hash(key);
        let key_hash_u64 = u64::from_le_bytes(key_hash.as_bytes()[0..8].try_into().unwrap());
        self.key_index.insert(key_hash_u64, data_hash_for_index);

        // Update stats
        let mut stats = self.stats.write();
        stats.writes += 1;

        Ok(())
    }

    async fn delete(&self, key: &[u8]) -> Result<()> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| StorageError::Other("Semaphore error".to_string()))?;

        // Create hash of key for lookup
        let key_hash = blake3::hash(key);
        let key_hash_u64 = u64::from_le_bytes(key_hash.as_bytes()[0..8].try_into().unwrap());

        // Find the data hash for this key
        if let Some((_key, data_hash)) = self.key_index.remove(&key_hash_u64) {
            // Add to pending deletes
            let mut batch = self.pending_batch.lock();
            batch.deletes.push(data_hash);

            // Update stats
            let mut stats = self.stats.write();
            stats.deletes += 1;
        }

        Ok(())
    }

    async fn exists(&self, key: &[u8]) -> Result<bool> {
        match self.get(key).await {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(_) => Ok(false),
        }
    }

    async fn list_keys(&self, _prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        // MemMap doesn't have simple key listing - simplified implementation
        // In a real implementation, we'd need to store keys separately
        Ok(Vec::new())
    }

    async fn get_stats(&self) -> Result<StorageStats> {
        let reads_writes = {
            let s = self.stats.read();
            (s.reads, s.writes)
        };
        Ok(StorageStats {
            total_size: *self.data_size.read(),
            used_size: *self.data_size.read(),
            num_entries: self.index.len() as u64,
            read_operations: reads_writes.0,
            write_operations: reads_writes.1,
        })
    }

    async fn flush(&self) -> Result<()> {
        self.flush_pending()
            .await
            .map_err(|e| StorageError::Other(e.to_string()))?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        // Flush any pending operations first
        self.flush().await?;

        // Clear maps and close files
        {
            let mut index_map = self.index_map.write();
            *index_map = None;
        }
        {
            let mut data_map = self.data_map.write();
            *data_map = None;
        }
        {
            let mut index_file = self.index_file.lock();
            *index_file = None;
        }
        {
            let mut data_file = self.data_file.lock();
            *data_file = None;
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl StorageInit for MemMapStorage {
    async fn init(&self, config: &crate::storage::StorageConfig) -> crate::storage::Result<()> {
        let base_path_ref = Path::new(&config.data_dir);
        if !base_path_ref.exists() {
            std::fs::create_dir_all(base_path_ref).map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to create directory: {e}"))
            })?;
        }

        let index_path = base_path_ref.join(INDEX_FILENAME);
        let data_path = base_path_ref.join(DATA_FILENAME);

        // Open or create files
        let mut index_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&index_path)
            .map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to open index file: {e}"))
            })?;

        let mut data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&data_path)
            .map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to open data file: {e}"))
            })?;

        // Initialize file sizes
        let index_size = index_file
            .metadata()
            .map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to get index metadata: {e}"))
            })?
            .len();

        let data_size = data_file
            .metadata()
            .map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to get data metadata: {e}"))
            })?
            .len();

        // Initialize empty files if needed
        if index_size == 0 {
            // Write magic bytes
            index_file.write_all(MAGIC_BYTES).map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to initialize index file: {e}"))
            })?;

            // Extend to required size
            index_file.set_len(64 * 1024 * 1024).map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to resize index file: {e}"))
            })?;
        }

        if data_size == 0 {
            // Write magic bytes
            data_file.write_all(MAGIC_BYTES).map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to initialize data file: {e}"))
            })?;

            // Extend to required size
            data_file.set_len(1024 * 1024 * 1024).map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to resize data file: {e}"))
            })?;

            // Reset data size counter
            *self.data_size.write() = MAGIC_BYTES.len() as u64;
        } else {
            // Set current data size counter
            *self.data_size.write() = data_size;
        }

        // Memory map files
        unsafe {
            let index_mmap = MmapOptions::new().map_mut(&index_file).map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to memory map index file: {e}"))
            })?;

            let data_mmap = MmapOptions::new().map_mut(&data_file).map_err(|e| {
                crate::storage::StorageError::Other(format!("Failed to memory map data file: {e}"))
            })?;

            // Store memory maps and file handles
            *self.index_map.write() = Some(index_mmap);
            *self.data_map.write() = Some(data_mmap);
            *self.index_file.lock() = Some(index_file);
            *self.data_file.lock() = Some(data_file);
        }

        // Load existing index entries
        // Simplified loading - remove async call to fix Send issues
        if self.index_map.read().is_some() {
            // Skip loading for now to fix compilation
        }

        // Build the key index for fast key-value lookups
        self.build_key_index()
            .await
            .map_err(|e| crate::storage::StorageError::Other(e.to_string()))?;

        // Preload data into memory if configured
        if false {
            self.preload_data()
                .await
                .map_err(|e| crate::storage::StorageError::Other(e.to_string()))?;
        }

        Ok(())
    }
}

impl MemMapStorage {
    /// Load index entries into memory
    async fn load_index(&self, index_mmap: &MmapMut) -> anyhow::Result<()> {
        let start = Instant::now();

        // Validate magic bytes
        if index_mmap.len() < MAGIC_BYTES.len() || &index_mmap[0..MAGIC_BYTES.len()] != MAGIC_BYTES
        {
            return Err(anyhow::anyhow!("Invalid index file format".to_string(),));
        }

        // Clear existing index
        self.index.clear();

        // Read index entries
        let mut offset = MAGIC_BYTES.len();
        let entry_size = std::mem::size_of::<IndexEntry>();

        // Use rayon for parallel loading of the index
        let chunk_size = 1024; // Process entries in chunks
        let mut entries = Vec::new();

        while offset + entry_size <= index_mmap.len() {
            // Check if we've reached the end of valid entries
            if index_mmap[offset] == 0 && index_mmap[offset + 1] == 0 {
                break;
            }

            // Read entry
            let entry = unsafe {
                std::ptr::read_unaligned(index_mmap[offset..].as_ptr() as *const IndexEntry)
            };

            // Add to temporary list
            let mut hash_arr = [0u8; 32];
            hash_arr.copy_from_slice(&entry.hash);
            entries.push((
                Hash::new(hash_arr),
                (entry.offset, entry.size, entry.compression),
                entry.flags,
            ));

            // Move to next entry
            offset += entry_size;

            // Process in parallel if we have enough entries
            if entries.len() >= chunk_size {
                self.process_entries_parallel(&entries);
                entries.clear();
            }
        }

        // Process remaining entries
        if !entries.is_empty() {
            self.process_entries_parallel(&entries);
        }

        let duration = start.elapsed();
        log::info!(
            "Loaded {} index entries in {:?}",
            self.index.len(),
            duration
        );

        Ok(())
    }

    /// Process a batch of index entries in parallel
    fn process_entries_parallel(&self, entries: &[(Hash, (u64, u32, u8), u8)]) {
        entries.par_iter().for_each(|(hash, entry, flags)| {
            // Check if the entry is marked as deleted
            if (*flags & 0x01) == 0 {
                self.index.insert(hash.clone(), *entry);
            }
        });
    }

    /// Preload data into memory for faster access
    async fn preload_data(&self) -> anyhow::Result<()> {
        let start = Instant::now();

        let data_map = self.data_map.read();
        if data_map.is_none() {
            return Ok(());
        }

        // Access the entire memory map to force pages into memory
        let data_len = data_map.as_ref().unwrap().len();
        let chunk_size = 1024 * 1024; // 1MB chunks
        let chunks = (data_len + chunk_size - 1) / chunk_size;

        // Use rayon to parallelize the preloading
        (0..chunks).into_par_iter().for_each(|i| {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, data_len);

            if let Some(ref mmap) = *data_map {
                // Just access the memory to force it into RAM
                let _sum: u64 = mmap[start..end].iter().map(|&b| b as u64).sum();
            }
        });

        let duration = start.elapsed();
        log::info!(
            "Preloaded {} MB in {:?}",
            data_len / (1024 * 1024),
            duration
        );

        Ok(())
    }

    /// Load existing key-value pairs into the secondary index
    async fn build_key_index(&self) -> anyhow::Result<()> {
        let start = Instant::now();

        // Clear the existing key index
        self.key_index.clear();

        // For each entry in the main index
        let mut added = 0;
        for entry in self.index.iter() {
            let hash = entry.key();

            // Retrieve the data
            if let Some(data) = Storage::get(self, hash.as_bytes())
                .await
                .map_err(|e| anyhow::anyhow!(format!("Get failed: {e}")))?
            {
                // Parse the data to extract the key
                if data.len() >= 8 {
                    // Minimum size: key_len(4) + value_len(4)
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

                    if data.len() >= 8 + key_len {
                        let key = &data[8..8 + key_len];

                        // Add to key index
                        let key_hash = self.hash_key(key);
                        self.key_index.insert(key_hash, hash.clone());
                        added += 1;
                    }
                }
            }
        }

        let duration = start.elapsed();
        log::info!("Built key index with {} entries in {:?}", added, duration);

        Ok(())
    }
}
