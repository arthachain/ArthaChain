use anyhow::Context;
use async_trait::async_trait;
use dashmap::DashMap;
use memmap2::{MmapMut, MmapOptions};
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tokio::task;

use crate::storage::{
    CompressionAlgorithm, Hash, MemMapOptions, Result as StorageResult, Storage, StorageError,
    StorageInit,
};

// Compression libraries
use brotli::{CompressorReader, Decompressor};
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
    stats: Arc<RwLock<StorageStats>>,
}

// Storage statistics for monitoring
struct StorageStats {
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

impl Default for StorageStats {
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
            stats: Arc::new(RwLock::new(StorageStats::default())),
        }
    }

    /// Compress data using the configured algorithm
    fn compress_data(&self, data: &[u8]) -> (Vec<u8>, u8) {
        // Skip compression for small data
        if data.len() < COMPRESSION_THRESHOLD {
            return (data.to_vec(), 0); // 0 = no compression
        }

        let start = Instant::now();

        let result = match self.options.compression_algorithm {
            CompressionAlgorithm::None => (data.to_vec(), 0),
            CompressionAlgorithm::LZ4 => {
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
            CompressionAlgorithm::Brotli => {
                let mut compressed = Vec::new();
                {
                    let mut reader = CompressorReader::new(data, 4096, 4, 22); // Quality level 4
                    if let Ok(_) = reader.read_to_end(&mut compressed) {
                        if compressed.len() < data.len() {
                            return (compressed, 3); // 3 = Brotli
                        }
                    }
                }
                (data.to_vec(), 0)
            }
            CompressionAlgorithm::Adaptive => {
                // Try different algorithms and choose the best one
                let lz4 = compress_prepend_size(data);
                let zstd = encode_all(data, 3).unwrap_or_else(|_| data.to_vec());

                let mut compressed = Vec::new();
                let mut reader = CompressorReader::new(data, 4096, 4, 22);
                let _ = reader.read_to_end(&mut compressed);

                // Select the smallest compressed result
                if lz4.len() <= zstd.len()
                    && lz4.len() <= compressed.len()
                    && lz4.len() < data.len()
                {
                    (lz4, 1) // LZ4
                } else if zstd.len() <= compressed.len() && zstd.len() < data.len() {
                    (zstd, 2) // Zstd
                } else if compressed.len() < data.len() {
                    (compressed, 3) // Brotli
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
    fn decompress_data(&self, data: &[u8], algorithm: u8) -> StorageResult<Vec<u8>> {
        let start = Instant::now();

        let result =
            match algorithm {
                0 => Ok(data.to_vec()), // No compression
                1 => decompress_size_prepended(data) // LZ4
                    .map_err(|e| {
                        StorageError::InvalidData(format!("LZ4 decompression error: {e}"))
                    }),
                2 => decode_all(data) // Zstd
                    .map_err(|e| {
                        StorageError::InvalidData(format!("Zstd decompression error: {e}"))
                    }),
                3 => {
                    // Brotli
                    let mut decompressor = Decompressor::new(data, 4096);
                    let mut decompressed = Vec::new();
                    match decompressor.read_to_end(&mut decompressed) {
                        Ok(_) => Ok(decompressed),
                        Err(e) => Err(StorageError::InvalidData(format!(
                            "Brotli decompression error: {e}"
                        ))),
                    }
                }
                _ => Err(StorageError::InvalidData(format!(
                    "Unknown compression algorithm: {algorithm}"
                ))),
            };

        // Track stats
        let mut stats = self.stats.write();
        stats.read_time_ns += start.elapsed().as_nanos() as u64;

        result
    }

    /// Flush pending writes and deletes to storage
    async fn flush_pending(&self) -> StorageResult<()> {
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
            let index_map = self.index_map.write();
            let data_map = self.data_map.write();

            // Ensure we have maps
            if index_map.is_none() || data_map.is_none() {
                return Err(StorageError::OperationError(
                    "Storage not initialized".to_string(),
                ));
            }

            let index_mmap = index_map.as_ref().unwrap();
            let data_mmap = data_map.as_ref().unwrap();

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
                        self.key_index.insert(key_hash, hash);
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
                        return Err(StorageError::OperationError(
                            "Index file is full".to_string(),
                        ));
                    }

                    // Create entry
                    let entry = IndexEntry {
                        hash: hash.0,
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
                        return Err(StorageError::OperationError(
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
                        return Err(StorageError::OperationError(
                            "Data file is full".to_string(),
                        ));
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
                        return Err(StorageError::OperationError(
                            "Index file is full".to_string(),
                        ));
                    }

                    // Create entry
                    let entry = IndexEntry {
                        hash: hash.0,
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
    pub fn get_stats(&self) -> StorageStats {
        self.stats.read().clone()
    }

    /// Put a key-value pair into storage, returns the hash
    pub async fn put(&self, key: &[u8], value: &[u8]) -> StorageResult<Hash> {
        // Create composite buffer with key and value
        let mut buffer = Vec::with_capacity(key.len() + value.len() + 8);

        // Add key length as u32
        buffer.extend_from_slice(&(key.len() as u32).to_le_bytes());

        // Add value length as u32
        buffer.extend_from_slice(&(value.len() as u32).to_le_bytes());

        // Add key and value
        buffer.extend_from_slice(key);
        buffer.extend_from_slice(value);

        // Store in the underlying storage
        let hash = self.store(&buffer).await?;

        // Add to key index
        let key_hash = self.hash_key(key);
        self.key_index.insert(key_hash, hash);

        Ok(hash)
    }

    /// Get a value by key using the secondary index
    pub async fn get(&self, key: &[u8]) -> StorageResult<Option<Vec<u8>>> {
        // Look up in secondary index
        let key_hash = self.hash_key(key);

        if let Some(hash) = self.key_index.get(&key_hash) {
            // Found in key index, retrieve data
            if let Some(data) = self.retrieve(hash.value()).await? {
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
    pub async fn delete_key(&self, key: &[u8]) -> StorageResult<()> {
        // Look up in secondary index
        let key_hash = self.hash_key(key);

        if let Some((_, hash)) = self.key_index.remove(&key_hash) {
            // Delete the underlying data
            self.delete(&hash).await?;
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
    async fn get_by_linear_scan(&self, key: &[u8]) -> StorageResult<Option<Vec<u8>>> {
        // If we get here, the secondary index failed us, so scan all entries
        for entry in self.index.iter() {
            let hash = entry.key();

            // Retrieve the data
            if let Some(data) = self.retrieve(hash).await? {
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
                            self.key_index.insert(key_hash, *hash);

                            return Ok(Some(value));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Fall back to a linear scan for deleting if the key index lookup fails
    async fn delete_by_linear_scan(&self, key: &[u8]) -> StorageResult<()> {
        // For each entry in the index
        for entry in self.index.iter() {
            let hash = entry.key();

            // Retrieve the data
            if let Some(data) = self.retrieve(hash).await? {
                // Parse the data to extract the key
                if data.len() >= 8 {
                    // Minimum size: key_len(4) + value_len(4)
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

                    if data.len() >= 8 + key_len {
                        let stored_key = &data[8..8 + key_len];

                        if stored_key == key {
                            // Found the key, delete it
                            self.delete(hash).await?;
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
    pub async fn list_keys(&self) -> StorageResult<Vec<Vec<u8>>> {
        let mut keys = Vec::new();

        // For each entry in the index
        for entry in self.index.iter() {
            let hash = entry.key();
            let (offset, size, compression) = *entry.value();

            // Retrieve the data
            if let Some(data) = self.retrieve(hash).await? {
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
    async fn hash_for_key(&self, key: &[u8]) -> StorageResult<Option<Hash>> {
        for entry in self.index.iter() {
            let hash = entry.key();
            let (offset, size, compression) = *entry.value();

            // Retrieve the data
            if let Some(data) = self.retrieve(hash).await? {
                // Parse the data to extract the key
                if data.len() >= 8 {
                    // Minimum size: key_len(4) + value_len(4)
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

                    if data.len() >= 8 + key_len {
                        let stored_key = &data[8..8 + key_len];
                        if stored_key == key {
                            return Ok(Some(*hash));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Get value for a specific key directly
    async fn get_by_key(&self, key: &[u8]) -> StorageResult<Option<Vec<u8>>> {
        // Get the hash for this key
        if let Some(hash) = self.hash_for_key(key).await? {
            // Retrieve the data
            if let Some(data) = self.retrieve(&hash).await? {
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

// Clone implementation for StorageStats
impl Clone for StorageStats {
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
    async fn store(&self, data: &[u8]) -> StorageResult<Hash> {
        // Acquire semaphore to limit concurrent operations
        let _permit = self.semaphore.acquire().await.unwrap();
        let start = Instant::now();

        // Calculate hash
        let hash_bytes = blake3::hash(data).as_bytes().clone();
        let hash = Hash::new(hash_bytes);

        // Check if data already exists
        if self.exists(&hash).await? {
            // Already stored, update stats
            let mut stats = self.stats.write();
            stats.writes += 1;
            stats.write_time_ns += start.elapsed().as_nanos() as u64;
            stats.cache_hits += 1;
            return Ok(hash);
        }

        // Compress data
        let (compressed_data, compression_type) = self.compress_data(data);

        // Add to pending batch
        {
            let mut batch = self.pending_batch.lock();
            batch.writes.push((hash, compressed_data.clone()));

            // Check if batch is full - if so, flush it
            if batch.writes.len() >= self.options.max_pending_writes {
                drop(batch); // Release lock before async call
                self.flush_pending().await?;
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.writes += 1;
            stats.write_time_ns += start.elapsed().as_nanos() as u64;

            if compression_type > 0 {
                stats.compression_saved += (data.len() - compressed_data.len()) as u64;
                stats.compressed_blocks += 1;
            } else {
                stats.uncompressed_blocks += 1;
            }
        }

        Ok(hash)
    }

    async fn retrieve(&self, hash: &Hash) -> StorageResult<Option<Vec<u8>>> {
        // Acquire semaphore to limit concurrent operations
        let _permit = self.semaphore.acquire().await.unwrap();
        let start = Instant::now();

        // Start with in-memory index lookup
        if let Some(entry) = self.index.get(hash) {
            let (offset, size, compression) = *entry;

            // Handle inlined data (small values stored directly in index)
            if (compression & 0x80) != 0 {
                // High bit indicates inline data
                let compression_algo = compression & 0x7F; // Low 7 bits are actual compression type

                // Read the data from the index map (it's inlined)
                let data = {
                    let index_map = self.index_map.read();
                    if let Some(ref mmap) = *index_map {
                        let entry_offset = offset as usize;
                        let entry_size = size as usize;

                        // Safety: bounds checked by size and we validate the index on load
                        if entry_offset + entry_size <= mmap.len() {
                            mmap[entry_offset..entry_offset + entry_size].to_vec()
                        } else {
                            return Err(StorageError::OperationError(
                                "Index entry points outside of mmap bounds".to_string(),
                            ));
                        }
                    } else {
                        return Err(StorageError::OperationError(
                            "Storage not initialized".to_string(),
                        ));
                    }
                };

                // Decompress if needed
                let result = if compression_algo > 0 {
                    self.decompress_data(&data, compression_algo)?
                } else {
                    data
                };

                // Update stats
                {
                    let mut stats = self.stats.write();
                    stats.reads += 1;
                    stats.read_time_ns += start.elapsed().as_nanos() as u64;
                }

                return Ok(Some(result));
            }

            // Read the data from the data file
            let data = {
                let data_map = self.data_map.read();
                if let Some(ref mmap) = *data_map {
                    let data_offset = offset as usize;
                    let data_size = size as usize;

                    // Safety: bounds checked by size and we validate the data map on load
                    if data_offset + data_size <= mmap.len() {
                        mmap[data_offset..data_offset + data_size].to_vec()
                    } else {
                        return Err(StorageError::OperationError(
                            "Data entry points outside of mmap bounds".to_string(),
                        ));
                    }
                } else {
                    return Err(StorageError::OperationError(
                        "Storage not initialized".to_string(),
                    ));
                }
            };

            // Decompress if needed
            let result = if compression > 0 {
                self.decompress_data(&data, compression)?
            } else {
                data
            };

            // Update stats
            {
                let mut stats = self.stats.write();
                stats.reads += 1;
                stats.read_time_ns += start.elapsed().as_nanos() as u64;
            }

            return Ok(Some(result));
        }

        // Not found in memory index
        Ok(None)
    }

    async fn exists(&self, hash: &Hash) -> StorageResult<bool> {
        // Simple check in the in-memory index
        Ok(self.index.contains_key(hash))
    }

    async fn delete(&self, hash: &Hash) -> StorageResult<()> {
        // Acquire semaphore to limit concurrent operations
        let _permit = self.semaphore.acquire().await.unwrap();

        // Add to pending deletes batch
        {
            let mut batch = self.pending_batch.lock();
            batch.deletes.push(*hash);

            // Check if batch is full
            if batch.deletes.len() >= self.options.max_pending_writes {
                drop(batch); // Release lock before async call
                self.flush_pending().await?;
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.deletes += 1;
        }

        Ok(())
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> StorageResult<bool> {
        // Calculate hash
        let calculated_hash = Hash::new(blake3::hash(data).as_bytes().clone());

        // Compare with provided hash
        Ok(*hash == calculated_hash)
    }

    async fn close(&self) -> StorageResult<()> {
        // Flush any pending writes
        self.flush_pending().await?;

        // Sync files to disk
        {
            if let Some(ref mut file) = *self.index_file.lock() {
                file.sync_all()?;
            }

            if let Some(ref mut file) = *self.data_file.lock() {
                file.sync_all()?;
            }
        }

        // Release memory maps
        {
            let mut index_map = self.index_map.write();
            *index_map = None;

            let mut data_map = self.data_map.write();
            *data_map = None;
        }

        // Close files
        {
            let mut index_file = self.index_file.lock();
            *index_file = None;

            let mut data_file = self.data_file.lock();
            *data_file = None;
        }

        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl StorageInit for MemMapStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> StorageResult<()> {
        let base_path = path.as_ref();

        // Create directory if it doesn't exist
        if !base_path.exists() {
            std::fs::create_dir_all(base_path)
                .map_err(|e| StorageError::InitError(format!("Failed to create directory: {e}")))?;
        }

        let index_path = base_path.join(INDEX_FILENAME);
        let data_path = base_path.join(DATA_FILENAME);

        // Open or create files
        let index_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&index_path)
            .map_err(|e| StorageError::InitError(format!("Failed to open index file: {e}")))?;

        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&data_path)
            .map_err(|e| StorageError::InitError(format!("Failed to open data file: {e}")))?;

        // Initialize file sizes
        let index_size = index_file
            .metadata()
            .map_err(|e| StorageError::InitError(format!("Failed to get index metadata: {e}")))?
            .len();

        let data_size = data_file
            .metadata()
            .map_err(|e| StorageError::InitError(format!("Failed to get data metadata: {e}")))?
            .len();

        // Initialize empty files if needed
        if index_size == 0 {
            // Write magic bytes
            index_file.write_all(MAGIC_BYTES).map_err(|e| {
                StorageError::InitError(format!("Failed to initialize index file: {e}"))
            })?;

            // Extend to required size
            index_file
                .set_len(self.options.map_size as u64)
                .map_err(|e| {
                    StorageError::InitError(format!("Failed to resize index file: {e}"))
                })?;
        }

        if data_size == 0 {
            // Write magic bytes
            data_file.write_all(MAGIC_BYTES).map_err(|e| {
                StorageError::InitError(format!("Failed to initialize data file: {e}"))
            })?;

            // Extend to required size
            data_file
                .set_len(self.options.map_size as u64)
                .map_err(|e| StorageError::InitError(format!("Failed to resize data file: {e}")))?;

            // Reset data size counter
            *self.data_size.write() = MAGIC_BYTES.len() as u64;
        } else {
            // Set current data size counter
            *self.data_size.write() = data_size;
        }

        // Memory map files
        unsafe {
            let index_mmap = MmapOptions::new().map_mut(&index_file).map_err(|e| {
                StorageError::InitError(format!("Failed to memory map index file: {e}"))
            })?;

            let data_mmap = MmapOptions::new().map_mut(&data_file).map_err(|e| {
                StorageError::InitError(format!("Failed to memory map data file: {e}"))
            })?;

            // Store memory maps and file handles
            *self.index_map.write() = Some(index_mmap);
            *self.data_map.write() = Some(data_mmap);
            *self.index_file.lock() = Some(index_file);
            *self.data_file.lock() = Some(data_file);
        }

        // Load existing index entries
        if let Some(ref index_mmap) = *self.index_map.read() {
            self.load_index(index_mmap).await?;
        }

        // Build the key index for fast key-value lookups
        self.build_key_index().await?;

        // Preload data into memory if configured
        if self.options.preload_data {
            self.preload_data().await?;
        }

        Ok(())
    }
}

impl MemMapStorage {
    /// Load index entries into memory
    async fn load_index(&self, index_mmap: &MmapMut) -> StorageResult<()> {
        let start = Instant::now();

        // Validate magic bytes
        if index_mmap.len() < MAGIC_BYTES.len() || &index_mmap[0..MAGIC_BYTES.len()] != MAGIC_BYTES
        {
            return Err(StorageError::InitError(
                "Invalid index file format".to_string(),
            ));
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
            entries.push((
                Hash::new(entry.hash),
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
                self.index.insert(*hash, *entry);
            }
        });
    }

    /// Preload data into memory for faster access
    async fn preload_data(&self) -> StorageResult<()> {
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
    async fn build_key_index(&self) -> StorageResult<()> {
        let start = Instant::now();

        // Clear the existing key index
        self.key_index.clear();

        // For each entry in the main index
        let mut added = 0;
        for entry in self.index.iter() {
            let hash = entry.key();

            // Retrieve the data
            if let Some(data) = self.retrieve(hash).await? {
                // Parse the data to extract the key
                if data.len() >= 8 {
                    // Minimum size: key_len(4) + value_len(4)
                    let key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

                    if data.len() >= 8 + key_len {
                        let key = &data[8..8 + key_len];

                        // Add to key index
                        let key_hash = self.hash_key(key);
                        self.key_index.insert(key_hash, *hash);
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
