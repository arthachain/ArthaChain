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
                        StorageError::InvalidData(format!("LZ4 decompression error: {}", e))
                    }),
                2 => decode_all(data) // Zstd
                    .map_err(|e| {
                        StorageError::InvalidData(format!("Zstd decompression error: {}", e))
                    }),
                3 => {
                    // Brotli
                    let mut decompressor = Decompressor::new(data, 4096);
                    let mut decompressed = Vec::new();
                    match decompressor.read_to_end(&mut decompressed) {
                        Ok(_) => Ok(decompressed),
                        Err(e) => Err(StorageError::InvalidData(format!(
                            "Brotli decompression error: {}",
                            e
                        ))),
                    }
                }
                _ => Err(StorageError::InvalidData(format!(
                    "Unknown compression algorithm: {}",
                    algorithm
                ))),
            };

        // Track stats
        let mut stats = self.stats.write();
        stats.read_time_ns += start.elapsed().as_nanos() as u64;

        result
    }

    /// Flush pending writes to storage
    async fn flush_pending(&self) -> StorageResult<()> {
        let mut batch = {
            let mut batch_guard = self.pending_batch.lock();
            std::mem::replace(
                &mut *batch_guard,
                Batch {
                    writes: Vec::new(),
                    deletes: Vec::new(),
                },
            )
        };

        // Process all writes
        if !batch.writes.is_empty() {
            let index_map = self.index_map.clone();
            let data_map = self.data_map.clone();
            let index = self.index.clone();
            let data_size = self.data_size.clone();
            let options = self.options.clone();
            let stats = self.stats.clone();
            let this = self.clone();

            // Process writes in a blocking task to not block the async runtime
            let result = task::spawn_blocking(move || {
                let start = Instant::now();
                let mut data_size_guard = data_size.write();
                let mut data_map_guard = data_map.write();
                let mut index_map_guard = index_map.write();

                if let (Some(ref mut data_mmap), Some(ref mut index_mmap)) =
                    (data_map_guard.as_mut(), index_map_guard.as_mut())
                {
                    // Sort writes by size (larger first) for better packing
                    batch.writes.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

                    // Process each write
                    for (hash, data) in batch.writes {
                        let (compressed_data, compression_algo) = this.compress_data(&data);

                        // For small data, store inline in index
                        if compressed_data.len() <= MAX_INLINE_DATA_SIZE {
                            // Create inline entry in index
                            let mut entry = IndexEntry {
                                hash: [0u8; 32],
                                offset: 0, // Special marker for inline
                                size: compressed_data.len() as u32,
                                compression: compression_algo,
                                flags: 1, // Bit 0 = inline
                                _padding: [0; 2],
                            };
                            entry.hash.copy_from_slice(hash.as_bytes());

                            // Find space in index or append
                            // For simplicity, we'll just append for now
                            let index_offset = index_mmap.len() as u64;
                            let entry_bytes = unsafe {
                                std::slice::from_raw_parts(
                                    &entry as *const _ as *const u8,
                                    std::mem::size_of::<IndexEntry>(),
                                )
                            };

                            // Resize index if needed
                            if index_offset as usize
                                + std::mem::size_of::<IndexEntry>()
                                + compressed_data.len()
                                > index_mmap.len()
                            {
                                // Grow to new size
                                let new_size = index_mmap
                                    .metadata()
                                    .map_err(|e| {
                                        StorageError::OperationError(format!(
                                            "Failed to get current metadata: {}",
                                            e
                                        ))
                                    })?
                                    .len()
                                    + BLOCK_SIZE as u64;

                                index_mmap.set_len(new_size).map_err(|e| {
                                    StorageError::OperationError(format!(
                                        "Failed to extend index file: {}",
                                        e
                                    ))
                                })?;

                                // Re-map the file
                                drop(index_map_guard); // Release lock to resize

                                let new_mmap = unsafe { MmapOptions::new().map_mut(&index_mmap) }
                                    .map_err(|e| {
                                    StorageError::OperationError(format!(
                                        "Failed to remap index file: {}",
                                        e
                                    ))
                                })?;

                                index_map_guard = index_map.write();
                                *index_map_guard = Some(new_mmap);

                                // Re-check if we have enough space now
                                let total_size =
                                    std::mem::size_of::<IndexEntry>() + compressed_data.len();

                                if let Some(ref mut index_mmap) = *index_map_guard {
                                    if index_offset + total_size > index_mmap.len() {
                                        return Err(StorageError::OperationError(
                                            "Failed to extend index file enough".to_string(),
                                        ));
                                    }
                                } else {
                                    return Err(StorageError::OperationError(
                                        "Index map is None after resize".to_string(),
                                    ));
                                }
                            }

                            // Write entry to index
                            if let Some(ref mut index_mmap) = *index_map_guard {
                                let start_pos = index_offset as usize;
                                let end_pos = start_pos + std::mem::size_of::<IndexEntry>();
                                index_mmap[start_pos..end_pos].copy_from_slice(entry_bytes);

                                // Write inline data right after entry
                                let data_start = end_pos;
                                let data_end = data_start + compressed_data.len();
                                index_mmap[data_start..data_end].copy_from_slice(&compressed_data);

                                // Update in-memory index
                                index.insert(
                                    hash,
                                    (index_offset, compressed_data.len() as u32, 0xFF),
                                ); // 0xFF = inline marker
                            }
                        } else {
                            // Append to data file
                            let offset = *data_size_guard;
                            let padded_size =
                                (compressed_data.len() + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

                            // Ensure data file has enough space
                            if let Some(ref data_map) = *data_mmap {
                                if (offset as usize + padded_size) > data_map.len() {
                                    // Need to grow data file
                                    drop(data_map_guard); // Release lock to resize
                                    if let Some(data_file) = this.data_file.lock().as_mut() {
                                        let new_size =
                                            offset as usize + padded_size + (1024 * 1024); // Add 1MB extra
                                        data_file.set_len(new_size as u64).map_err(|e| {
                                            StorageError::OperationError(format!(
                                                "Failed to resize data file: {}",
                                                e
                                            ))
                                        })?;

                                        // Recreate mmap with new size
                                        let new_mmap =
                                            unsafe { MmapOptions::new().map_mut(data_file) }
                                                .map_err(|e| {
                                                    StorageError::OperationError(format!(
                                                        "Failed to remap data: {}",
                                                        e
                                                    ))
                                                })?;

                                        data_map_guard = data_map.write();
                                        *data_map_guard = Some(new_mmap);
                                    } else {
                                        return Err(StorageError::OperationError(
                                            "Data file not open".to_string(),
                                        ));
                                    }
                                }
                            } else {
                                return Err(StorageError::OperationError(
                                    "Data map is None".to_string(),
                                ));
                            }

                            // Write data
                            if let Some(ref mut data_mmap) = *data_map_guard {
                                let start_pos = offset as usize;
                                let end_pos = start_pos + compressed_data.len();
                                data_mmap[start_pos..end_pos].copy_from_slice(&compressed_data);

                                // Create index entry
                                let mut entry = IndexEntry {
                                    hash: [0u8; 32],
                                    offset,
                                    size: compressed_data.len() as u32,
                                    compression: compression_algo,
                                    flags: 0, // Bit 0 = 0 (not inline)
                                    _padding: [0; 2],
                                };
                                entry.hash.copy_from_slice(hash.as_bytes());

                                // Write entry to index
                                let index_offset = index_mmap.len() as u64;
                                let entry_bytes = unsafe {
                                    std::slice::from_raw_parts(
                                        &entry as *const _ as *const u8,
                                        std::mem::size_of::<IndexEntry>(),
                                    )
                                };

                                // Resize index if needed
                                if index_offset as usize + std::mem::size_of::<IndexEntry>()
                                    > index_mmap.len()
                                {
                                    // Grow to new size
                                    let new_size = index_mmap
                                        .metadata()
                                        .map_err(|e| {
                                            StorageError::OperationError(format!(
                                                "Failed to get current metadata: {}",
                                                e
                                            ))
                                        })?
                                        .len()
                                        + BLOCK_SIZE as u64;

                                    index_mmap.set_len(new_size).map_err(|e| {
                                        StorageError::OperationError(format!(
                                            "Failed to extend index file: {}",
                                            e
                                        ))
                                    })?;

                                    // Re-map the file
                                    drop(index_map_guard); // Release lock to resize

                                    let new_mmap =
                                        unsafe { MmapOptions::new().map_mut(&index_mmap) }
                                            .map_err(|e| {
                                                StorageError::OperationError(format!(
                                                    "Failed to remap index file: {}",
                                                    e
                                                ))
                                            })?;

                                    index_map_guard = index_map.write();
                                    *index_map_guard = Some(new_mmap);

                                    // Re-check if we have enough space now
                                    let total_size =
                                        std::mem::size_of::<IndexEntry>() + compressed_data.len();

                                    if let Some(ref mut index_mmap) = *index_map_guard {
                                        if index_offset + total_size > index_mmap.len() {
                                            return Err(StorageError::OperationError(
                                                "Failed to extend index file enough".to_string(),
                                            ));
                                        }
                                    } else {
                                        return Err(StorageError::OperationError(
                                            "Index map is None after resize".to_string(),
                                        ));
                                    }
                                }

                                // Write entry to index
                                if let Some(ref mut index_mmap) = *index_map_guard {
                                    let start_pos = index_offset as usize;
                                    let end_pos = start_pos + std::mem::size_of::<IndexEntry>();
                                    index_mmap[start_pos..end_pos].copy_from_slice(entry_bytes);

                                    // Update in-memory index
                                    index.insert(
                                        hash,
                                        (offset, compressed_data.len() as u32, compression_algo),
                                    );

                                    // Update data size
                                    *data_size_guard = offset + padded_size as u64;
                                }
                            }
                        }
                    }

                    // Flush mmaps to ensure durability
                    if let Some(ref mut data_mmap) = *data_mmap {
                        data_mmap.flush().map_err(|e| {
                            StorageError::OperationError(format!(
                                "Failed to flush data file: {}",
                                e
                            ))
                        })?;
                    } else {
                        return Err(StorageError::OperationError("Data map is None".to_string()));
                    }

                    if let Some(ref mut index_mmap) = *index_mmap {
                        index_mmap.flush().map_err(|e| {
                            StorageError::OperationError(format!(
                                "Failed to flush index file: {}",
                                e
                            ))
                        })?;
                    } else {
                        return Err(StorageError::OperationError(
                            "Index map is None".to_string(),
                        ));
                    }

                    // Update stats
                    let mut stats_guard = stats.write();
                    stats_guard.writes += batch.writes.len() as u64;
                    stats_guard.write_time_ns += start.elapsed().as_nanos() as u64;
                }

                Ok(())
            })
            .await
            .map_err(|e| StorageError::OperationError(format!("Task join error: {}", e)))?;

            return Ok(());
        }

        // Process deletes (mark as deleted in index)
        // For simplicity, we're not reclaiming space yet
        if !batch.deletes.is_empty() {
            for hash in batch.deletes {
                self.index.remove(&hash);
            }

            let mut stats = self.stats.write();
            stats.deletes += batch.deletes.len() as u64;
        }

        Ok(())
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> StorageStats {
        self.stats.read().clone()
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
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            StorageError::OperationError(format!("Failed to acquire semaphore: {}", e))
        })?;

        // Calculate hash
        let hash_value = blake3::hash(data);
        let mut hash_bytes = [0u8; 32];
        hash_bytes.copy_from_slice(hash_value.as_bytes());
        let hash = Hash::new(hash_bytes.to_vec());

        // Add to pending batch
        {
            let mut batch = self.pending_batch.lock();
            batch.writes.push((hash.clone(), data.to_vec()));

            // If batch is full, flush it
            if batch.writes.len() >= self.options.max_pending_writes {
                drop(batch); // Release lock before flush
                self.flush_pending().await?;
            }
        }

        Ok(hash)
    }

    async fn retrieve(&self, hash: &Hash) -> StorageResult<Option<Vec<u8>>> {
        // Acquire semaphore to limit concurrent operations
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            StorageError::OperationError(format!("Failed to acquire semaphore: {}", e))
        })?;

        let start = Instant::now();

        // Check in-memory index
        if let Some(entry) = self.index.get(hash) {
            let (offset, size, compression) = *entry;

            // Track stats
            {
                let mut stats = self.stats.write();
                stats.reads += 1;
            }

            // Check if data is stored inline (compression = 0xFF)
            if compression == 0xFF {
                // Data is stored inline in index
                let index_map = self.index_map.read();
                if let Some(ref index_mmap) = *index_map {
                    let entry_size = std::mem::size_of::<IndexEntry>();
                    let data_start = offset as usize + entry_size;
                    let data_end = data_start + size as usize;

                    if data_end <= index_mmap.as_ref().unwrap().len() {
                        let data = index_mmap[data_start..data_end].to_vec();

                        // Track stats
                        {
                            let mut stats = self.stats.write();
                            stats.cache_hits += 1;
                            stats.read_time_ns += start.elapsed().as_nanos() as u64;
                        }

                        return Ok(Some(data));
                    }
                }
            } else {
                // Data is in main storage
                let data_map = self.data_map.read();
                if let Some(ref data_mmap) = *data_map {
                    let start_pos = offset as usize;
                    let end_pos = start_pos + size as usize;

                    if end_pos <= data_mmap.as_ref().unwrap().len() {
                        let compressed_data = data_mmap[start_pos..end_pos].to_vec();

                        // Decompress if needed
                        let result = self.decompress_data(&compressed_data, compression)?;

                        // Track stats
                        {
                            let mut stats = self.stats.write();
                            stats.read_time_ns += start.elapsed().as_nanos() as u64;
                        }

                        return Ok(Some(result));
                    }
                }
            }
        }

        // Data not found
        Ok(None)
    }

    async fn exists(&self, hash: &Hash) -> StorageResult<bool> {
        // Just check in-memory index
        Ok(self.index.contains_key(hash))
    }

    async fn delete(&self, hash: &Hash) -> StorageResult<()> {
        // Add to pending batch
        {
            let mut batch = self.pending_batch.lock();
            batch.deletes.push(hash.clone());

            // If batch is full, flush it
            if batch.deletes.len() >= self.options.max_pending_writes {
                drop(batch); // Release lock before flush
                self.flush_pending().await?;
            }
        }

        Ok(())
    }

    async fn verify(&self, hash: &Hash, data: &[u8]) -> StorageResult<bool> {
        let calculated_hash = blake3::hash(data);
        let calculated = Hash::new(calculated_hash.as_bytes().to_vec());
        Ok(calculated == *hash)
    }

    async fn close(&self) -> StorageResult<()> {
        // Flush any pending changes
        self.flush_pending().await?;

        // Explicitly flush and close mmaps
        {
            let mut index_map = self.index_map.write();
            if let Some(ref mut mmap) = *index_map {
                mmap.as_mut().unwrap().flush().map_err(|e| {
                    StorageError::OperationError(format!("Failed to flush index: {}", e))
                })?;
            }
            *index_map = None;
        }

        {
            let mut data_map = self.data_map.write();
            if let Some(ref mut mmap) = *data_map {
                mmap.as_mut().unwrap().flush().map_err(|e| {
                    StorageError::OperationError(format!("Failed to flush data: {}", e))
                })?;
            }
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[async_trait]
impl StorageInit for MemMapStorage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> StorageResult<()> {
        let base_path = path.as_ref();
        let index_path = {
            let path_ref = base_path.as_ref();
            path_ref.join(INDEX_FILENAME)
        };
        let data_path = {
            let path_ref = base_path.as_ref();
            path_ref.join(DATA_FILENAME)
        };

        // Create directory if it doesn't exist
        std::fs::create_dir_all(base_path)
            .map_err(|e| StorageError::InitError(format!("Failed to create directory: {}", e)))?;

        // Open index file
        let index_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&index_path)
            .map_err(|e| StorageError::InitError(format!("Failed to open index file: {}", e)))?;

        // Get file size and initialize if new
        let index_metadata = index_file
            .metadata()
            .map_err(|e| StorageError::InitError(format!("Failed to get index metadata: {}", e)))?;

        let is_new_index = index_metadata.len() == 0;
        if is_new_index {
            // Initialize with header
            index_file.set_len(BLOCK_SIZE as u64).map_err(|e| {
                StorageError::InitError(format!("Failed to set index file size: {}", e))
            })?;

            // Initialize with magic bytes and version
            let mut header = [0u8; BLOCK_SIZE];
            header[0..8].copy_from_slice(MAGIC_BYTES);

            // Write version (1.0)
            header[8] = 1;
            header[9] = 0;

            index_file.write_all(&header).map_err(|e| {
                StorageError::InitError(format!("Failed to write index header: {}", e))
            })?;
        }

        // Open data file
        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&data_path)
            .map_err(|e| StorageError::InitError(format!("Failed to open data file: {}", e)))?;

        // Get file size and initialize if new
        let data_metadata = data_file
            .metadata()
            .map_err(|e| StorageError::InitError(format!("Failed to get data metadata: {}", e)))?;

        let is_new_data = data_metadata.len() == 0;
        let data_len = if is_new_data {
            // Initialize with minimum size
            let initial_size = std::cmp::max(BLOCK_SIZE, self.options.map_size / 100);
            data_file.set_len(initial_size as u64).map_err(|e| {
                StorageError::InitError(format!("Failed to set data file size: {}", e))
            })?;
            initial_size as u64
        } else {
            data_metadata.len()
        };

        // Create memory maps
        let index_mmap = unsafe { MmapOptions::new().map_mut(&index_file) }
            .map_err(|e| StorageError::InitError(format!("Failed to map index file: {}", e)))?;

        let data_mmap = unsafe { MmapOptions::new().map_mut(&data_file) }
            .map_err(|e| StorageError::InitError(format!("Failed to map data file: {}", e)))?;

        // Check magic bytes
        if !is_new_index {
            let magic = &index_mmap[0..8];
            if magic != MAGIC_BYTES {
                return Err(StorageError::InitError(
                    "Invalid magic bytes in index file".to_string(),
                ));
            }

            // Load index entries into memory
            self.load_index(&index_mmap).await?;
        }

        // Store file handles and mmaps
        {
            let mut index_file_guard = self.index_file.lock();
            *index_file_guard = Some(index_file);
        }

        {
            let mut data_file_guard = self.data_file.lock();
            *data_file_guard = Some(data_file);
        }

        {
            let mut index_map_guard = self.index_map.write();
            *index_map_guard = Some(index_mmap);
        }

        {
            let mut data_map_guard = self.data_map.write();
            *data_map_guard = Some(data_mmap);
        }

        {
            let mut data_size_guard = self.data_size.write();
            *data_size_guard = data_len;
        }

        // Preload data if configured
        if self.options.preload_data && !is_new_data {
            self.preload_data().await?;
        }

        Ok(())
    }
}

impl MemMapStorage {
    /// Load index entries into memory
    async fn load_index(&self, index_mmap: &MmapMut) -> StorageResult<()> {
        let start = Instant::now();

        // Skip header block
        let mut offset = BLOCK_SIZE;
        let entry_size = std::mem::size_of::<IndexEntry>();

        while offset + entry_size <= index_mmap.len() {
            // Read entry
            let entry_bytes = &index_mmap[offset..offset + entry_size];
            let entry =
                unsafe { std::ptr::read_unaligned(entry_bytes.as_ptr() as *const IndexEntry) };

            // Check if entry is valid (non-zero hash)
            if entry.hash.iter().any(|&b| b != 0) {
                let hash = Hash::new(entry.hash.to_vec());

                // Add to in-memory index
                let is_inline = (entry.flags & 1) != 0;
                if is_inline {
                    // Inline data
                    self.index.insert(hash, (offset as u64, entry.size, 0xFF)); // 0xFF = inline marker
                } else {
                    // Normal data reference
                    self.index
                        .insert(hash, (entry.offset, entry.size, entry.compression));
                }
            }

            // Move to next entry
            offset += entry_size;

            // If entry contains inline data, skip it
            if (entry.flags & 1) != 0 {
                offset += entry.size as usize;
            }
        }

        // Log loading time
        let elapsed = start.elapsed();
        log::info!("Loaded {} index entries in {:?}", self.index.len(), elapsed);

        Ok(())
    }

    /// Preload data into memory to improve performance
    async fn preload_data(&self) -> StorageResult<()> {
        let start = Instant::now();

        // For memory-mapped files, reading the data forces it into memory
        // We'll read in parallel for better performance
        let data_map = self.data_map.read();
        if let Some(ref data_mmap) = *data_map {
            let chunk_size = 1024 * 1024; // 1MB chunks
            let total_size = data_mmap.len();
            let chunks = (total_size + chunk_size - 1) / chunk_size;

            // Read in parallel
            (0..chunks).into_par_iter().for_each(|i| {
                let start = i * chunk_size;
                let end = std::cmp::min(start + chunk_size, total_size);

                // Read chunk (forces page into memory)
                let _data = &data_mmap[start..end];

                // This is enough to cause the kernel to load the page
                let _sum: u64 = _data.iter().take(10).map(|&b| b as u64).sum();
            });
        }

        let elapsed = start.elapsed();
        log::info!("Preloaded data in {:?}", elapsed);

        Ok(())
    }
}
