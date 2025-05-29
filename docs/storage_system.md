# Storage System Documentation

## Overview

The ArthaChain blockchain employs a sophisticated multi-layered storage system designed for high performance, durability, and scalability. The storage system is responsible for persisting blockchain data, state, and supporting AI model data in an efficient manner.

## Architecture

The storage module is organized into several specialized components:

```
blockchain_node/src/storage/
├── blockchain_storage.rs     # Core blockchain data storage
├── hybrid_storage.rs         # Combined memory and disk storage
├── memmap_storage.rs         # Memory-mapped file storage
├── mod.rs                    # Module definition
├── rocksdb_storage.rs        # RocksDB-based persistent storage
├── svdb_storage.rs           # Specialized vector database storage
└── transaction.rs            # Transaction handling for storage
```

## Core Components

### 1. Blockchain Storage (blockchain_storage.rs)

The `BlockchainStorage` is the high-level abstraction for storing blockchain data:

```rust
pub struct BlockchainStorage {
    // Block storage
    block_store: Box<dyn BlockStore>,
    
    // Transaction storage
    tx_store: Box<dyn TransactionStore>,
    
    // State storage
    state_store: Box<dyn StateStore>,
    
    // Receipt storage
    receipt_store: Box<dyn ReceiptStore>,
    
    // Storage configuration
    config: StorageConfig,
}
```

Key functionality:
- Storage and retrieval of blocks, transactions, state, and receipts
- Support for different storage backends
- Optimized access patterns for blockchain data
- Transaction batch support for atomicity

### 2. RocksDB Storage (rocksdb_storage.rs)

The `RocksDbStorage` provides persistent storage using the RocksDB key-value store:

```rust
pub struct RocksDbStorage {
    /// RocksDB database instance
    db: Arc<RwLock<Option<DB>>>,
    /// Path to database for reopening
    db_path: Arc<RwLock<Option<std::path::PathBuf>>>,
}
```

Key functionality:
- Persistent storage of blockchain data
- Blake3 hashing for data integrity
- Thread-safe access via Arc<RwLock<>> wrapper
- Automated database reopening if closed
- Efficient key-value operations

### 3. Memory-Mapped Storage (memmap_storage.rs)

The `MemMapStorage` provides high-performance storage using memory-mapped files:

```rust
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
```

Key functionality:
- Zero-copy data access via memory mapping
- Adaptive compression with multiple algorithms (LZ4, Zstd, Brotli)
- Small data inline storage optimization
- Concurrent access control with fine-grained synchronization
- Performance statistics tracking
- Parallel data processing capabilities

### 4. SVDB Storage (svdb_storage.rs)

The `SvdbStorage` provides specialized storage for vector data, particularly useful for AI models:

```rust
pub struct SvdbStorage {
    /// HTTP client for SVDB API requests
    _client: Client,

    /// Base URL for SVDB API
    _base_url: String,

    /// Database instance
    db: Arc<RwLock<Option<DB>>>,

    /// Path to database for reopening
    db_path: Arc<RwLock<Option<std::path::PathBuf>>>,

    _data: HashMap<String, Vec<u8>>,
}
```

Key functionality:
- Specialized storage for vector data used by AI models
- HTTP client for remote SVDB API integration
- Local RocksDB fallback for offline operation
- Concurrent access support
- Blake3 hashing for data integrity

### 5. Hybrid Storage (hybrid_storage.rs)

The `HybridStorage` combines multiple storage backends for optimized performance:

```rust
pub struct HybridStorage {
    /// RocksDB storage for on-chain data
    rocksdb: Box<dyn Storage>,

    /// SVDB storage for off-chain data
    svdb: Box<dyn Storage>,

    /// Size threshold (in bytes) for deciding between RocksDB and SVDB
    size_threshold: usize,
}
```

Key functionality:
- Intelligent data routing based on size thresholds
- Combined query capability across both backends
- Seamless fallback between storage types
- Optimized for different data access patterns
- Size-based optimization for storage efficiency

## Storage Interfaces

The storage system defines several key interfaces:

### Storage Trait

```rust
#[async_trait]
pub trait Storage: Send + Sync + 'static {
    async fn store(&self, data: &[u8]) -> Result<Hash>;
    async fn retrieve(&self, hash: &Hash) -> Result<Option<Vec<u8>>>;
    async fn exists(&self, hash: &Hash) -> Result<bool>;
    async fn delete(&self, hash: &Hash) -> Result<()>;
    async fn verify(&self, hash: &Hash, data: &[u8]) -> Result<bool>;
    async fn close(&self) -> Result<()>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
```

### StorageInit Trait

```rust
#[async_trait]
pub trait StorageInit: Storage {
    async fn init(&mut self, path: Box<dyn AsRef<Path> + Send + Sync>) -> Result<()>;
}
```

## Compression Features

The storage system supports multiple compression algorithms to optimize storage efficiency:

```rust
pub enum CompressionAlgorithm {
    None,
    LZ4,
    Zstd,
    Brotli,
    Adaptive,
}
```

Key compression features:
- **LZ4**: Fast compression with good ratio
- **Zstd**: Balanced compression with excellent ratio
- **Brotli**: High compression ratio for cold data
- **Adaptive**: Dynamically selects the best algorithm based on data characteristics
- **Threshold-based compression**: Only compresses data above a certain size

## Performance Characteristics

The storage system is designed for high performance:

- **Memory-mapped access**: Sub-microsecond access times for cached data
- **Concurrent operations**: Fine-grained locking for maximum parallelism
- **Batched operations**: Efficient handling of multiple operations
- **Zero-copy access**: Direct memory access without intermediate copying
- **Adaptive compression**: Optimizes for both space and speed
- **Inline storage**: Small values stored directly in index for faster access

## Configuration

The storage system can be configured through multiple options:

### MemMapOptions

```rust
pub struct MemMapOptions {
    pub compression_algorithm: CompressionAlgorithm,
    pub preload_data: bool,
    pub sync_writes: bool,
    pub index_cache_size: usize,
    pub enable_stats: bool,
}
```

### Storage Configuration Examples

#### Production Configuration
```rust
let storage_config = StorageConfig {
    path: "/data/blockchain/production",
    cache_size_mb: 1024, 
    compression: CompressionAlgorithm::Adaptive,
    sync_writes: true,
    max_open_files: 10000,
    keep_log_file_num: 10,
    paranoid_checks: true,
    background_compaction: true,
    use_direct_io: true,
};
```

#### Testing Configuration
```rust
let testing_config = StorageConfig {
    path: "/tmp/blockchain/test",
    cache_size_mb: 128,
    compression: CompressionAlgorithm::None,
    sync_writes: false,
    max_open_files: 100,
    keep_log_file_num: 2,
    paranoid_checks: false,
    background_compaction: false,
    use_direct_io: false,
};
```

## Monitoring and Instrumentation

The storage system provides comprehensive monitoring capabilities:

### StorageStats

```rust
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
```

### Metrics Collection

- **Operation Counts**: Tracking of reads, writes, and deletes
- **Timing Information**: Nanosecond-precision timing of operations
- **Cache Performance**: Hit rates and efficiency metrics
- **Compression Efficiency**: Space saved and compression ratio
- **Throughput Calculation**: Operations per second

## Error Handling Patterns

The storage system employs a comprehensive error handling strategy:

```rust
pub enum StorageError {
    NotFound,
    AlreadyExists,
    Corrupted(String),
    IO(String),
    Serialization(String),
    Compression(String),
    InvalidArgument(String),
    DatabaseError(String),
    LockError(String),
    Other(String),
}
```

Error handling patterns:
- **Specific Error Types**: Detailed error categorization
- **Error Context**: Additional information about the error source
- **Graceful Degradation**: Fallback mechanisms for partial system failures
- **Retry Logic**: Automatic retry for transient failures
- **Propagation Control**: Appropriate error bubbling

## Security Considerations

The storage system implements several security features:

### Data Integrity

- **Cryptographic Hashing**: Blake3 hashing for all stored data
- **Content Verification**: Hash verification on retrieval
- **Atomic Operations**: Transactional guarantees for multi-part operations

### Data Protection

- **Encryption Options**: Support for encrypted storage
- **Access Controls**: Fine-grained permissions for data access
- **Isolation**: Strong separation between different data types
- **Secure Deletion**: Proper data wiping on deletion requests

```rust
// Example of using encrypted storage
let encrypted_storage = EncryptedStorageWrapper::new(
    Box::new(RocksDbStorage::new()),
    EncryptionKey::from_secure_source(),
    EncryptionAlgorithm::AES256GCM
);
```

## Best Practices

### Storage Access Patterns

- **Batch Operations**: Group related operations to minimize overhead
- **Prefix Scanning**: Use key prefixes for efficient range queries
- **Caching Hot Data**: Cache frequently accessed data in memory
- **Write Amplification Awareness**: Structure writes to minimize amplification

### Storage Management

- **Regular Compaction**: Schedule compaction during low-usage periods
- **Backup Strategies**: Implement regular backup procedures
- **Monitoring**: Set up alerts for storage-related metrics
- **Capacity Planning**: Proactive management of storage growth

## Future Enhancements

The storage system roadmap includes:

- **Sharded Storage**: Horizontal scaling across multiple storage nodes
- **Tiered Storage**: Automatic data migration between hot and cold storage
- **Enhanced Encryption**: Additional encryption algorithms and performance optimizations
- **Remote Storage Integration**: Support for cloud storage backends
- **Quantum-Resistant Cryptography**: Preparation for post-quantum era 