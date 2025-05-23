# SVDB Python

Python bindings for the State Variable Database (SVDB), a high-performance, blockchain-backed key-value store.

## Features

- **High Performance**: Built on Rust with RocksDB for blazing-fast storage operations
- **Blockchain Integration**: Optional verification and proof of storage on the blockchain
- **Data Encryption**: AES-256-GCM encryption for sensitive data
- **Asynchronous API**: Modern async Python API for non-blocking operations
- **Batch Operations**: Efficient handling of multiple operations at once
- **Flexible Storage**: Can be used as an in-memory database or persistent storage

## Installation

```bash
pip install svdb
```

## Quick Start

```python
import asyncio
from svdb import SvdbClient

async def main():
    # Create a client with RocksDB storage
    client = SvdbClient("/path/to/db")
    
    # Store some data
    hash = await client.store("my_key", b"Hello, World!")
    print(f"Stored data with hash: {hash}")
    
    # Retrieve the data
    data = await client.retrieve("my_key")
    print(f"Retrieved data: {data.decode('utf-8')}")
    
    # Check existence
    exists = await client.exists("my_key")
    print(f"Key exists: {exists}")
    
    # Verify integrity
    is_valid = await client.verify("my_key", hash)
    print(f"Data integrity valid: {is_valid}")
    
    # Delete the data
    await client.delete("my_key")
    print("Data deleted")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage

### Encryption

```python
import os
from svdb import SvdbClient

# Generate a secure 32-byte key
encryption_key = os.urandom(32)

# Create a client with encryption
client = SvdbClient("/path/to/db", encryption_key=encryption_key)

# Store encrypted data
await client.store("secret", b"My secret data", encrypted=True)
```

### Blockchain Integration

```python
from svdb import SvdbClient

# Create a client with blockchain verification
client = SvdbClient(
    "/path/to/db", 
    blockchain_endpoint="http://localhost:8545"
)

# Store data with blockchain verification
hash = await client.store("important_data", b"Important data to verify")

# Later, verify the data against the blockchain
is_valid = await client.verify("important_data", hash)
```

### Batch Operations

```python
from svdb import SvdbClient

client = SvdbClient("/path/to/db")

# Batch store
items = [
    ("key1", b"data1", False),  # key, data, encrypted
    ("key2", b"data2", True),
]
results = await client.batch_store(items)

# Batch retrieve
data_dict = await client.batch_retrieve(["key1", "key2"])
for key, data in data_dict.items():
    print(f"{key}: {data}")

# Batch delete
await client.batch_delete(["key1", "key2"])
```

## API Reference

### `SvdbClient`

**Constructor**

```python
SvdbClient(path, encryption_key=None, blockchain_endpoint=None)
```

- `path` (str): Path to the database directory
- `encryption_key` (bytes, optional): 32-byte encryption key
- `blockchain_endpoint` (str, optional): Blockchain RPC endpoint URL

**Methods**

- `async store(key, data, encrypted=False) -> str`: Store data and return hash
- `async retrieve(key) -> bytes`: Retrieve data for a key
- `async delete(key) -> None`: Delete data for a key
- `async exists(key) -> bool`: Check if a key exists
- `async verify(key, expected_hash) -> bool`: Verify data integrity
- `async batch_store(items) -> dict`: Store multiple items
- `async batch_retrieve(keys) -> dict`: Retrieve multiple items
- `async batch_delete(keys) -> None`: Delete multiple items

## Development

```bash
# Clone the repository
git clone https://github.com/ArthaChain/svdb.git
cd svdb/svdb-python

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT 