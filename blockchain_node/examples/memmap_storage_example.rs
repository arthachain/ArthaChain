//! Memory-Mapped Storage Example
//!
//! This example demonstrates the usage of memory-mapped storage for
//! efficient data persistence and retrieval using the Storage trait.

use anyhow::Result;
use blockchain_node::storage::{MemMapOptions, MemMapStorage, Storage};
use blockchain_node::types::Hash;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🗄️ Memory-Mapped Storage Example");
    println!("{}", "=".repeat(50));

    // Create storage with default options
    let options = MemMapOptions::default();
    let storage = MemMapStorage::new(options);

    println!("\n📊 Storage Configuration:");
    println!("   • Memory-mapped file storage");
    println!("   • Automatic persistence");
    println!("   • High-performance I/O");

    // Test basic operations
    test_basic_operations(&storage).await?;

    // Test bulk operations
    test_bulk_operations(&storage).await?;

    // Test performance
    test_performance(&storage).await?;

    println!("\n✅ All memory-mapped storage tests completed successfully!");

    Ok(())
}

async fn test_basic_operations(storage: &MemMapStorage) -> Result<()> {
    println!("\n🔧 Testing Basic Operations");
    println!("{}", "-".repeat(30));

    // Create some test data with key-value structure
    let key = b"test_key";
    let value = b"test_value_12345";

    // Create composite data (key + value)
    let mut data = Vec::new();
    data.extend_from_slice(&(key.len() as u32).to_le_bytes());
    data.extend_from_slice(&(value.len() as u32).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(value);

    println!("1. Storing data...");
    let hash = storage.store(&data).await?;
    println!("   ✅ Data stored with hash: {}", hex::encode(&hash.0));

    println!("2. Retrieving data...");
    let retrieved = storage.retrieve(&hash).await?;

    match retrieved {
        Some(retrieved_data) => {
            println!("   ✅ Retrieved {} bytes", retrieved_data.len());

            // Parse the composite data
            if retrieved_data.len() >= 8 {
                let key_len = u32::from_le_bytes([
                    retrieved_data[0],
                    retrieved_data[1],
                    retrieved_data[2],
                    retrieved_data[3],
                ]) as usize;
                let value_len = u32::from_le_bytes([
                    retrieved_data[4],
                    retrieved_data[5],
                    retrieved_data[6],
                    retrieved_data[7],
                ]) as usize;

                if retrieved_data.len() >= 8 + key_len + value_len {
                    let retrieved_key = &retrieved_data[8..8 + key_len];
                    let retrieved_value = &retrieved_data[8 + key_len..8 + key_len + value_len];

                    println!("   📁 Key: {}", String::from_utf8_lossy(retrieved_key));
                    println!("   📄 Value: {}", String::from_utf8_lossy(retrieved_value));

                    assert_eq!(retrieved_key, key, "Retrieved key should match stored key");
                    assert_eq!(
                        retrieved_value, value,
                        "Retrieved value should match stored value"
                    );
                }
            }
        }
        None => {
            println!("   ❌ No data found");
            return Err(anyhow::anyhow!("Expected data not found"));
        }
    }

    println!("3. Checking if data exists...");
    let exists = storage.exists(&hash).await?;
    assert!(exists, "Data should exist");
    println!("   ✅ Data exists: {}", exists);

    println!("4. Verifying data integrity...");
    let is_valid = storage.verify(&hash, &data).await?;
    assert!(is_valid, "Data should be valid");
    println!("   ✅ Data integrity verified: {}", is_valid);

    println!("5. Deleting data...");
    storage.delete(&hash).await?;

    let exists_after_delete = storage.exists(&hash).await?;
    assert!(!exists_after_delete, "Data should not exist after deletion");
    println!("   ✅ Data deleted successfully");

    Ok(())
}

async fn test_bulk_operations(storage: &MemMapStorage) -> Result<()> {
    println!("\n📦 Testing Bulk Operations");
    println!("{}", "-".repeat(30));

    let start = Instant::now();
    let mut hashes = Vec::new();

    // Store multiple data entries
    println!("1. Storing 100 data entries...");
    for i in 0..100 {
        let key = format!("bulk_key_{:03}", i);
        let value = format!("bulk_value_{:03}_data_payload", i);

        // Create composite data
        let mut data = Vec::new();
        data.extend_from_slice(&(key.len() as u32).to_le_bytes());
        data.extend_from_slice(&(value.len() as u32).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(value.as_bytes());

        let hash = storage.store(&data).await?;
        hashes.push(hash);
    }

    let store_time = start.elapsed();
    println!("   ✅ Stored 100 items in {:?}", store_time);

    // Retrieve all items
    println!("2. Retrieving all stored items...");
    let retrieve_start = Instant::now();
    let mut retrieved_count = 0;

    for hash in &hashes {
        if (storage.retrieve(hash).await?).is_some() {
            retrieved_count += 1;
        }
    }

    let retrieve_time = retrieve_start.elapsed();
    println!(
        "   ✅ Retrieved {} items in {:?}",
        retrieved_count, retrieve_time
    );
    assert_eq!(retrieved_count, 100, "Should retrieve all stored items");

    // Test batch deletion
    println!("3. Deleting all bulk items...");
    let delete_start = Instant::now();

    for hash in &hashes {
        storage.delete(hash).await?;
    }

    let delete_time = delete_start.elapsed();
    println!("   ✅ Deleted 100 items in {:?}", delete_time);

    Ok(())
}

async fn test_performance(storage: &MemMapStorage) -> Result<()> {
    println!("\n⚡ Performance Testing");
    println!("{}", "-".repeat(30));

    let num_operations = 1000;
    let data_size = 1024; // 1KB per entry

    // Generate test data
    println!("1. Generating test data...");
    let test_key = "performance_test_key";
    let test_value = vec![0xAB; data_size];

    // Create composite data
    let mut test_data = Vec::new();
    test_data.extend_from_slice(&(test_key.len() as u32).to_le_bytes());
    test_data.extend_from_slice(&(test_value.len() as u32).to_le_bytes());
    test_data.extend_from_slice(test_key.as_bytes());
    test_data.extend_from_slice(&test_value);

    // Write performance test
    println!(
        "2. Testing write performance ({} operations)...",
        num_operations
    );
    let write_start = Instant::now();
    let mut write_hashes = Vec::new();

    for i in 0..num_operations {
        // Vary the data slightly for each operation
        let mut data = test_data.clone();
        data.extend_from_slice(&(i as u32).to_le_bytes());

        let hash = storage.store(&data).await?;
        write_hashes.push(hash);
    }

    let write_duration = write_start.elapsed();
    let write_ops_per_sec = num_operations as f64 / write_duration.as_secs_f64();

    println!(
        "   ✅ Write: {:.2} ops/sec ({:?} total)",
        write_ops_per_sec, write_duration
    );

    // Read performance test
    println!(
        "3. Testing read performance ({} operations)...",
        num_operations
    );
    let read_start = Instant::now();

    for hash in &write_hashes {
        let _ = storage.retrieve(hash).await?;
    }

    let read_duration = read_start.elapsed();
    let read_ops_per_sec = num_operations as f64 / read_duration.as_secs_f64();

    println!(
        "   ✅ Read: {:.2} ops/sec ({:?} total)",
        read_ops_per_sec, read_duration
    );

    // Mixed workload test
    println!("4. Testing mixed workload (70% reads, 30% writes)...");
    let mixed_start = Instant::now();
    let mixed_operations = 500;
    let mut mixed_hashes = Vec::new();

    for i in 0..mixed_operations {
        if i % 10 < 7 && !write_hashes.is_empty() {
            // Read operation
            let hash_index = i % write_hashes.len();
            let _ = storage.retrieve(&write_hashes[hash_index]).await?;
        } else {
            // Write operation
            let mut data = test_data.clone();
            data.extend_from_slice(&(i as u32 + 10000).to_le_bytes());

            let hash = storage.store(&data).await?;
            mixed_hashes.push(hash);
        }
    }

    let mixed_duration = mixed_start.elapsed();
    let mixed_ops_per_sec = mixed_operations as f64 / mixed_duration.as_secs_f64();

    println!(
        "   ✅ Mixed: {:.2} ops/sec ({:?} total)",
        mixed_ops_per_sec, mixed_duration
    );

    // Cleanup performance test data
    println!("5. Cleaning up test data...");
    for hash in &write_hashes {
        storage.delete(hash).await.ok(); // Ignore errors for cleanup
    }

    for hash in &mixed_hashes {
        storage.delete(hash).await.ok(); // Ignore errors for cleanup
    }

    println!("   ✅ Cleanup completed");

    Ok(())
}

/// Create a hash from data (utility function for testing)
fn _create_hash(data: &[u8]) -> Hash {
    let hash_bytes = blake3::hash(data);
    Hash::from_data(hash_bytes.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_storage_basic_functionality() {
        let storage = MemMapStorage::new(MemMapOptions::default());

        let test_data = b"test_data";

        // Test store and retrieve
        let hash = storage.store(test_data).await.unwrap();
        let retrieved = storage.retrieve(&hash).await.unwrap().unwrap();
        assert_eq!(retrieved, test_data);

        // Test exists
        let exists = storage.exists(&hash).await.unwrap();
        assert!(exists);

        // Test verify
        let is_valid = storage.verify(&hash, test_data).await.unwrap();
        assert!(is_valid);

        // Test delete
        storage.delete(&hash).await.unwrap();
        let exists_after_delete = storage.exists(&hash).await.unwrap();
        assert!(!exists_after_delete);
    }

    // Test Hash conversion
    #[test]
    fn test_hash_conversion() {
        let test_bytes = b"hello world";
        let blake3_hash = blake3::hash(test_bytes);
        let hash_bytes = blake3_hash.as_bytes();
        let hash = Hash::from_data(hash_bytes);

        assert_eq!(hash.as_ref(), hash_bytes);
    }
}
