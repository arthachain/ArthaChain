use std::sync::Arc;
use std::time::{Instant, Duration};
use blockchain_node::network::dos_protection::{DOSProtector, SecurityMetrics, RequestInfo, RequestType};
use libp2p::PeerId;
use tokio::runtime::Runtime;

fn main() {
    println!("Running DOS Protection Benchmark");
    println!("===============================\n");
    
    // Create a runtime for running our tests
    let runtime = Runtime::new().unwrap();
    
    // Basic performance test to show how fast our rate limiter is
    performance_test();
    
    // Create a protector with the test version that will properly accumulate bytes
    let metrics = Arc::new(SecurityMetrics::new());
    let protector = DOSProtector::new(metrics);
    
    println!("\n1. Byte Limit Test With Our Fix (Proper Accumulation):");
    println!("--------------------------------------------------");
    println!("Each test sends a series of requests with specific byte sizes to verify proper rate limiting");
    
    // Run test scenarios to verify our fixes
    runtime.block_on(async {
        sequential_byte_limit_test(&protector).await;
    });
    
    println!("\n2. Performance Comparison With Different Request Sizes:");
    println!("--------------------------------------------------");
    let request_sizes = vec![10, 100, 500, 1000];
    
    println!("| Request Size | Requests/sec | Time for 100,000 requests (ms) |");
    println!("|--------------|--------------|--------------------------------|");
    
    for &size in &request_sizes {
        // Using 100,000 requests for a more accurate measurement
        let num_requests = 100_000;
        let start = Instant::now();
        
        runtime.block_on(async {
            test_throughput(&protector, size, num_requests).await;
        });
        
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_millis();
        let requests_per_sec = (num_requests as f64 / elapsed.as_secs_f64()) as u64;
        
        println!("| {:<12} | {:<12} | {:<30} |", size, requests_per_sec, elapsed_ms);
    }
    
    println!("\nBenchmark complete!");
}

fn performance_test() {
    println!("Basic Performance Test:");
    println!("---------------------");
    
    // Create a runtime for the benchmark
    let runtime = Runtime::new().unwrap();
    
    // Create a protector
    let metrics = Arc::new(SecurityMetrics::new());
    let protector = DOSProtector::new(metrics);
    
    // Benchmarking parameters
    let num_requests = 10_000;
    let request_size = 100;
    
    let start = Instant::now();
    
    runtime.block_on(async {
        // Use different peers to avoid rate limiting affecting our performance test
        for i in 0..num_requests {
            // Create a unique peer for each request to avoid rate limiting
            let peer_id = PeerId::random();
            
            let request = RequestInfo {
                request_type: RequestType::Transaction,
                timestamp: 0,
                size: request_size,
                source_ip: "127.0.0.1".to_string(),
            };
            
            // Process the request
            let _ = protector.check_request(peer_id, request).await;
            
            // Small delay to keep runtime responsive
            if i % 1000 == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    });
    
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_millis();
    let requests_per_sec = (num_requests as f64 / elapsed.as_secs_f64()) as u64;
    
    println!("Processed {} requests of {} bytes each", num_requests, request_size);
    println!("Total time: {} ms", elapsed_ms);
    println!("Performance: {} requests/second", requests_per_sec);
}

async fn sequential_byte_limit_test(protector: &DOSProtector) {
    // Test with fixed data_per_second limit of 500 bytes
    // We'll modify this limit in the RateLimiter::new() function
    
    println!("Testing with data_per_second=500 bytes (from RateLimiter::new())");
    println!("| Test Case                      | Expected | Result |");
    println!("|--------------------------------|----------|--------|");
    
    // Test 1: 400 byte request - should be accepted
    let peer1 = PeerId::random();
    let request1 = RequestInfo {
        request_type: RequestType::Transaction,
        timestamp: 0,
        size: 400,
        source_ip: "127.0.0.1".to_string(),
    };
    
    let result1 = match protector.check_request(peer1.clone(), request1).await {
        Ok(_) => {
            println!("  First request of 400 bytes ACCEPTED");
            "PASS"
        },
        Err(e) => {
            println!("  First request REJECTED unexpectedly: {}", e);
            "FAIL"
        },
    };
    println!("| Single 400 byte request        | Accept   | {:<6} |", result1);
    
    // Small delay to ensure we're in the same time window
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    
    // Test 2: Second request of 200 bytes (total 600) - should be rejected due to our fix
    let request2 = RequestInfo {
        request_type: RequestType::Transaction,
        timestamp: 0,
        size: 200, 
        source_ip: "127.0.0.1".to_string(),
    };
    
    let result2 = match protector.check_request(peer1.clone(), request2).await {
        Ok(_) => {
            println!("  Second request ACCEPTED unexpectedly (should be rejected)");
            "FAIL" // Should reject, so acceptance is a failure
        },
        Err(e) => {
            println!("  Second request of 200 bytes correctly REJECTED: {}", e);
            "PASS"
        },
    };
    println!("| Second request (400+200 bytes) | Reject   | {:<6} |", result2);
    
    // Test 3: Create another peer for a fresh test
    println!("\n| Sequential Requests Test        | Expected | Result |");
    println!("|--------------------------------|----------|--------|");
    
    let peer2 = PeerId::random();
    
    // Reset the DOSProtector to start with a clean state for this test
    let metrics = Arc::new(SecurityMetrics::new());
    let clean_protector = DOSProtector::new(metrics);
    
    // Test sending 6 sequential requests of 100 bytes each - first 5 should be accepted, 6th rejected
    let mut results = Vec::new();
    
    println!("  Starting sequential test with 6 requests of 100 bytes each");
    for i in 0..6 {
        let request = RequestInfo {
            request_type: RequestType::Transaction, // Using the same type for all requests
            timestamp: 0,
            size: 100,
            source_ip: "127.0.0.1".to_string(),
        };
        
        // Small delay to ensure sequential processing
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        match clean_protector.check_request(peer2.clone(), request).await {
            Ok(_) => {
                println!("  Request #{} (100 bytes): ACCEPTED (Total so far: {} bytes)", i+1, (i+1) * 100);
                results.push("Accepted");
            },
            Err(e) => {
                println!("  Request #{} (100 bytes): REJECTED - {}", i+1, e);
                results.push("Rejected");
            },
        }
    }
    
    // We expect the first 5 requests (500 bytes total) to be accepted and the 6th to be rejected
    let expected = vec!["Accepted", "Accepted", "Accepted", "Accepted", "Accepted", "Rejected"];
    let result3 = if results == expected { "PASS" } else { "FAIL" };
    let actual_pattern = format!("{}A/{}R", 
        results.iter().filter(|&r| *r == "Accepted").count(),
        results.iter().filter(|&r| *r == "Rejected").count()
    );
    println!("| Six 100-byte requests          | 5A/1R    | {:<6} ({}) |", result3, actual_pattern);
    
    // Test 4: Create another peer and test with delay between requests
    let peer3 = PeerId::random();
    println!("\nTime Window Reset Test:");
    println!("| Test Case                      | Expected | Result |");
    println!("|--------------------------------|----------|--------|");
    
    // Create a fresh protector instance for this test
    let fresh_metrics = Arc::new(SecurityMetrics::new());
    let time_window_protector = DOSProtector::new(fresh_metrics);
    
    // First request of 400 bytes (should be accepted)
    let request4 = RequestInfo {
        request_type: RequestType::Transaction,
        timestamp: 0,
        size: 400,
        source_ip: "127.0.0.1".to_string(),
    };
    
    let result4 = match time_window_protector.check_request(peer3.clone(), request4).await {
        Ok(_) => {
            println!("  First request of 400 bytes ACCEPTED");
            "PASS"
        },
        Err(e) => {
            println!("  First request REJECTED unexpectedly: {}", e);
            "FAIL"
        },
    };
    println!("| First request (400 bytes)      | Accept   | {:<6} |", result4);
    
    // Wait for 2 seconds to reset the rate limit window
    println!("  Waiting 2 seconds to reset rate limit window...");
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    
    // Second request of 400 bytes (should be accepted since window reset)
    let request5 = RequestInfo {
        request_type: RequestType::Transaction,
        timestamp: 0,
        size: 400, 
        source_ip: "127.0.0.1".to_string(),
    };
    
    let result5 = match time_window_protector.check_request(peer3.clone(), request5).await {
        Ok(_) => {
            println!("  Second request after delay of 400 bytes ACCEPTED");
            "PASS"
        },
        Err(e) => {
            println!("  Second request after delay REJECTED unexpectedly: {}", e);
            "FAIL"
        },
    };
    println!("| Second request after delay     | Accept   | {:<6} |", result5);
}

async fn test_throughput(protector: &DOSProtector, request_size: u64, num_requests: u32) {
    for i in 0..num_requests {
        // Create a unique peer for each request to avoid rate limiting
        let peer_id = PeerId::random();
        
        let request = RequestInfo {
            request_type: RequestType::Transaction,
            timestamp: 0,
            size: request_size,
            source_ip: "127.0.0.1".to_string(),
        };
        
        // Process the request
        let _ = protector.check_request(peer_id, request).await;
        
        // Small delay to keep runtime responsive
        if i % 10000 == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
} 