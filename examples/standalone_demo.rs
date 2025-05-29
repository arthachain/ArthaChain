//! Standalone Network Monitoring Demo

use std::sync::Once;
use std::time::{Duration, SystemTime};

static mut NODE_START_TIME: Option<SystemTime> = None;
static INIT: Once = Once::new();

fn init_node_start_time() {
    unsafe {
        INIT.call_once(|| {
            NODE_START_TIME = Some(SystemTime::now());
        });
    }
}

fn get_node_start_time() -> Option<SystemTime> {
    unsafe { NODE_START_TIME }
}

#[derive(Debug, Clone)]
enum NetworkHealthStatus {
    Healthy,
    Warning,
    Critical,
    Offline,
}

#[derive(Debug, Clone)]
enum MempoolHealthStatus {
    Normal,
    Busy,
    Congested,
    Full,
}

fn format_duration(seconds: u64) -> String {
    let days = seconds / 86400;
    let hours = (seconds % 86400) / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;

    if days > 0 {
        format!("{days}d {hours}h {minutes}m {secs}s")
    } else if hours > 0 {
        format!("{hours}h {minutes}m {secs}s")
    } else if minutes > 0 {
        format!("{minutes}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

fn assess_network_health(peer_count: usize, min_peers: usize) -> NetworkHealthStatus {
    match peer_count {
        0 => NetworkHealthStatus::Offline,
        n if n < min_peers => NetworkHealthStatus::Critical,
        n if n < min_peers * 2 => NetworkHealthStatus::Warning,
        _ => NetworkHealthStatus::Healthy,
    }
}

fn assess_mempool_health(utilization_percent: f64) -> MempoolHealthStatus {
    match utilization_percent {
        p if p >= 95.0 => MempoolHealthStatus::Full,
        p if p >= 80.0 => MempoolHealthStatus::Congested,
        p if p >= 60.0 => MempoolHealthStatus::Busy,
        _ => MempoolHealthStatus::Normal,
    }
}

fn main() {
    println!("🚀 Network Monitoring Demo");
    println!("{}", "=".repeat(50));

    // Initialize uptime tracking
    init_node_start_time();

    // Wait a moment to show some uptime
    std::thread::sleep(Duration::from_millis(200));

    println!("\n📊 Network Status Dashboard");
    println!("{}", "-".repeat(30));

    // Demo blockchain height
    let height = 1337;
    println!("🔗 Blockchain Height: {height}");

    // Demo peer information
    let peer_count = 8;
    let min_peers = 3;
    let network_health = assess_network_health(peer_count, min_peers);
    println!("👥 Connected Peers: {peer_count}");
    println!("🌐 Network Health: {network_health:?}");

    // Demo mempool information
    let transaction_count = 25;
    let size_bytes = 51200; // 50KB
    let max_size_bytes = 10485760; // 10MB
    let utilization = (size_bytes as f64 / max_size_bytes as f64) * 100.0;
    let mempool_health = assess_mempool_health(utilization);

    println!("💾 Mempool Transactions: {transaction_count}");
    println!("📏 Mempool Size: {size_bytes} bytes");
    println!("📈 Utilization: {utilization:.2}%");
    println!("🎯 Mempool Health: {mempool_health:?}");

    // Demo uptime
    if let Some(start_time) = get_node_start_time() {
        let current_time = SystemTime::now();
        if let Ok(uptime_duration) = current_time.duration_since(start_time) {
            let uptime_seconds = uptime_duration.as_secs();
            let formatted = format_duration(uptime_seconds);
            println!("⏰ Node Uptime: {formatted}");
            println!("🕐 Uptime Seconds: {uptime_seconds}");
        }
    }

    // Demo health assessment examples
    println!("\n🏥 Health Assessment Examples");
    println!("{}", "-".repeat(30));

    let test_cases = vec![
        (0, "No peers - Network Offline"),
        (2, "Below minimum - Critical"),
        (5, "Low peers - Warning"),
        (15, "Good connectivity - Healthy"),
    ];

    for (peer_count, description) in test_cases {
        let health = assess_network_health(peer_count, 3);
        println!("📊 {peer_count} peers: {health:?} - {description}");
    }

    println!("\n💾 Mempool Health Examples");
    println!("{}", "-".repeat(30));

    let utilization_tests = vec![
        (30.0, "Normal operations"),
        (65.0, "Busy period"),
        (85.0, "High congestion"),
        (97.0, "At capacity"),
    ];

    for (util, description) in utilization_tests {
        let health = assess_mempool_health(util);
        println!("📈 {util:.1}% utilization: {health:?} - {description}");
    }

    println!("\n✅ Demo completed successfully!");
    println!("🎯 All monitoring features working correctly!");

    // Demo API endpoint simulation
    println!("\n🌐 API Endpoint Simulation");
    println!("{}", "-".repeat(30));

    // Simulate JSON responses that would be returned by our API endpoints
    println!("GET /api/monitoring/peers/count would return:");
    println!("{{");
    println!("  \"peer_count\": {peer_count},");
    println!("  \"max_peers\": 50,");
    println!("  \"min_peers\": {min_peers},");
    println!("  \"network_health\": \"{network_health:?}\"");
    println!("}}");

    println!("\nGET /api/monitoring/mempool/size would return:");
    println!("{{");
    println!("  \"transaction_count\": {transaction_count},");
    println!("  \"size_bytes\": {size_bytes},");
    println!("  \"max_size_bytes\": {max_size_bytes},");
    println!("  \"utilization_percent\": {utilization:.2},");
    println!("  \"health_status\": \"{mempool_health:?}\"");
    println!("}}");

    if let Some(start_time) = get_node_start_time() {
        let current_time = SystemTime::now();
        if let Ok(uptime_duration) = current_time.duration_since(start_time) {
            let uptime_seconds = uptime_duration.as_secs();
            let formatted = format_duration(uptime_seconds);

            println!("\nGET /api/monitoring/uptime would return:");
            println!("{{");
            println!("  \"uptime_seconds\": {uptime_seconds},");
            println!("  \"uptime_formatted\": \"{formatted}\",");
            println!(
                "  \"start_timestamp\": {},",
                start_time
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            );
            println!(
                "  \"current_timestamp\": {}",
                current_time
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            );
            println!("}}");
        }
    }

    println!("\n🎉 All API monitoring functionality demonstrated!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(3661), "1h 1m 1s");
        assert_eq!(format_duration(90061), "1d 1h 1m 1s");
    }

    #[test]
    fn test_health_assessment() {
        // Test network health
        assert!(matches!(
            assess_network_health(0, 3),
            NetworkHealthStatus::Offline
        ));
        assert!(matches!(
            assess_network_health(2, 3),
            NetworkHealthStatus::Critical
        ));
        assert!(matches!(
            assess_network_health(4, 3),
            NetworkHealthStatus::Warning
        ));
        assert!(matches!(
            assess_network_health(10, 3),
            NetworkHealthStatus::Healthy
        ));

        // Test mempool health
        assert!(matches!(
            assess_mempool_health(30.0),
            MempoolHealthStatus::Normal
        ));
        assert!(matches!(
            assess_mempool_health(65.0),
            MempoolHealthStatus::Busy
        ));
        assert!(matches!(
            assess_mempool_health(85.0),
            MempoolHealthStatus::Congested
        ));
        assert!(matches!(
            assess_mempool_health(97.0),
            MempoolHealthStatus::Full
        ));
    }
}
