use super::*;
use custom_udp::MessageFlags;

#[test]
fn test_message_flags() {
    // Test the MessageFlags implementation
    let mut flags = MessageFlags::empty();
    assert!(flags.is_empty());
    
    flags.insert(MessageFlags::REQUEST_ACK);
    assert!(flags.contains(MessageFlags::REQUEST_ACK));
    assert!(!flags.contains(MessageFlags::IS_ACK));
    
    flags.insert(MessageFlags::IS_ACK);
    assert!(flags.contains(MessageFlags::REQUEST_ACK));
    assert!(flags.contains(MessageFlags::IS_ACK));
    
    flags.remove(MessageFlags::REQUEST_ACK);
    assert!(!flags.contains(MessageFlags::REQUEST_ACK));
    assert!(flags.contains(MessageFlags::IS_ACK));
    
    // Test bit combinations
    let mut combined = MessageFlags::empty();
    combined.insert(MessageFlags::HIGH_PRIORITY);
    combined.insert(MessageFlags::ENCRYPTED);
    assert!(combined.contains(MessageFlags::HIGH_PRIORITY));
    assert!(combined.contains(MessageFlags::ENCRYPTED));
    assert!(!combined.contains(MessageFlags::IS_ACK));
}

#[test]
fn test_network_stats() {
    let stats = NetworkStats {
        active_connections: 5,
        bytes_sent: 1024,
        bytes_received: 2048, 
        avg_latency_ms: 50.0,
        success_rate: 0.98,
        blocks_received: 10,
        transactions_received: 100,
        last_activity: chrono::Utc::now(),
    };
    
    assert_eq!(stats.total_bytes(), 3072);
    assert_eq!(stats.active_connections, 5);
    assert_eq!(stats.avg_latency_ms, 50.0);
} 