#[cfg(test)]
mod tests {
    use crate::network::custom_udp::MessageFlags;

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
} 