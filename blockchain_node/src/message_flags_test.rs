#[derive(Debug, Clone, Copy)]
pub struct MessageFlags {
    bits: u16,
}

impl MessageFlags {
    pub const REQUEST_ACK: Self = Self { bits: 0x0001 };
    pub const IS_ACK: Self = Self { bits: 0x0002 };
    pub const ENCRYPTED: Self = Self { bits: 0x0004 };
    pub const COMPRESSED: Self = Self { bits: 0x0008 };
    pub const FRAGMENT: Self = Self { bits: 0x0010 };
    pub const LAST_FRAGMENT: Self = Self { bits: 0x0020 };
    pub const HIGH_PRIORITY: Self = Self { bits: 0x0040 };
    pub const NO_RETRY: Self = Self { bits: 0x0080 };
    pub const RELAY: Self = Self { bits: 0x0100 };
    pub const SIGNED: Self = Self { bits: 0x0200 };
    
    pub fn empty() -> Self {
        Self { bits: 0 }
    }
    
    pub fn from_bits_truncate(bits: u16) -> Self {
        Self { bits }
    }
    
    pub fn bits(&self) -> u16 {
        self.bits
    }
    
    pub fn contains(&self, other: Self) -> bool {
        (self.bits & other.bits) == other.bits
    }
    
    pub fn insert(&mut self, other: Self) {
        self.bits |= other.bits;
    }
    
    pub fn remove(&mut self, other: Self) {
        self.bits &= !other.bits;
    }
    
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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