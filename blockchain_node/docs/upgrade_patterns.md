# Smart Contract Upgrade Patterns

This document provides comprehensive documentation for the smart contract upgrade patterns implemented in our blockchain platform, including UUPS (Universal Upgradeable Proxy Standard) and Diamond patterns.

## Table of Contents

- [Overview](#overview)
- [Upgrade Patterns](#upgrade-patterns)
  - [UUPS Pattern](#uups-pattern)
  - [Diamond Pattern](#diamond-pattern)
  - [Transparent Proxy Pattern](#transparent-proxy-pattern)
- [Implementation Architecture](#implementation-architecture)
- [Usage Examples](#usage-examples)
- [Security Considerations](#security-considerations)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Overview

Smart contract upgradeability is crucial for maintaining and improving deployed contracts without losing state or requiring user migration. Our platform implements three main upgrade patterns:

1. **UUPS (Universal Upgradeable Proxy Standard)** - Implementation-controlled upgrades
2. **Diamond Pattern** - Modular functionality with facets
3. **Transparent Proxy** - Admin-controlled upgrades

All patterns include:
- ✅ Storage layout compatibility checking
- ✅ Access control and authorization
- ✅ Version management and tracking
- ✅ Gas optimization
- ✅ Security validations

## Upgrade Patterns

### UUPS Pattern

The Universal Upgradeable Proxy Standard (EIP-1822) places upgrade logic in the implementation contract itself, providing a more secure and gas-efficient approach compared to transparent proxies.

#### Key Features

- **Implementation-controlled upgrades**: The implementation contract contains the upgrade logic
- **EIP-1967 compliance**: Uses standard storage slots for proxy metadata
- **Gas efficient**: No admin checks on every function call
- **Security**: Prevents unauthorized upgrades through implementation validation

#### Architecture

```
┌─────────────────┐    delegate    ┌──────────────────────┐
│   Proxy         │───────────────>│   Implementation     │
│                 │     calls      │                      │
│ - Storage slots │                │ - Upgrade logic      │
│ - Fallback      │                │ - Business logic     │
└─────────────────┘                └──────────────────────┘
```

#### Implementation Example

```rust
use blockchain_node::wasm::upgrade::{UpgradeManager, UpgradePattern};

// Create UUPS pattern
let pattern = UpgradePattern::new_uups(implementation_address, admin);
let mut manager = UpgradeManager::new(storage, proxy_address, version, pattern);

// Perform upgrade
let result = manager.upgrade(new_implementation, upgrade_data, admin).await?;
```

#### Required Implementation Functions

UUPS implementations must export these functions:

- `upgradeTo(address)` - Upgrade to new implementation
- `upgradeToAndCall(address, bytes)` - Upgrade and initialize
- `proxiableUUID()` - Return implementation UUID
- `admin()` - Return current admin address

### Diamond Pattern

The Diamond pattern (EIP-2535) allows contracts to be composed of multiple facets (contracts), each providing specific functionality. This enables modular upgrades and unlimited contract size.

#### Key Features

- **Modular architecture**: Separate facets for different functionality
- **Selective upgrades**: Upgrade individual functions without affecting others
- **Unlimited size**: Bypass contract size limits through modularity
- **Storage isolation**: Facets can have isolated storage spaces
- **Function routing**: Automatic routing based on function selectors

#### Architecture

```
                    ┌─────────────────┐
                    │   Diamond       │
                    │   Proxy         │
                    └─────────┬───────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        v                     v                     v
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   ERC20     │     │   ERC721    │     │   Loupe     │
│   Facet     │     │   Facet     │     │   Facet     │
│             │     │             │     │             │
│ - transfer  │     │ - mint      │     │ - facets    │
│ - approve   │     │ - burn      │     │ - selectors │
└─────────────┘     └─────────────┘     └─────────────┘
```

#### Implementation Example

```rust
// Create Diamond pattern
let mut cut_roles = HashSet::new();
cut_roles.insert(admin);
let pattern = UpgradePattern::new_diamond(storage_address, loupe_facet, cut_roles);

// Add new facet
let facet_cut = FacetCut {
    facet_address: erc20_facet,
    action: FacetCutAction::Add,
    function_selectors: vec![
        UpgradeManager::generate_function_selector("transfer(address,uint256)"),
        UpgradeManager::generate_function_selector("approve(address,uint256)"),
    ],
};

let result = manager.diamond_cut(vec![facet_cut], None, vec![], admin).await?;
```

#### Storage Management

The Diamond pattern supports multiple storage models:

1. **Diamond Storage**: Shared across all facets
2. **Facet Storage**: Isolated per facet
3. **Inherited Storage**: From base contracts

```rust
let storage_manager = DiamondStorageManager::new(storage, diamond_address);

// Set facet-specific storage
storage_manager.set_facet_storage(&erc20_facet, key, value)?;

// Set shared diamond storage
storage_manager.set_diamond_storage(key, value)?;
```

### Transparent Proxy Pattern

The transparent proxy pattern separates admin and user interfaces, routing calls based on the caller's identity.

#### Key Features

- **Role separation**: Admin functions vs user functions
- **Automatic routing**: Based on caller identity
- **Simple implementation**: Easy to understand and audit
- **Gas overhead**: Admin checks on every call

#### Implementation Example

```rust
let pattern = UpgradePattern::new_transparent_proxy(implementation, admin);
let mut manager = UpgradeManager::new(storage, proxy_address, version, pattern);
```

## Implementation Architecture

### Core Components

1. **UpgradeManager**: Central coordinator for upgrade operations
2. **UpgradePattern**: Enum defining the upgrade strategy
3. **ContractVersion**: Version tracking with metadata
4. **WasmStorage**: Storage abstraction for contract data
5. **DiamondStorageManager**: Specialized storage for Diamond pattern

### Storage Layout Compatibility

The system automatically checks storage layout compatibility during upgrades:

```rust
pub struct StorageVariable {
    pub name: String,
    pub var_type: String,
    pub slot: u32,
    pub offset: u32,
    pub size: u32,
    pub is_constant: bool,
}
```

#### Compatibility Rules

- ✅ **Safe**: Adding new variables at the end
- ✅ **Safe**: Increasing variable size
- ❌ **Unsafe**: Changing variable order
- ❌ **Unsafe**: Changing variable types
- ❌ **Unsafe**: Removing variables

### Access Control

All upgrade patterns implement comprehensive access control:

- **UUPS**: Admin address stored in implementation
- **Diamond**: Role-based permissions for diamond cuts
- **Transparent**: Admin-only upgrade functions

## Usage Examples

### Running Examples

The platform includes comprehensive examples demonstrating each pattern:

```bash
# UUPS Pattern Example
cargo run --example uups_proxy_example

# Diamond Pattern Example  
cargo run --example diamond_proxy_example
```

### UUPS Example Output

```
🎯 UUPS Proxy Pattern Example
==============================
🚀 Deploying UUPS Proxy...
✅ UUPS Proxy deployed at: uups_proxy_v1
📋 Initial implementation: implementation_v1
👤 Admin: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

🔄 Upgrading UUPS implementation...
✅ Upgrade completed!
📋 New implementation: implementation_v2
⛽ Gas used: 50000
```

### Diamond Example Output

```
💎 Diamond Proxy Pattern Example
=================================
💎 Deploying Diamond Proxy...
✅ Diamond Proxy deployed at: diamond_proxy

💰 Adding ERC20 Facet...
✅ ERC20 Facet added successfully!
📋 Functions added:
  - transfer(address,uint256)
  - transferFrom(address,address,uint256)
  - approve(address,uint256)
  - balanceOf(address)
  - totalSupply()
⛽ Gas used: 75000
```

## Security Considerations

### 1. Storage Collision Prevention

- Use EIP-1967 standard storage slots
- Implement storage layout validation
- Avoid overlapping storage between proxy and implementation

### 2. Function Selector Conflicts

- Validate function selectors don't conflict
- Use proper namespacing for Diamond facets
- Implement selector collision detection

### 3. Access Control

- Implement proper admin role management
- Use role-based access for Diamond cuts
- Validate upgrade authorization before execution

### 4. Implementation Validation

- Verify implementation contracts have required functions
- Validate bytecode before deployment
- Check implementation compatibility

## Best Practices

### 1. Version Management

```rust
// Always increment version numbers
let new_version = ContractVersion {
    version: current_version.version + 1,
    implementation: new_implementation,
    storage_layout_hash: new_layout_hash,
    upgraded_at: current_timestamp(),
    upgraded_by: admin,
    migration_hash: Some(calculate_migration_hash(&upgrade_data)),
    notes: Some("Added new feature X".to_string()),
};
```

### 2. Testing Strategy

- Test all upgrade paths thoroughly
- Validate storage compatibility
- Test access control mechanisms
- Perform gas optimization analysis

### 3. Documentation

- Document all storage variables
- Maintain upgrade history
- Document breaking changes
- Provide migration guides

### 4. Monitoring

- Track upgrade events
- Monitor gas usage
- Log all administrative actions
- Implement upgrade notifications

## API Reference

### UpgradeManager

Main interface for managing contract upgrades.

#### Methods

```rust
pub fn new(
    storage: Arc<dyn WasmStorage>,
    contract_address: WasmContractAddress,
    current_version: ContractVersion,
    pattern: UpgradePattern,
) -> Self

pub async fn upgrade(
    &mut self,
    new_implementation: WasmContractAddress,
    upgrade_data: Vec<u8>,
    caller: Address,
) -> Result<WasmExecutionResult, WasmError>

pub async fn diamond_cut(
    &mut self,
    facet_cuts: Vec<FacetCut>,
    init_address: Option<WasmContractAddress>,
    init_data: Vec<u8>,
    caller: Address,
) -> Result<WasmExecutionResult, WasmError>
```

### UpgradePattern

Enum defining upgrade strategies.

```rust
pub enum UpgradePattern {
    UUPS {
        implementation: WasmContractAddress,
        admin: Address,
        implementation_slot: StorageSlot,
        admin_slot: StorageSlot,
    },
    Diamond {
        diamond_storage: WasmContractAddress,
        facets: HashMap<FunctionSelector, DiamondFacet>,
        cut_roles: HashSet<Address>,
        loupe_facet: WasmContractAddress,
    },
    TransparentProxy {
        implementation: WasmContractAddress,
        admin: Address,
    },
}
```

#### Helper Methods

```rust
impl UpgradePattern {
    pub fn new_uups(implementation: WasmContractAddress, admin: Address) -> Self
    pub fn new_diamond(storage: WasmContractAddress, loupe: WasmContractAddress, roles: HashSet<Address>) -> Self
    pub fn new_transparent_proxy(implementation: WasmContractAddress, admin: Address) -> Self
    pub fn get_implementation(&self) -> &WasmContractAddress
    pub fn get_admin(&self) -> Option<&Address>
}
```

### DiamondStorageManager

Specialized storage manager for Diamond pattern.

```rust
impl DiamondStorageManager {
    pub fn new(storage: Arc<dyn WasmStorage>, diamond_address: WasmContractAddress) -> Self
    pub fn get_facet_storage(&self, facet: &WasmContractAddress, key: &[u8]) -> Result<Option<Vec<u8>>, WasmError>
    pub fn set_facet_storage(&self, facet: &WasmContractAddress, key: &[u8], value: &[u8]) -> Result<(), WasmError>
    pub fn get_diamond_storage(&self, key: &[u8]) -> Result<Option<Vec<u8>>, WasmError>
    pub fn set_diamond_storage(&self, key: &[u8], value: &[u8]) -> Result<(), WasmError>
}
```

### UniversalProxy

Universal proxy contract supporting multiple patterns.

```rust
impl UniversalProxy {
    pub fn new(storage: Arc<dyn WasmStorage>, proxy_address: WasmContractAddress, pattern: UpgradePattern) -> Self
    pub async fn delegate_call(&self, selector: &FunctionSelector, call_data: &[u8], caller: Address) -> Result<WasmExecutionResult, WasmError>
    pub fn supports_function(&self, selector: &FunctionSelector) -> bool
    pub fn get_implementation_for_function(&self, selector: &FunctionSelector) -> Option<WasmContractAddress>
}
```

### Utility Functions

```rust
// Generate function selector from signature
pub fn generate_function_selector(signature: &str) -> FunctionSelector

// EIP-1967 standard storage slots
pub struct EIP1967Slots {
    pub const IMPLEMENTATION_SLOT: StorageSlot = [...];
    pub const ADMIN_SLOT: StorageSlot = [...];
}
```

## Conclusion

This comprehensive upgrade pattern implementation provides enterprise-grade smart contract upgradeability with:

- **Multiple upgrade strategies** for different use cases
- **Robust security measures** including access control and validation
- **Storage compatibility checking** to prevent data corruption
- **Gas optimization** for cost-effective operations
- **Comprehensive testing** with example implementations

The system supports both simple proxy upgrades and complex modular contracts, enabling developers to choose the appropriate pattern for their specific requirements while maintaining security and efficiency.

For more information, see the example implementations in the `examples/` directory and the test suite in `src/wasm/upgrade.rs`. 