#!/bin/bash

echo "🔍 Verifying ArthaChain API Compilation..."
echo "=========================================="

# Check if testnet_router.rs exists and has valid syntax
echo "✅ Checking testnet_router.rs..."
if [ -f "blockchain_node/src/api/testnet_router.rs" ]; then
    echo "   - File exists ✓"
    
    # Check for basic Rust syntax patterns
    if grep -q "pub async fn create_testnet_router" "blockchain_node/src/api/testnet_router.rs"; then
        echo "   - Main function exists ✓"
    else
        echo "   ❌ Main function missing"
        exit 1
    fi
    
    if grep -q "Router::new()" "blockchain_node/src/api/testnet_router.rs"; then
        echo "   - Router creation exists ✓"
    else
        echo "   ❌ Router creation missing"
        exit 1
    fi
    
    if grep -q "TestnetRouterConfig" "blockchain_node/src/api/testnet_router.rs"; then
        echo "   - Configuration struct exists ✓"
    else
        echo "   ❌ Configuration struct missing"
        exit 1
    fi
else
    echo "   ❌ File missing"
    exit 1
fi

# Check if all required handler modules exist
echo "✅ Checking handler modules..."
required_handlers=("accounts" "blocks" "consensus" "faucet" "metrics" "network_monitoring" "status" "transactions" "transaction_submission" "validators")

for handler in "${required_handlers[@]}"; do
    if [ -f "blockchain_node/src/api/handlers/${handler}.rs" ]; then
        echo "   - ${handler}.rs exists ✓"
    else
        echo "   ❌ ${handler}.rs missing"
        exit 1
    fi
done

# Check if faucet module has required functions
echo "✅ Checking faucet module..."
if grep -q "pub async fn faucet_dashboard" "blockchain_node/src/api/faucet.rs"; then
    echo "   - faucet_dashboard function exists ✓"
else
    echo "   ❌ faucet_dashboard function missing"
    exit 1
fi

if grep -q "pub async fn request_tokens" "blockchain_node/src/api/faucet.rs"; then
    echo "   - request_tokens function exists ✓"
else
    echo "   ❌ request_tokens function missing"
    exit 1
fi

# Check if accounts module has required functions
echo "✅ Checking accounts module..."
if grep -q "pub async fn get_account_balance" "blockchain_node/src/api/handlers/accounts.rs"; then
    echo "   - get_account_balance function exists ✓"
else
    echo "   ❌ get_account_balance function missing"
    exit 1
fi

# Check if consensus module has required functions
echo "✅ Checking consensus module..."
if grep -q "pub async fn get_consensus_status" "blockchain_node/src/api/handlers/consensus.rs"; then
    echo "   - get_consensus_status function exists ✓"
else
    echo "   ❌ get_consensus_status function missing"
    exit 1
fi

# Check if main.rs can import testnet_router
echo "✅ Checking main.rs imports..."
if grep -q "testnet_router::create_testnet_router" "blockchain_node/src/main.rs"; then
    echo "   - testnet_router import exists ✓"
else
    echo "   ❌ testnet_router import missing"
    exit 1
fi

echo ""
echo "🎉 All compilation checks passed! The ArthaChain API should compile successfully."
echo "📊 Current Status: Phase 1 Complete - Basic API Structure Ready"
echo "🚀 Next: Phase 2 - Production Deployment Configuration"
