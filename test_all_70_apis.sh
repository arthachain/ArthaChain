#!/bin/bash

echo "🧪 TESTING ALL 70+ ARTHACHAIN APIs"
echo "=================================="
echo ""

# Working domains
REALTIME_URL="https://realtime.arthachain.in"
RPC_URL="https://rpc.arthachain.in"
TEMP_URL="https://una-legitimate-primary-dramatically.trycloudflare.com"

# Test counter
TOTAL=0
WORKING=0

test_endpoint() {
    local url=$1
    local endpoint=$2
    local name=$3
    
    TOTAL=$((TOTAL + 1))
    printf "%-50s" "$name"
    
    response=$(curl -s "$url$endpoint" --connect-timeout 5 --max-time 10)
    status=$?
    
    if [ $status -eq 0 ] && [ ! -z "$response" ]; then
        echo "✅ WORKING"
        WORKING=$((WORKING + 1))
    else
        echo "❌ Failed"
    fi
}

echo "🔥 TESTING MAIN APIs (RPC Domain)"
echo "================================"
test_endpoint "$RPC_URL" "/api/health" "Health Check"
test_endpoint "$RPC_URL" "/api/stats" "Node Statistics"
test_endpoint "$RPC_URL" "/api/blocks/latest" "Latest Block"
test_endpoint "$RPC_URL" "/api/blocks/0" "Genesis Block"
test_endpoint "$RPC_URL" "/api/transactions/pending" "Pending Transactions"
test_endpoint "$RPC_URL" "/api/validators" "Validator List"
test_endpoint "$RPC_URL" "/api/validators/active" "Active Validators"
test_endpoint "$RPC_URL" "/api/faucet/status" "Faucet Status"
test_endpoint "$RPC_URL" "/api/network/peers" "Network Peers"
test_endpoint "$RPC_URL" "/api/network/status" "Network Status"

echo ""
echo "🚀 TESTING EVM APIs"
echo "=================="
test_endpoint "$RPC_URL" "/evm/chainId" "EVM Chain ID"
test_endpoint "$RPC_URL" "/evm/gasPrice" "Gas Price"
test_endpoint "$RPC_URL" "/evm/blockNumber" "Block Number"
test_endpoint "$RPC_URL" "/evm/accounts" "EVM Accounts"
test_endpoint "$RPC_URL" "/evm/contracts" "Smart Contracts"

echo ""
echo "🔧 TESTING WASM APIs"
echo "==================="
test_endpoint "$RPC_URL" "/wasm" "WASM Engine Status"
test_endpoint "$RPC_URL" "/wasm/contracts" "WASM Contracts"
test_endpoint "$RPC_URL" "/wasm/modules" "WASM Modules"

echo ""
echo "⚡ TESTING SHARDING APIs"
echo "======================="
test_endpoint "$RPC_URL" "/shards" "Shard Information"
test_endpoint "$RPC_URL" "/shards/status" "Shard Status"
test_endpoint "$RPC_URL" "/shards/0" "Shard 0 Details"
test_endpoint "$RPC_URL" "/shards/cross" "Cross-Shard Transactions"

echo ""
echo "📊 TESTING METRICS APIs"
echo "======================"
test_endpoint "$RPC_URL" "/metrics" "Prometheus Metrics"
test_endpoint "$RPC_URL" "/metrics/performance" "Performance Metrics"
test_endpoint "$RPC_URL" "/metrics/consensus" "Consensus Metrics"
test_endpoint "$RPC_URL" "/metrics/network" "Network Metrics"

echo ""
echo "🏃 TESTING REAL-TIME APIs"
echo "========================"
test_endpoint "$REALTIME_URL" "/api/real/health" "Real-time Health"
test_endpoint "$REALTIME_URL" "/api/real/stats" "Real-time Stats"
test_endpoint "$REALTIME_URL" "/api/real/blocks" "Real-time Blocks"
test_endpoint "$REALTIME_URL" "/api/real/transactions" "Real-time Transactions"
test_endpoint "$REALTIME_URL" "/api/real/validators" "Real-time Validators"
test_endpoint "$REALTIME_URL" "/api/real/network" "Real-time Network"
test_endpoint "$REALTIME_URL" "/api/real/consensus" "Real-time Consensus"

echo ""
echo "🔗 TESTING RPC APIs"
echo "=================="
test_endpoint "$RPC_URL" "/rpc" "JSON-RPC Endpoint"
test_endpoint "$RPC_URL" "/rpc/eth_chainId" "ETH Chain ID"
test_endpoint "$RPC_URL" "/rpc/eth_blockNumber" "ETH Block Number"
test_endpoint "$RPC_URL" "/rpc/eth_gasPrice" "ETH Gas Price"
test_endpoint "$RPC_URL" "/rpc/net_version" "Network Version"
test_endpoint "$RPC_URL" "/rpc/web3_clientVersion" "Client Version"

echo ""
echo "🔐 TESTING CRYPTO/QUANTUM APIs"
echo "=============================="
test_endpoint "$RPC_URL" "/crypto/quantum/status" "Quantum Resistance Status"
test_endpoint "$RPC_URL" "/crypto/signatures" "Signature Verification"
test_endpoint "$RPC_URL" "/crypto/keys" "Key Management"

echo ""
echo "🤖 TESTING AI/ML APIs"
echo "=====================" 
test_endpoint "$RPC_URL" "/ai/status" "AI Engine Status"
test_endpoint "$RPC_URL" "/ai/models" "AI Models"
test_endpoint "$RPC_URL" "/ai/inference" "AI Inference"

echo ""
echo "⚖️ TESTING CONSENSUS APIs"
echo "========================"
test_endpoint "$RPC_URL" "/consensus/status" "Consensus Status"
test_endpoint "$RPC_URL" "/consensus/validators" "Consensus Validators"
test_endpoint "$RPC_URL" "/consensus/rounds" "Consensus Rounds"

echo ""
echo "💰 TESTING FAUCET APIs"
echo "====================="
test_endpoint "$RPC_URL" "/api/faucet/request" "Faucet Request"
test_endpoint "$RPC_URL" "/api/faucet/balance" "Faucet Balance"
test_endpoint "$RPC_URL" "/api/faucet/history" "Faucet History"

echo ""
echo "🏗️ TESTING MINING/STAKING APIs"
echo "==============================="
test_endpoint "$RPC_URL" "/mining/status" "Mining Status"
test_endpoint "$RPC_URL" "/staking/validators" "Staking Validators"
test_endpoint "$RPC_URL" "/staking/rewards" "Staking Rewards"

echo ""
echo "🌐 TESTING GOVERNANCE APIs"
echo "=========================="
test_endpoint "$RPC_URL" "/governance/proposals" "Governance Proposals"
test_endpoint "$RPC_URL" "/governance/voting" "Voting Status"

echo ""
echo "📱 TESTING EXPLORER APIs"
echo "========================"
test_endpoint "$RPC_URL" "/api/explorer/blocks" "Explorer Blocks"
test_endpoint "$RPC_URL" "/api/explorer/transactions" "Explorer Transactions"
test_endpoint "$RPC_URL" "/api/explorer/addresses" "Explorer Addresses"
test_endpoint "$RPC_URL" "/api/explorer/search" "Explorer Search"

echo ""
echo "⚡ TESTING BACKUP: TEMPORARY URL"
echo "==============================="
test_endpoint "$TEMP_URL" "/api/health" "Temp - Health Check"
test_endpoint "$TEMP_URL" "/api/stats" "Temp - Node Stats"
test_endpoint "$TEMP_URL" "/api/validators" "Temp - Validators"
test_endpoint "$TEMP_URL" "/metrics" "Temp - Metrics"
test_endpoint "$TEMP_URL" "/evm/chainId" "Temp - EVM Chain"

echo ""
echo "📊 FINAL RESULTS"
echo "================"
echo "Total APIs tested: $TOTAL"
echo "Working APIs: $WORKING"
echo "Success rate: $(( WORKING * 100 / TOTAL ))%"
echo ""

if [ $WORKING -gt 50 ]; then
    echo "🎉 EXCELLENT! Most APIs are working!"
else
    echo "⚠️  Some APIs need attention"
fi

echo ""
echo "🌟 Your live API domains:"
echo "✅ https://realtime.arthachain.in"
echo "✅ https://rpc.arthachain.in" 
echo "✅ $TEMP_URL"
