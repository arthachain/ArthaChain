#!/bin/bash

# ArthaChain Multi-Node Validation Script
set -e

echo "🧪 Validating ArthaChain Multi-Node Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
NUM_NODES=4
BASE_API_PORT=3000
BASE_RPC_PORT=8540
TEST_RESULTS_FILE="./test_results.json"

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_test() {
    echo -e "${PURPLE}[TEST]${NC} $1"
}

print_result() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✓ PASS${NC} $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAIL${NC} $test_name"
        if [ -n "$details" ]; then
            echo -e "  ${YELLOW}Details:${NC} $details"
        fi
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# Function to run a test with timeout
run_test() {
    local test_name="$1"
    local test_command="$2"
    local timeout_seconds="${3:-10}"
    
    print_test "Running: $test_name"
    
    if timeout "$timeout_seconds" bash -c "$test_command" >/dev/null 2>&1; then
        print_result "$test_name" "PASS"
        return 0
    else
        print_result "$test_name" "FAIL" "Command failed or timed out"
        return 1
    fi
}

# Function to test API endpoint
test_api_endpoint() {
    local node_id="$1"
    local endpoint="$2"
    local expected_status="${3:-200}"
    local api_port=$((BASE_API_PORT + node_id))
    
    local response=$(curl -s -w "%{http_code}" -o /dev/null "http://127.0.0.1:$api_port$endpoint" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        return 0
    else
        return 1
    fi
}

# Function to test RPC endpoint
test_rpc_endpoint() {
    local node_id="$1"
    local method="$2"
    local rpc_port=$((BASE_RPC_PORT + node_id))
    
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"method\":\"$method\",\"params\":[],\"id\":1}" \
        "http://127.0.0.1:$rpc_port" 2>/dev/null || echo "")
    
    if echo "$response" | grep -q '"result"'; then
        return 0
    else
        return 1
    fi
}

# Function to check node health
test_node_health() {
    print_status "Testing node health endpoints..."
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        if test_api_endpoint $i "/health"; then
            print_result "Node $i Health Check" "PASS"
        else
            print_result "Node $i Health Check" "FAIL" "Health endpoint not responding"
        fi
    done
}

# Function to test node info endpoints
test_node_info() {
    print_status "Testing node info endpoints..."
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        if test_api_endpoint $i "/node/info"; then
            print_result "Node $i Info Endpoint" "PASS"
        else
            print_result "Node $i Info Endpoint" "FAIL" "Info endpoint not responding"
        fi
    done
}

# Function to test RPC endpoints
test_rpc_endpoints() {
    print_status "Testing RPC endpoints..."
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        if test_rpc_endpoint $i "web3_clientVersion"; then
            print_result "Node $i RPC Client Version" "PASS"
        else
            print_result "Node $i RPC Client Version" "FAIL" "RPC method not responding"
        fi
        
        if test_rpc_endpoint $i "eth_blockNumber"; then
            print_result "Node $i RPC Block Number" "PASS"
        else
            print_result "Node $i RPC Block Number" "FAIL" "Block number method not responding"
        fi
    done
}

# Function to test block height consistency
test_block_height_consistency() {
    print_status "Testing block height consistency..."
    
    local heights=()
    local all_responding=true
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        local api_port=$((BASE_API_PORT + i))
        local height=$(curl -s "http://127.0.0.1:$api_port/blocks/latest" 2>/dev/null | grep -o '"height":[0-9]*' | cut -d':' -f2 2>/dev/null || echo "")
        
        if [ -n "$height" ]; then
            heights[$i]=$height
        else
            heights[$i]="N/A"
            all_responding=false
        fi
    done
    
    if [ "$all_responding" = false ]; then
        print_result "Block Height Retrieval" "FAIL" "Some nodes not responding"
        return
    fi
    
    # Check if all heights are within 1 block of each other
    local min_height=${heights[0]}
    local max_height=${heights[0]}
    
    for height in "${heights[@]}"; do
        if [ "$height" != "N/A" ]; then
            if [ "$height" -lt "$min_height" ]; then
                min_height=$height
            fi
            if [ "$height" -gt "$max_height" ]; then
                max_height=$height
            fi
        fi
    done
    
    local height_diff=$((max_height - min_height))
    
    if [ $height_diff -le 1 ]; then
        print_result "Block Height Consistency" "PASS" "Heights: ${heights[*]}"
    else
        print_result "Block Height Consistency" "FAIL" "Height difference: $height_diff, Heights: ${heights[*]}"
    fi
}

# Function to test transaction submission
test_transaction_submission() {
    print_status "Testing transaction submission..."
    
    local api_port=$BASE_API_PORT
    local tx_data='{
        "from": "0x742d35Cc6634C0532925a3b8D6Dd6782b4c99999",
        "to": "0x742d35Cc6634C0532925a3b8D6Dd6782b4c88888",
        "amount": 100,
        "fee": 10
    }'
    
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$tx_data" \
        "http://127.0.0.1:$api_port/transactions/send" 2>/dev/null || echo "")
    
    if [ -n "$response" ] && ! echo "$response" | grep -q "error"; then
        print_result "Transaction Submission" "PASS"
        
        # Wait for transaction propagation
        sleep 3
        
        # Test transaction propagation by checking if other nodes see the transaction
        test_transaction_propagation
    else
        print_result "Transaction Submission" "FAIL" "Transaction submission failed"
    fi
}

# Function to test transaction propagation
test_transaction_propagation() {
    print_status "Testing transaction propagation..."
    
    local propagated_count=0
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        local api_port=$((BASE_API_PORT + i))
        local mempool_response=$(curl -s "http://127.0.0.1:$api_port/transactions/mempool" 2>/dev/null || echo "")
        
        if [ -n "$mempool_response" ] && ! echo "$mempool_response" | grep -q "error"; then
            propagated_count=$((propagated_count + 1))
        fi
    done
    
    if [ $propagated_count -ge $((NUM_NODES / 2)) ]; then
        print_result "Transaction Propagation" "PASS" "Propagated to $propagated_count/$NUM_NODES nodes"
    else
        print_result "Transaction Propagation" "FAIL" "Only propagated to $propagated_count/$NUM_NODES nodes"
    fi
}

# Function to test consensus participation
test_consensus_participation() {
    print_status "Testing consensus participation..."
    
    local participating_nodes=0
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        local api_port=$((BASE_API_PORT + i))
        local consensus_info=$(curl -s "http://127.0.0.1:$api_port/consensus/status" 2>/dev/null || echo "")
        
        if [ -n "$consensus_info" ] && ! echo "$consensus_info" | grep -q "error"; then
            participating_nodes=$((participating_nodes + 1))
        fi
    done
    
    if [ $participating_nodes -ge $((NUM_NODES * 2 / 3)) ]; then
        print_result "Consensus Participation" "PASS" "$participating_nodes/$NUM_NODES nodes participating"
    else
        print_result "Consensus Participation" "FAIL" "Only $participating_nodes/$NUM_NODES nodes participating"
    fi
}

# Function to test peer connectivity
test_peer_connectivity() {
    print_status "Testing peer connectivity..."
    
    local connected_nodes=0
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        local api_port=$((BASE_API_PORT + i))
        local peers_response=$(curl -s "http://127.0.0.1:$api_port/network/peers" 2>/dev/null || echo "")
        
        # Check if node has at least 1 peer
        if [ -n "$peers_response" ] && ! echo "$peers_response" | grep -q "error"; then
            local peer_count=$(echo "$peers_response" | grep -o '"peer_count":[0-9]*' | cut -d':' -f2 2>/dev/null || echo "0")
            if [ "$peer_count" -gt 0 ]; then
                connected_nodes=$((connected_nodes + 1))
            fi
        fi
    done
    
    if [ $connected_nodes -ge $((NUM_NODES / 2)) ]; then
        print_result "Peer Connectivity" "PASS" "$connected_nodes/$NUM_NODES nodes have peers"
    else
        print_result "Peer Connectivity" "FAIL" "Only $connected_nodes/$NUM_NODES nodes have peers"
    fi
}

# Function to test AI engine functionality
test_ai_engine() {
    print_status "Testing AI engine functionality..."
    
    local ai_enabled_nodes=0
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        local api_port=$((BASE_API_PORT + i))
        local ai_status=$(curl -s "http://127.0.0.1:$api_port/ai/status" 2>/dev/null || echo "")
        
        if [ -n "$ai_status" ] && ! echo "$ai_status" | grep -q "error"; then
            ai_enabled_nodes=$((ai_enabled_nodes + 1))
        fi
    done
    
    if [ $ai_enabled_nodes -gt 0 ]; then
        print_result "AI Engine Status" "PASS" "$ai_enabled_nodes/$NUM_NODES nodes have AI enabled"
    else
        print_result "AI Engine Status" "FAIL" "No nodes have AI engine responding"
    fi
}

# Function to test metrics endpoints
test_metrics() {
    print_status "Testing metrics endpoints..."
    
    local metrics_responding=0
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        local prometheus_port=$((9090 + i))
        if curl -s "http://127.0.0.1:$prometheus_port/metrics" > /dev/null 2>&1; then
            metrics_responding=$((metrics_responding + 1))
        fi
    done
    
    if [ $metrics_responding -ge $((NUM_NODES / 2)) ]; then
        print_result "Metrics Endpoints" "PASS" "$metrics_responding/$NUM_NODES nodes serving metrics"
    else
        print_result "Metrics Endpoints" "FAIL" "Only $metrics_responding/$NUM_NODES nodes serving metrics"
    fi
}

# Function to generate test report
generate_test_report() {
    print_status "Generating test report..."
    
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local pass_rate=$(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
    
    cat > "$TEST_RESULTS_FILE" << EOF
{
    "timestamp": "$timestamp",
    "total_tests": $TOTAL_TESTS,
    "passed_tests": $PASSED_TESTS,
    "failed_tests": $FAILED_TESTS,
    "pass_rate": "$pass_rate%",
    "test_environment": {
        "num_nodes": $NUM_NODES,
        "base_api_port": $BASE_API_PORT,
        "base_rpc_port": $BASE_RPC_PORT
    },
    "status": "$([ $FAILED_TESTS -eq 0 ] && echo "PASS" || echo "FAIL")"
}
EOF
    
    print_success "Test report saved to $TEST_RESULTS_FILE"
}

# Function to display final results
show_final_results() {
    echo ""
    echo "==============================================="
    echo "🧪 Multi-Node Validation Results"
    echo "==============================================="
    echo ""
    echo "Test Summary:"
    echo "  Total Tests: $TOTAL_TESTS"
    echo "  Passed: $PASSED_TESTS"
    echo "  Failed: $FAILED_TESTS"
    
    if [ $TOTAL_TESTS -gt 0 ]; then
        local pass_rate=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
        echo "  Pass Rate: $pass_rate%"
    fi
    
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! Multi-node environment is working correctly.${NC}"
    else
        echo -e "${RED}❌ $FAILED_TESTS test(s) failed. Please check the logs and fix issues.${NC}"
    fi
    
    echo ""
    echo "Report: $TEST_RESULTS_FILE"
    echo "==============================================="
}

# Main test execution
main() {
    echo "🔗 ArthaChain Multi-Node Validator"
    echo "==================================="
    
    # Check if nodes are running
    print_status "Checking if multi-node environment is running..."
    local running_count=0
    
    for i in $(seq 0 $((NUM_NODES-1))); do
        local api_port=$((BASE_API_PORT + i))
        if curl -s "http://127.0.0.1:$api_port/health" > /dev/null 2>&1; then
            running_count=$((running_count + 1))
        fi
    done
    
    if [ $running_count -eq 0 ]; then
        print_error "No nodes appear to be running. Start the multi-node environment first:"
        echo "  ./multi_node_test.sh"
        exit 1
    elif [ $running_count -lt $NUM_NODES ]; then
        print_warning "Only $running_count/$NUM_NODES nodes are responding. Some tests may fail."
    else
        print_success "All $NUM_NODES nodes are responding"
    fi
    
    echo ""
    print_status "Starting validation tests..."
    echo ""
    
    # Run all test suites
    test_node_health
    test_node_info
    test_rpc_endpoints
    test_block_height_consistency
    test_peer_connectivity
    test_consensus_participation
    test_transaction_submission
    test_ai_engine
    test_metrics
    
    # Generate report and show results
    generate_test_report
    show_final_results
    
    # Exit with appropriate code
    if [ $FAILED_TESTS -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Handle Ctrl+C
trap 'print_warning "Validation interrupted by user"; exit 1' INT

# Run main function
main "$@"
