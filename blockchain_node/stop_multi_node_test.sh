#!/bin/bash

# ArthaChain Multi-Node Test Stop Script
set -e

echo "🛑 Stopping ArthaChain Multi-Node Test Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
TEST_DIR="./multi_node_test"
LOG_DIR="$TEST_DIR/logs"

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

print_node() {
    echo -e "${CYAN}[NODE $1]${NC} $2"
}

# Function to stop individual node
stop_node() {
    local node_id=$1
    local pid_file="$LOG_DIR/node-$node_id.pid"
    
    if [ ! -f "$pid_file" ]; then
        print_warning "PID file for node $node_id not found"
        return 1
    fi
    
    local pid=$(cat "$pid_file")
    
    if ! kill -0 "$pid" 2>/dev/null; then
        print_warning "Node $node_id (PID $pid) is not running"
        rm -f "$pid_file"
        return 1
    fi
    
    print_node $node_id "Stopping (PID: $pid)..."
    
    # Send TERM signal first
    kill -TERM "$pid" 2>/dev/null || true
    
    # Wait for graceful shutdown
    local timeout=15
    local count=0
    
    while kill -0 "$pid" 2>/dev/null && [ $count -lt $timeout ]; do
        sleep 1
        count=$((count + 1))
    done
    
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        print_node $node_id "Force killing..."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi
    
    # Verify process is stopped
    if kill -0 "$pid" 2>/dev/null; then
        print_error "Failed to stop node $node_id (PID $pid)"
        return 1
    else
        print_node $node_id "Stopped successfully"
        rm -f "$pid_file"
        return 0
    fi
}

# Function to stop all nodes
stop_all_nodes() {
    print_status "Stopping all test nodes..."
    
    if [ ! -d "$LOG_DIR" ]; then
        print_warning "Log directory not found. No nodes to stop."
        return 0
    fi
    
    local stopped_count=0
    local total_nodes=0
    
    # Count and stop nodes based on PID files
    for pid_file in "$LOG_DIR"/node-*.pid; do
        if [ -f "$pid_file" ]; then
            local node_id=$(basename "$pid_file" .pid | sed 's/node-//')
            total_nodes=$((total_nodes + 1))
            
            if stop_node "$node_id"; then
                stopped_count=$((stopped_count + 1))
            fi
        fi
    done
    
    if [ $total_nodes -eq 0 ]; then
        print_warning "No node PID files found"
    else
        print_success "Stopped $stopped_count/$total_nodes nodes"
    fi
}

# Function to cleanup remaining processes
cleanup_processes() {
    print_status "Cleaning up remaining processes..."
    
    # Kill any remaining arthachain processes
    local remaining=$(pkill -f "arthachain.*node-" 2>/dev/null && echo "found" || echo "none")
    
    if [ "$remaining" = "found" ]; then
        print_warning "Killed remaining arthachain processes"
        sleep 2
    fi
    
    # Clean up any lock files
    if [ -d "$TEST_DIR" ]; then
        find "$TEST_DIR" -name "*.lock" -delete 2>/dev/null || true
    fi
    
    print_success "Process cleanup completed"
}

# Function to show cleanup summary
show_summary() {
    echo ""
    echo "================================"
    echo "🔴 Multi-Node Test Environment Stopped"
    echo "================================"
    echo ""
    echo "Resources:"
    if [ -d "$LOG_DIR" ]; then
        echo "  Log files: $LOG_DIR/ (preserved)"
        echo "  View logs: tail -f $LOG_DIR/node-*.log"
    fi
    if [ -d "$TEST_DIR" ]; then
        echo "  Data directory: $TEST_DIR/ (preserved)"
    fi
    echo ""
    echo "To restart: ./multi_node_test.sh"
    echo "To clean all data: rm -rf $TEST_DIR"
    echo "================================"
}

# Function to check for running nodes
check_running_nodes() {
    print_status "Checking for running nodes..."
    
    local running_count=0
    
    # Check API endpoints
    for port in $(seq 3000 3010); do
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            print_warning "Node still responding on port $port"
            running_count=$((running_count + 1))
        fi
    done
    
    # Check for arthachain processes
    local process_count=$(pgrep -f "arthachain" | wc -l)
    
    if [ $process_count -gt 0 ]; then
        print_warning "$process_count arthachain processes still running"
        running_count=$((running_count + process_count))
    fi
    
    if [ $running_count -gt 0 ]; then
        print_error "$running_count nodes/processes may still be running"
        return 1
    else
        print_success "All nodes appear to be stopped"
        return 0
    fi
}

# Function to force cleanup
force_cleanup() {
    print_warning "Performing force cleanup..."
    
    # Kill all arthachain processes
    pkill -9 -f "arthachain" 2>/dev/null || true
    
    # Remove all PID files
    if [ -d "$LOG_DIR" ]; then
        rm -f "$LOG_DIR"/*.pid 2>/dev/null || true
    fi
    
    sleep 2
    print_success "Force cleanup completed"
}

# Main execution
main() {
    local force_flag=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force|-f)
                force_flag="true"
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--force|-f] [--help|-h]"
                echo ""
                echo "Options:"
                echo "  --force, -f    Force stop all processes"
                echo "  --help, -h     Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    echo "🔗 ArthaChain Multi-Node Test Stopper"
    echo "======================================"
    
    if [ "$force_flag" = "true" ]; then
        force_cleanup
    else
        stop_all_nodes
        cleanup_processes
        
        # Verify all nodes are stopped
        if ! check_running_nodes; then
            print_warning "Some nodes may still be running. Use --force to force stop all."
        fi
    fi
    
    show_summary
}

# Handle Ctrl+C
trap 'print_warning "Stop interrupted by user"; exit 1' INT

# Run main function
main "$@"
