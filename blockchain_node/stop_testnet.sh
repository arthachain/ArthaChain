#!/bin/bash

# ArthaChain Testnet Stop Script
set -e

echo "🛑 Stopping ArthaChain Testnet..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PID_FILE="./testnet.pid"
LOG_FILE="./testnet.log"

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

# Function to stop the testnet
stop_testnet() {
    if [ ! -f "$PID_FILE" ]; then
        print_warning "PID file not found. Testnet may not be running."
        return 1
    fi

    local PID=$(cat "$PID_FILE")
    
    if ! kill -0 "$PID" 2>/dev/null; then
        print_warning "Process with PID $PID is not running."
        rm -f "$PID_FILE"
        return 1
    fi

    print_status "Stopping testnet process (PID: $PID)..."
    
    # Send TERM signal first
    kill -TERM "$PID" 2>/dev/null || true
    
    # Wait for graceful shutdown
    local timeout=30
    local count=0
    
    while kill -0 "$PID" 2>/dev/null && [ $count -lt $timeout ]; do
        echo -n "."
        sleep 1
        count=$((count + 1))
    done
    
    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
        print_warning "Forcing termination..."
        kill -KILL "$PID" 2>/dev/null || true
        sleep 2
    fi
    
    # Verify process is stopped
    if kill -0 "$PID" 2>/dev/null; then
        print_error "Failed to stop process with PID $PID"
        return 1
    else
        print_success "Testnet stopped successfully"
        rm -f "$PID_FILE"
        return 0
    fi
}

# Function to cleanup resources
cleanup() {
    print_status "Cleaning up resources..."
    
    # Stop any remaining arthachain processes
    pkill -f "arthachain" 2>/dev/null || true
    
    # Clean up lock files
    rm -f "./testnet_data/*.lock" 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to show status
show_status() {
    echo ""
    echo "================================"
    echo "🔴 ArthaChain Testnet Stopped"
    echo "================================"
    echo ""
    echo "Resources:"
    echo "  Log file: $LOG_FILE (preserved)"
    echo "  Data directory: ./testnet_data (preserved)"
    echo ""
    echo "To restart: ./launch_testnet.sh"
    echo "To view logs: tail -f $LOG_FILE"
    echo "================================"
}

# Main execution
main() {
    echo "🔗 ArthaChain Testnet Stopper"
    echo "=============================="
    
    if stop_testnet; then
        cleanup
        show_status
    else
        print_error "Failed to stop testnet cleanly"
        cleanup
        exit 1
    fi
}

# Handle Ctrl+C
trap 'print_warning "Stop interrupted by user"; exit 1' INT

# Run main function
main "$@"