#!/bin/bash
# ArthaChain Real Blockchain Stop Script
# Gracefully stops all real blockchain services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_header() { echo -e "${PURPLE}[ARTHACHAIN]${NC} $1"; }
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

print_header "🛑 Stopping ArthaChain Real Blockchain Services"

# Stop API server
if [ -f api_server.pid ]; then
    API_PID=$(cat api_server.pid)
    if ps -p $API_PID > /dev/null; then
        print_status "🌐 Stopping API server (PID: $API_PID)..."
        kill -TERM $API_PID
        sleep 2
        if ps -p $API_PID > /dev/null; then
            kill -KILL $API_PID
        fi
        print_success "API server stopped"
    else
        print_warning "API server was not running"
    fi
    rm -f api_server.pid
else
    print_warning "No API server PID file found"
fi

# Stop dashboard server
if [ -f dashboard.pid ]; then
    DASHBOARD_PID=$(cat dashboard.pid)
    if ps -p $DASHBOARD_PID > /dev/null; then
        print_status "📊 Stopping dashboard server (PID: $DASHBOARD_PID)..."
        kill -TERM $DASHBOARD_PID
        sleep 2
        if ps -p $DASHBOARD_PID > /dev/null; then
            kill -KILL $DASHBOARD_PID
        fi
        print_success "Dashboard server stopped"
    else
        print_warning "Dashboard server was not running"
    fi
    rm -f dashboard.pid
else
    print_warning "No dashboard server PID file found"
fi

# Stop blockchain node
if [ -f blockchain.pid ]; then
    BLOCKCHAIN_PID=$(cat blockchain.pid)
    if ps -p $BLOCKCHAIN_PID > /dev/null; then
        print_status "⛏️  Stopping blockchain node (PID: $BLOCKCHAIN_PID)..."
        kill -TERM $BLOCKCHAIN_PID
        sleep 5
        if ps -p $BLOCKCHAIN_PID > /dev/null; then
            kill -KILL $BLOCKCHAIN_PID
        fi
        print_success "Blockchain node stopped"
    else
        print_warning "Blockchain node was not running"
    fi
    rm -f blockchain.pid
else
    print_warning "No blockchain node PID file found"
fi

# Kill any remaining processes
print_status "🧹 Cleaning up any remaining processes..."
pkill -f "arthachain" 2>/dev/null || true
pkill -f "testnet_api_server" 2>/dev/null || true

print_success "✅ All ArthaChain services stopped successfully"
print_status "📊 Log files preserved in logs/ directory"
