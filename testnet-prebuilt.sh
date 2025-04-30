#!/bin/bash

# Artha Chain Pre-built Testnet Management Script

set -e

TESTNET_DIR=$(pwd)
DATA_DIR="$TESTNET_DIR/data"

function print_help {
    echo "Artha Chain Pre-built Testnet Management Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build        Build the Artha Chain binary on the host"
    echo "  start        Start the testnet (after building)"
    echo "  stop         Stop the testnet"
    echo "  restart      Restart the testnet"
    echo "  status       Check the status of the testnet"
    echo "  logs         Show logs (use -f to follow)"
    echo "  clean        Clean all data (WARNING: This will reset the blockchain state)"
    echo "  generate-tx  Generate sample transactions for testing"
    echo "  help         Show this help message"
    echo ""
}

function build_binary {
    echo "Building mock Artha Chain binary for testing..."
    
    # Create the mock binary
    ./mock-binary.sh
    
    # Check if the binary exists
    if [ ! -f "target/release/arthachain" ]; then
        echo "Error: Binary was not built successfully."
        exit 1
    fi
    
    echo "Mock binary built successfully at target/release/arthachain"
}

function start_testnet {
    echo "Starting Artha Chain testnet with pre-built binary..."
    
    # Check if binary exists
    if [ ! -f "target/release/arthachain" ]; then
        echo "Error: Binary not found. Please run '$0 build' first."
        exit 1
    fi
    
    # Create data directories
    mkdir -p "$DATA_DIR/validator1"
    
    # Start the containers
    docker-compose -f docker-compose-simple.yml up -d
    
    echo "Testnet is starting. Check status with: $0 status"
}

function stop_testnet {
    echo "Stopping Artha Chain testnet..."
    docker-compose -f docker-compose-simple.yml down
    echo "Testnet stopped."
}

function restart_testnet {
    stop_testnet
    sleep 2
    start_testnet
}

function check_status {
    echo "Checking testnet status..."
    docker-compose -f docker-compose-simple.yml ps
    
    # Check if the API is responding
    echo ""
    echo "API status:"
    curl -s http://localhost:3000/api/status 2>/dev/null || echo "API not responding yet (this is normal during startup)"
}

function show_logs {
    if [[ "$1" == "-f" ]]; then
        docker-compose -f docker-compose-simple.yml logs -f
    else
        docker-compose -f docker-compose-simple.yml logs
    fi
}

function clean_data {
    echo "WARNING: This will delete all blockchain data and reset the testnet."
    echo "Are you sure you want to continue? (y/n)"
    read -r confirm
    if [[ "$confirm" != "y" ]]; then
        echo "Operation cancelled."
        return
    fi
    
    stop_testnet
    echo "Removing data directories..."
    rm -rf "$DATA_DIR"
    echo "Data cleaned. You can start a fresh testnet with: $0 start"
}

function generate_tx {
    echo "Generating sample transactions for the testnet..."
    
    # Check if testnet is running
    if ! docker ps | grep -q validator1; then
        echo "Testnet is not running. Please start it first with: $0 start"
        return 1
    fi
    
    # Generate 5 sample transactions
    for i in {1..5}; do
        echo "Generating transaction $i..."
        curl -X POST http://localhost:3000/api/transactions \
            -H "Content-Type: application/json" \
            -d "{\"sender\":\"validator1\",\"recipient\":\"user$i\",\"amount\":100,\"data\":\"Sample transaction $i\"}" \
            2>/dev/null || echo "API not responding"
        sleep 1
    done
    
    echo "Sample transactions generated."
}

# Parse command
case "$1" in
    build)
        build_binary
        ;;
    start)
        start_testnet
        ;;
    stop)
        stop_testnet
        ;;
    restart)
        restart_testnet
        ;;
    status)
        check_status
        ;;
    logs)
        shift
        show_logs "$@"
        ;;
    clean)
        clean_data
        ;;
    generate-tx)
        generate_tx
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo "Unknown command: $1"
        print_help
        exit 1
        ;;
esac

exit 0 