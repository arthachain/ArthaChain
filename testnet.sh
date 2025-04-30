#!/bin/bash

# Artha Chain Testnet Management Script

set -e

TESTNET_DIR=$(pwd)
DATA_DIR="$TESTNET_DIR/data"

function print_help {
    echo "Artha Chain Testnet Management Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start        Start the testnet"
    echo "  stop         Stop the testnet"
    echo "  restart      Restart the testnet"
    echo "  status       Check the status of the testnet"
    echo "  logs         Show logs (use -f to follow, --node=<nodename> for specific node)"
    echo "  clean        Clean all data (WARNING: This will reset the blockchain state)"
    echo "  generate-tx  Generate sample transactions for testing"
    echo "  help         Show this help message"
    echo ""
}

function start_testnet {
    echo "Starting Artha Chain testnet..."
    
    # Create data directories
    mkdir -p "$DATA_DIR/validator1" "$DATA_DIR/validator2" "$DATA_DIR/validator3" "$DATA_DIR/validator4"
    
    # Start the containers
    docker-compose up -d
    
    echo "Testnet is starting. Check status with: $0 status"
}

function stop_testnet {
    echo "Stopping Artha Chain testnet..."
    docker-compose down
    echo "Testnet stopped."
}

function restart_testnet {
    stop_testnet
    sleep 2
    start_testnet
}

function check_status {
    echo "Checking testnet status..."
    docker-compose ps
    
    # Get the number of blocks from validator1
    echo ""
    echo "Latest block height:"
    curl -s http://localhost:3000/api/status 2>/dev/null | grep latest_block_height || echo "API not responding"
}

function show_logs {
    if [[ "$1" == "-f" ]]; then
        if [[ "$2" == "--node="* ]]; then
            NODE=$(echo "$2" | cut -d= -f2)
            docker-compose logs -f "$NODE"
        else
            docker-compose logs -f
        fi
    elif [[ "$1" == "--node="* ]]; then
        NODE=$(echo "$1" | cut -d= -f2)
        docker-compose logs "$NODE"
    else
        docker-compose logs
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
            2>/dev/null | grep -E "hash|error" || echo "API not responding"
        sleep 1
    done
    
    echo "Sample transactions generated."
}

# Parse command
case "$1" in
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