#!/bin/bash

# Artha Chain Simple Testnet Management Script

set -e

TESTNET_DIR=$(pwd)
DATA_DIR="$TESTNET_DIR/data"

function print_help {
    echo "Artha Chain Simple Testnet Management Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start        Start the simplified testnet"
    echo "  stop         Stop the testnet"
    echo "  status       Check the status of the testnet"
    echo "  logs         Show logs (use -f to follow)"
    echo "  help         Show this help message"
    echo ""
}

function start_testnet {
    echo "Starting Artha Chain simplified testnet..."
    
    # Make sure the API directories exist
    mkdir -p explorer/api
    
    # Create a status endpoint if it doesn't exist
    if [ ! -f "explorer/api/status" ]; then
        echo '{"status":"running","latest_block_height":3,"pending_tx_count":0,"active_validators":1}' > explorer/api/status
    fi
    
    # Start the containers
    docker-compose -f docker-compose-simple.yml up -d
    
    echo "Testnet is starting. Check status with: $0 status"
    echo "Explorer available at: http://localhost:8080"
    echo "API available at: http://localhost:3000/api/"
}

function stop_testnet {
    echo "Stopping Artha Chain testnet..."
    docker-compose -f docker-compose-simple.yml down
    echo "Testnet stopped."
}

function check_status {
    echo "Checking testnet status..."
    docker-compose -f docker-compose-simple.yml ps
    
    # Check if the API is responding
    echo ""
    echo "API status:"
    curl -s http://localhost:3000/api/status
}

function show_logs {
    if [[ "$1" == "-f" ]]; then
        docker-compose -f docker-compose-simple.yml logs -f
    else
        docker-compose -f docker-compose-simple.yml logs
    fi
}

# Parse command
case "$1" in
    start)
        start_testnet
        ;;
    stop)
        stop_testnet
        ;;
    status)
        check_status
        ;;
    logs)
        shift
        show_logs "$@"
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