#!/bin/bash

# ArthChain Explorer Local Server Starter
echo "🚀 Starting ArthChain Explorer Locally..."

# Kill any existing servers
echo "🧹 Cleaning up existing servers..."
pkill -f "python3.*http.server" 2>/dev/null || true
pkill -f "serve" 2>/dev/null || true

# Navigate to dist directory
cd dist

# Start Python HTTP server
echo "🌐 Starting local server on port 8000..."
echo "📱 ArthChain Explorer will be available at:"
echo "   👉 http://localhost:8000"
echo ""
echo "🔗 The explorer is configured to connect to:"
echo "   📡 API: https://api.arthachain.in"
echo "   🔗 RPC: https://rpc.arthachain.in"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================="

python3 -m http.server 8000
