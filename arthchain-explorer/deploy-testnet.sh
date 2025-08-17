#!/bin/bash

# ArthChain Explorer Testnet Deployment Script
# This script builds and deploys the explorer to testnet domains

echo "🚀 Starting ArthChain Explorer Testnet Deployment..."

# Build the project for production
echo "📦 Building project for production..."
npm run build:production

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build completed successfully!"

# Create deployment package
echo "📋 Creating deployment package..."
cd dist
tar -czf ../arthachain-explorer-testnet.tar.gz *
cd ..

echo "📦 Deployment package created: arthachain-explorer-testnet.tar.gz"

# Instructions for manual deployment
echo ""
echo "🌐 Deployment Package Ready!"
echo "=================================================="
echo "Deploy this package to your web servers:"
echo "• testnet.arthachain.online"
echo "• testnet.arthachain.in"
echo ""
echo "Instructions:"
echo "1. Upload arthachain-explorer-testnet.tar.gz to your server"
echo "2. Extract: tar -xzf arthachain-explorer-testnet.tar.gz"
echo "3. Serve the files with your web server (nginx, apache, etc.)"
echo "4. Ensure proper CORS headers are configured"
echo ""
echo "Nginx configuration example:"
echo "server {"
echo "  listen 80;"
echo "  server_name testnet.arthachain.online testnet.arthachain.in;"
echo "  root /var/www/arthachain-explorer;"
echo "  index index.html;"
echo ""
echo "  location / {"
echo "    try_files \$uri \$uri/ /index.html;"
echo "  }"
echo ""
echo "  # Enable CORS for API calls"
echo "  add_header 'Access-Control-Allow-Origin' '*' always;"
echo "  add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;"
echo "  add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range' always;"
echo "}"
echo "=================================================="

echo "✅ Deployment script completed!"
