#!/bin/bash

echo "🚀 ArthaChain 10-Node Deployment Script"
echo "🔗 Bootstrap Node: 223.228.101.153:30303"
echo ""

# Node configuration template
create_node_script() {
    local node_id=$1
    cat > "node_${node_id}_setup.sh" << NODE_EOF
#!/bin/bash
echo "🚀 Setting up ArthaChain Node ${node_id}"

# Install dependencies
apt update && apt install -y python3 curl git

# Download P2P node script
curl -sSL https://raw.githubusercontent.com/arthachain/ArthaChain/main/blockchain_node/simple_p2p_server.py > arthachain_node.py
chmod +x arthachain_node.py

# Modify for this node
sed -i 's/port=30303/port=3030${node_id}/' arthachain_node.py
sed -i 's/api_port=8080/api_port=808${node_id}/' arthachain_node.py

# Start node
echo "🔗 Starting ArthaChain Node ${node_id}"
echo "📡 P2P Port: 3030${node_id}"
echo "🌐 API Port: 808${node_id}"
echo "🎯 Bootstrap: 223.228.101.153:30303"

python3 arthachain_node.py
NODE_EOF
    chmod +x "node_${node_id}_setup.sh"
}

# Create scripts for 10 nodes
for i in {1..10}; do
    create_node_script $i
    echo "✅ Created setup script for Node $i"
done

echo ""
echo "🎯 DEPLOYMENT INSTRUCTIONS:"
echo ""
echo "For each VM/server, run:"
echo "  curl -sSL https://raw.githubusercontent.com/arthachain/ArthaChain/main/blockchain_node/node_X_setup.sh | bash"
echo ""
echo "Replace X with node number (1-10)"
echo ""
echo "📊 Node Ports:"
for i in {1..10}; do
    echo "  Node $i: P2P=3030$i, API=808$i"
done
echo ""
echo "🔗 All nodes will connect to bootstrap: 223.228.101.153:30303"
