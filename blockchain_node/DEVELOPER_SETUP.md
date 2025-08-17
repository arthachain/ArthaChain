# 🚀 ArthaChain Developer Setup

## Join the ArthaChain Network in 1 Command

### Quick Start
```bash
curl -sSL https://raw.githubusercontent.com/arthachain/ArthaChain/main/blockchain_node/universal_node_setup.sh | bash
```

### What This Does
- ✅ Installs all dependencies automatically
- ✅ Auto-detects free ports (no conflicts)
- ✅ Connects to the main ArthaChain network
- ✅ Starts mining and participating in consensus
- ✅ Provides API endpoints for your applications

### Your Node Will Have
- **P2P Port**: Auto-detected (usually 30301+)
- **API Port**: Auto-detected (usually 8081+)
- **Bootstrap**: Connects to main network automatically
- **Blockchain**: Syncs with the network

### API Endpoints
Once running, your node provides:
- `GET /api/health` - Node health check
- `GET /api/stats` - Blockchain statistics
- `POST /api/transactions` - Submit transactions
- `GET /api/blocks/latest` - Latest block info

### Run in Background
```bash
# Stop foreground process
Ctrl+C

# Start in background
nohup python3 arthachain_node.py > node.log 2>&1 &

# Monitor logs
tail -f node.log
```

### Network Status
Check network health: https://api.arthachain.in/api/stats

### Support
- GitHub: https://github.com/arthachain/ArthaChain
- Issues: Report problems via GitHub issues
- Community: Join our developer community

---
**That's it! One command to join the ArthaChain network.** 🎉
