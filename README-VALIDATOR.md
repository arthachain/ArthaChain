# 🚀 ArthChain Validator Setup - Simple Guide

## Quick Setup (One Command)

### Using cURL
```bash
curl -O https://raw.githubusercontent.com/arthachain/ArthaChain/main/install-validator.sh && chmod +x install-validator.sh && ./install-validator.sh
```

The terminal will ask setup questions:

**Join ArthChain Network:**
```
By running this installer, you agree to join the ArthChain network as a validator. (y/n)?:
```

**Enable Dashboard:**
```
Do you want to run the web based Dashboard? (y/n):
```

**Dashboard Port:**
```
Enter the port (1025-65536) to access the web based Dashboard (default 8080):
```

**External IP:**
```
If you wish to set an explicit external IP, enter an IPv4 address (default=auto):
```

**P2P Port:**
```
Enter the first port (1025-65536) for p2p communication (default 30303):
```

**API Port:**
```
Enter the API port (1025-65536) for blockchain API (default 8080):
```

**Installation Path:**
```
What base directory should the node use (defaults to ~/.arthachain):
```

**Dashboard Password:**
```
Set the password to access the Dashboard:
```

---

## Using Docker (Alternative)

```bash
# Pull ArthChain validator image
docker pull ghcr.io/arthachain/validator:latest

# Run validator
docker run \
    --name arthachain-validator \
    -p 8080:8080 \
    -p 30303:30303 \
    -v $(pwd)/arthachain:/home/node/config \
    --restart=always \
    --detach \
    ghcr.io/arthachain/validator:latest

# Set dashboard password
docker exec -it arthachain-validator operator-cli gui set password "YOUR_PASSWORD"
```

### Custom Ports Example:
```bash
docker run \
    --name arthachain-validator \
    -p 10080:10080 \
    -p 31303:31303 \
    -e P2P_PORT=31303 \
    -e API_PORT=8080 \
    -e DASHBOARD_PORT=10080 \
    -v $(pwd)/arthachain:/home/node/config \
    --restart=always \
    --detach \
    ghcr.io/arthachain/validator:latest
```

---

## After Installation

### Start Validator:
```bash
./start-validator.sh
```

### Check Status:
```bash
./check-status.sh
```

### Stop Validator:
```bash
./stop-validator.sh
```

### Access Dashboard:
```
http://localhost:8080
```

---

## Network Info

- **Network**: ArthChain Testnet
- **Chain ID**: 201766
- **Bootstrap**: 103.160.27.61:30303
- **Block Time**: ~5 seconds
- **Consensus**: SVCP + Quantum SVBFT

---

## System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 50GB
- Network: Stable internet

**Recommended:**
- CPU: 4+ cores  
- RAM: 8GB+
- Storage: 100GB+ SSD
- Network: High-speed connection

---

## Support

- **Status**: https://api.arthachain.in/api/status
- **GitHub**: https://github.com/arthachain/ArthaChain  
- **Issues**: Report via GitHub Issues

**🎯 Simple, clear, just like Shardeum!**
