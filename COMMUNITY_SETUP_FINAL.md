# ArthaChain Community Node Setup

## Quick Start - One Command Installation

```bash
curl -O https://raw.githubusercontent.com/arthachain/ArthaChain/main/install-community-node.sh && chmod +x install-community-node.sh && ./install-community-node.sh
```

**That's it!** The installer will handle everything for you.

---

## System Requirements

### Minimum:
- **RAM:** 8 GB
- **Storage:** 100 GB SSD  
- **CPU:** 2 cores
- **OS:** Ubuntu 20.04+ / Debian 11+ / CentOS 8+ / macOS 12+

### Recommended:
- **RAM:** 16 GB
- **Storage:** 250 GB SSD
- **CPU:** 4 cores
- **Network:** 100 Mbps

---

## What the Installer Does

1. ✅ Checks your system requirements
2. ✅ Installs all dependencies
3. ✅ Sets up Rust environment
4. ✅ Downloads ArthaChain
5. ✅ Builds the node software
6. ✅ Configures your node
7. ✅ Starts the node automatically

---

## During Installation

The installer will ask you:

1. **Node Name:** Give your node a unique name (or press Enter for default)
2. **Dashboard Port:** Which port for the web dashboard (default: 8080)
3. **P2P Port:** Which port for peer connections (default: 30303)
4. **External IP:** Your public IP (auto-detected, just press Enter)

---

## After Installation

### Access Your Dashboard
Open your browser and go to:
```
http://localhost:8080
```

### Management Commands

All commands are in your home directory under `.arthachain/`:

**Start Node:**
```bash
~/.arthachain/start-node.sh
```

**Stop Node:**
```bash
~/.arthachain/stop-node.sh
```

**Check Status:**
```bash
~/.arthachain/node-status.sh
```

**Update Node:**
```bash
~/.arthachain/update-node.sh
```

**View Logs:**
```bash
tail -f ~/.arthachain/node.log
```

---

## Docker Alternative

If you prefer Docker:

```bash
docker run -d \
  --name arthachain-node \
  -p 8080:8080 \
  -p 30303:30303 \
  -v ~/arthachain:/data \
  --restart always \
  ghcr.io/arthachain/node:latest
```

---

## Firewall Configuration

Open these ports in your firewall:

```bash
# Ubuntu/Debian
sudo ufw allow 8080/tcp
sudo ufw allow 30303/tcp
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --add-port=8080/tcp --permanent
sudo firewall-cmd --add-port=30303/tcp --permanent
sudo firewall-cmd --reload
```

---

## Cloud Setup (AWS/GCP/Azure)

### AWS EC2:
1. Launch Ubuntu 20.04 instance (t3.large or better)
2. Open ports 8080 and 30303 in Security Group
3. SSH into instance and run the installer

### Quick AWS Setup:
```bash
# After SSH into EC2
curl -O https://raw.githubusercontent.com/arthachain/ArthaChain/main/install-community-node.sh && chmod +x install-community-node.sh && ./install-community-node.sh
```

---

## Systemd Service (Auto-start on boot)

```bash
# Create service file
sudo nano /etc/systemd/system/arthachain.service
```

Paste this:
```ini
[Unit]
Description=ArthaChain Node
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/home/$USER/.arthachain/start-node.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable it:
```bash
sudo systemctl enable arthachain
sudo systemctl start arthachain
```

---

## Troubleshooting

### Node won't start?
```bash
# Check if port is in use
lsof -i:8080

# Kill any existing processes
pkill -f testnet_api_server

# Restart
~/.arthachain/start-node.sh
```

### Can't connect to other nodes?
```bash
# Check firewall
sudo ufw status

# Check node status
~/.arthachain/node-status.sh
```

### Low disk space?
```bash
# Clean old data
rm -rf ~/.arthachain/data/rocksdb
~/.arthachain/start-node.sh
```

---

## Support

**Discord:** discord.gg/arthachain  
**Telegram:** t.me/arthachain  
**GitHub:** github.com/arthachain  

---

## Important Notes

✅ This connects to the REAL ArthaChain network  
✅ Your node helps decentralize the network  
✅ No duplicate nodes will be created  
✅ Automatic updates available  

---

**Thank you for supporting ArthaChain!** 🚀
