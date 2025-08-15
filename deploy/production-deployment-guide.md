# 🚀 ArthaChain Production Deployment Guide

This guide will help you deploy ArthaChain to production with proper security, monitoring, and scaling capabilities.

## 📋 Prerequisites

### Required Tools
```bash
# Docker & Docker Compose
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Git (if deploying from source)
sudo apt update && sudo apt install -y git
```

### Server Requirements
- **CPU**: 8+ cores (recommended)
- **RAM**: 16GB+ (minimum 8GB)
- **Storage**: 500GB+ SSD
- **Bandwidth**: Unlimited or high allowance
- **OS**: Ubuntu 22.04 LTS (recommended)

## 🏗️ Deployment Methods

### Method 1: Docker Compose (Recommended)

#### 1. Prepare Environment
```bash
# Create deployment directory
mkdir -p /opt/arthachain && cd /opt/arthachain

# Download deployment files (replace with your actual repository)
# For now, copy your files manually or use rsync/scp
```

#### 2. Configure Environment Variables
```bash
# Copy and edit environment template
cp deploy/.env.template deploy/.env
nano deploy/.env

# Required variables to update:
# - PUBLIC_IP: Your server's public IP
# - DOMAIN: Your domain name
# - CADDY_EMAIL: Your email for SSL certificates
# - API_ADMIN_KEY: Generate strong random key
# - GRAFANA_PASSWORD: Strong password for Grafana
```

#### 3. Update Caddyfile for Your Domain
```bash
nano deploy/Caddyfile
# Update with your actual domain names
```

#### 4. Deploy
```bash
cd deploy
docker-compose up -d
```

#### 5. Verify Deployment
```bash
# Check all services are running
docker-compose ps

# Test blockchain node
curl http://localhost:3000/api/health

# Check Grafana dashboard
open http://your-domain:3001
```

### Method 2: Cloud Provider Auto-Deploy

#### Digital Ocean
```bash
# Use the updated deployment script
chmod +x deploy/deploy-digitalocean.sh
./deploy/deploy-digitalocean.sh
```

#### AWS/GCP/Azure
See the respective cloud deployment guides in the `/docs` directory.

## 🔒 Security Configuration

### 1. Firewall Setup
```bash
# UFW firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 30303/tcp   # P2P
sudo ufw --force enable
```

### 2. SSL/TLS Configuration
The deployment automatically configures SSL certificates via Let's Encrypt through Caddy.

### 3. Security Hardening
```bash
# Disable root login
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# Install fail2ban
sudo apt install -y fail2ban
sudo systemctl enable fail2ban

# Configure automatic security updates
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades
```

## 📊 Monitoring Setup

### Access Monitoring Dashboards

1. **Grafana Dashboard**: `https://your-domain:3001`
   - Username: `admin`
   - Password: Set in `.env` file

2. **Prometheus Metrics**: `https://your-domain:9091`

3. **ArthaChain Metrics**: `https://metrics.your-domain.com/metrics`

### Custom Alerts

Edit `deploy/monitoring/prometheus.yml` to add custom alert rules.

## 🔄 Maintenance & Updates

### Updating ArthaChain
```bash
cd /opt/arthachain/deploy

# Pull latest changes (when available from your repository)
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup Strategy
```bash
#!/bin/bash
# Add to crontab for automated backups
# 0 2 * * * /opt/arthachain/backup.sh

BACKUP_DIR="/backup/arthachain"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup blockchain data
docker run --rm -v arthachain_node_data:/data -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/blockchain_data_$DATE.tar.gz -C /data .

# Backup configuration
tar czf $BACKUP_DIR/config_$DATE.tar.gz /opt/arthachain/deploy

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Log Management
```bash
# View logs
docker-compose logs -f node

# Check log sizes
docker system df

# Clean old logs
docker system prune -f
```

## 📈 Scaling & Performance

### Horizontal Scaling
```bash
# Scale to multiple nodes
docker-compose up -d --scale node=3
```

### Performance Tuning
Edit `blockchain_node/testnet_config_global.toml`:
```toml
[consensus]
block_time = 1          # Faster blocks
max_tx_pool_size = 50000 # Larger mempool

[storage] 
rocksdb_max_files = 5000
memmap_size = 4294967296 # 4GB

[network_p2p]
max_peers = 200
```

## 🆘 Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check what's using ports
sudo netstat -tlnp | grep :3000
sudo lsof -i :3000

# Kill conflicting processes
sudo kill -9 PID
```

#### 2. SSL Certificate Issues
```bash
# Check Caddy logs
docker-compose logs caddy

# Manually trigger certificate
docker-compose exec caddy caddy reload --config /etc/caddy/Caddyfile
```

#### 3. Blockchain Sync Issues
```bash
# Check node status
curl http://localhost:3000/api/status

# Restart node
docker-compose restart node

# Check peer connections
curl http://localhost:3000/api/network/peers
```

#### 4. High Resource Usage
```bash
# Monitor resources
docker stats

# Check disk usage
df -h
docker system df

# Clean up
docker system prune -f
```

### Log Analysis
```bash
# Real-time monitoring
docker-compose logs -f --tail=100 node

# Search for errors
docker-compose logs node | grep -i error

# Export logs for analysis
docker-compose logs node > arthachain.log
```

## 🌐 Production Network Setup

### Multi-Node Network
For a production network with multiple validators:

1. Deploy multiple nodes across different regions
2. Update boot nodes in configuration
3. Set up proper validator keys
4. Configure cross-region networking

### Load Balancer Configuration
```nginx
upstream arthachain_api {
    server node1.your-domain.com:3000;
    server node2.your-domain.com:3000;
    server node3.your-domain.com:3000;
}

server {
    listen 443 ssl;
    server_name api.your-domain.com;
    
    location / {
        proxy_pass http://arthachain_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📞 Support & Community

- **GitHub Issues**: Report bugs and feature requests
- **Discord**: Join the community for support
- **Documentation**: Comprehensive docs at docs.arthachain.com
- **Status Page**: Monitor network status

## ✅ Production Checklist

Before going live, ensure:

- [ ] SSL certificates are working
- [ ] All monitoring dashboards accessible
- [ ] Backup strategy implemented
- [ ] Security hardening completed
- [ ] Performance testing done
- [ ] Disaster recovery plan in place
- [ ] Support team trained
- [ ] Documentation updated

---

**🎉 Congratulations! Your ArthaChain node is now production-ready!**

For advanced configurations and enterprise features, contact the ArthaChain team.
