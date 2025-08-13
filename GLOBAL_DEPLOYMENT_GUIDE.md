# 🌍 **ARTHACHAIN GLOBAL TESTNET DEPLOYMENT GUIDE**

## **🚀 DEPLOY YOUR TESTNET WORLDWIDE IN 5 MINUTES!**

---

## **📋 TABLE OF CONTENTS**
1. [Quick Start - One-Click Deploy](#quick-start)
2. [Cloud Provider Deployment](#cloud-deployment)
3. [VPS/Dedicated Server Setup](#vps-setup)
4. [Domain & DNS Configuration](#dns-setup)
5. [Security Configuration](#security)
6. [Monitoring & Maintenance](#monitoring)

---

## **🚀 QUICK START - ONE-CLICK DEPLOY** {#quick-start}

### **Option 1: DigitalOcean (Easiest)**
```bash
# 1. Create a DigitalOcean account and get API token
# 2. Run our automated deployment
curl -sSL https://arthachain.io/deploy/digitalocean.sh | bash

# 3. Follow the prompts to enter:
#    - Your DigitalOcean API token
#    - Desired region (nyc3, sfo3, lon1, etc.)
#    - Node size (recommended: 4GB RAM minimum)
```

### **Option 2: AWS EC2**
```bash
# Using AWS CLI
aws cloudformation create-stack \
  --stack-name arthachain-testnet \
  --template-url https://arthachain.io/deploy/aws-template.json \
  --parameters ParameterKey=InstanceType,ParameterValue=t3.large
```

### **Option 3: Google Cloud Platform**
```bash
# Using gcloud CLI
gcloud deployment-manager deployments create arthachain-testnet \
  --config https://arthachain.io/deploy/gcp-config.yaml
```

---

## **☁️ CLOUD PROVIDER DEPLOYMENT** {#cloud-deployment}

### **🔷 DIGITALOCEAN DEPLOYMENT**

1. **Create Droplet**
```bash
doctl compute droplet create arthachain-testnet \
  --region nyc3 \
  --image ubuntu-22-04-x64 \
  --size s-4vcpu-8gb \
  --ssh-keys YOUR_SSH_KEY_ID \
  --user-data-file deploy-script.sh
```

2. **Deployment Script (deploy-script.sh)**
```bash
#!/bin/bash
# ArthaChain Global Testnet Auto-Deploy Script

# Update system
apt-get update && apt-get upgrade -y

# Install dependencies
apt-get install -y build-essential pkg-config libssl-dev git curl

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Clone ArthaChain
git clone https://github.com/arthachain/arthachain.git
cd arthachain

# Use global config
cp blockchain_node/testnet_config_global.toml blockchain_node/testnet_config.toml

# Build
cargo build --release

# Configure firewall
ufw allow 22/tcp    # SSH
ufw allow 30303/tcp # P2P
ufw allow 8545/tcp  # HTTP RPC
ufw allow 8546/tcp  # WebSocket
ufw allow 3000/tcp  # REST API
ufw allow 9090/tcp  # Metrics
ufw --force enable

# Create systemd service
cat > /etc/systemd/system/arthachain.service << EOF
[Unit]
Description=ArthaChain Testnet Node
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/arthachain
ExecStart=/root/arthachain/target/release/arthachain run --config testnet_config.toml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start service
systemctl daemon-reload
systemctl enable arthachain
systemctl start arthachain

echo "✅ ArthaChain Testnet deployed successfully!"
echo "🌐 Access your node at: http://$(curl -s ifconfig.me):3000"
```

### **🔶 AWS EC2 DEPLOYMENT**

1. **Launch Instance**
   - AMI: Ubuntu Server 22.04 LTS
   - Instance Type: t3.large (minimum)
   - Storage: 100GB SSD
   - Security Group:
     ```
     - SSH (22): Your IP
     - Custom TCP (30303): 0.0.0.0/0
     - Custom TCP (8545): 0.0.0.0/0
     - Custom TCP (8546): 0.0.0.0/0
     - Custom TCP (3000): 0.0.0.0/0
     - Custom TCP (9090): 0.0.0.0/0
     ```

2. **Deploy Using User Data**
```bash
#!/bin/bash
# Same script as above, but add:

# Configure AWS-specific settings
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d " " -f 2)
PUBLIC_IP=$(ec2-metadata --public-ipv4 | cut -d " " -f 2)

# Update config with public IP
sed -i "s/public_ip = \"\"/public_ip = \"$PUBLIC_IP\"/" testnet_config.toml
```

### **🔵 GOOGLE CLOUD DEPLOYMENT**

```bash
# Create VM instance
gcloud compute instances create arthachain-testnet \
  --machine-type=n2-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --tags=arthachain \
  --metadata-from-file startup-script=deploy-script.sh

# Create firewall rules
gcloud compute firewall-rules create arthachain-p2p \
  --allow tcp:30303 --source-ranges 0.0.0.0/0 --target-tags arthachain

gcloud compute firewall-rules create arthachain-rpc \
  --allow tcp:8545,tcp:8546,tcp:3000,tcp:9090 \
  --source-ranges 0.0.0.0/0 --target-tags arthachain
```

---

## **🖥️ VPS/DEDICATED SERVER SETUP** {#vps-setup}

### **Minimum Requirements**
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 100GB SSD
- **Bandwidth**: 1TB/month
- **OS**: Ubuntu 22.04 LTS

### **Popular VPS Providers**
1. **Hetzner** - Best value (€13/month)
2. **OVH** - Good EU coverage (€20/month)
3. **Linode** - Global presence ($40/month)
4. **Vultr** - High performance ($40/month)

### **Manual Setup Steps**
```bash
# 1. SSH into your server
ssh root@YOUR_SERVER_IP

# 2. Download and run setup script
wget https://raw.githubusercontent.com/arthachain/arthachain/main/deploy-script.sh
chmod +x deploy-script.sh
./deploy-script.sh

# 3. Configure public IP in config
nano blockchain_node/testnet_config.toml
# Set: public_ip = "YOUR_SERVER_IP"

# 4. Restart service
systemctl restart arthachain
```

---

## **🌐 DOMAIN & DNS CONFIGURATION** {#dns-setup}

### **1. Domain Setup**
```
Example: testnet.arthachain.com

A Records:
- testnet.arthachain.com → YOUR_SERVER_IP
- rpc.testnet.arthachain.com → YOUR_SERVER_IP
- api.testnet.arthachain.com → YOUR_SERVER_IP
- ws.testnet.arthachain.com → YOUR_SERVER_IP
```

### **2. Cloudflare Setup (Recommended)**
1. Add your domain to Cloudflare
2. Configure DNS records
3. Enable SSL/TLS: Full (strict)
4. Configure Page Rules:
   ```
   *testnet.arthachain.com/rpc* - Cache Level: Bypass
   *testnet.arthachain.com/api* - Cache Level: Bypass
   ```

### **3. Nginx Reverse Proxy**
```nginx
# /etc/nginx/sites-available/arthachain
server {
    listen 80;
    server_name testnet.arthachain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name testnet.arthachain.com;
    
    ssl_certificate /etc/letsencrypt/live/testnet.arthachain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/testnet.arthachain.com/privkey.pem;
    
    # API endpoint
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # RPC endpoint
    location /rpc {
        proxy_pass http://localhost:8545;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # WebSocket endpoint
    location /ws {
        proxy_pass http://localhost:8546;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### **4. SSL Certificate (Let's Encrypt)**
```bash
# Install certbot
snap install --classic certbot

# Get certificate
certbot --nginx -d testnet.arthachain.com -d rpc.testnet.arthachain.com

# Auto-renew
certbot renew --dry-run
```

---

## **🔒 SECURITY CONFIGURATION** {#security}

### **1. Basic Security Hardening**
```bash
# Update system
apt update && apt upgrade -y

# Configure firewall (already done in deploy script)
ufw status

# Fail2ban for SSH protection
apt install fail2ban -y
systemctl enable fail2ban

# Disable root login
sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
systemctl restart sshd
```

### **2. DDoS Protection**
```bash
# Install and configure DDoS deflate
wget https://github.com/jgmdev/ddos-deflate/archive/master.zip
unzip master.zip
cd ddos-deflate-master
./install.sh
```

### **3. Rate Limiting (iptables)**
```bash
# Limit connections per IP
iptables -A INPUT -p tcp --dport 8545 -m connlimit --connlimit-above 10 -j REJECT
iptables -A INPUT -p tcp --dport 3000 -m connlimit --connlimit-above 20 -j REJECT

# Save rules
iptables-save > /etc/iptables/rules.v4
```

---

## **📊 MONITORING & MAINTENANCE** {#monitoring}

### **1. Basic Monitoring**
```bash
# Check node status
systemctl status arthachain

# View logs
journalctl -u arthachain -f

# Check connections
netstat -tnlp | grep arthachain
```

### **2. Prometheus + Grafana Setup**
```bash
# Install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.37.0/prometheus-2.37.0.linux-amd64.tar.gz
tar xvf prometheus-2.37.0.linux-amd64.tar.gz
cd prometheus-2.37.0.linux-amd64

# Configure prometheus.yml
cat > prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'arthachain'
    static_configs:
      - targets: ['localhost:9090']
EOF

# Install Grafana
apt-get install -y software-properties-common
add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | apt-key add -
apt-get update
apt-get install grafana

# Start services
systemctl start prometheus
systemctl start grafana-server
```

### **3. Uptime Monitoring**
- **UptimeRobot**: Free monitoring
- **Pingdom**: Professional monitoring
- **StatusCake**: Good free tier

### **4. Alerts Setup**
```bash
# Discord webhook alerts
WEBHOOK_URL="https://discord.com/api/webhooks/YOUR_WEBHOOK"

# CPU alert
if [ $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1) -gt 80 ]; then
  curl -H "Content-Type: application/json" -X POST -d '{"content":"⚠️ High CPU usage on testnet!"}' $WEBHOOK_URL
fi
```

---

## **🎉 YOUR TESTNET IS NOW GLOBALLY ACCESSIBLE!**

### **📡 Access Points**
```
🌐 REST API: http://YOUR_IP:3000
📊 RPC Endpoint: http://YOUR_IP:8545
🔌 WebSocket: ws://YOUR_IP:8546
📈 Metrics: http://YOUR_IP:9090

With domain:
🌐 REST API: https://api.testnet.arthachain.com
📊 RPC: https://rpc.testnet.arthachain.com
🔌 WebSocket: wss://ws.testnet.arthachain.com
```

### **🧪 Test Your Deployment**
```bash
# Health check
curl http://YOUR_IP:3000/health

# Get block height
curl http://YOUR_IP:3000/blocks/latest

# Connect with Web3
const web3 = new Web3('http://YOUR_IP:8545');
await web3.eth.getBlockNumber();
```

### **📱 Share Your Testnet**
```
🔗 Network Name: ArthaChain Testnet
🆔 Chain ID: 1337
🌐 RPC URL: http://YOUR_IP:8545
💰 Symbol: ARTHA
🔍 Explorer: http://YOUR_IP:3000
```

---

## **🆘 TROUBLESHOOTING**

### **Common Issues**

1. **Port already in use**
   ```bash
   lsof -i :3000
   kill -9 PID
   ```

2. **Can't connect externally**
   - Check firewall: `ufw status`
   - Check cloud security groups
   - Verify public IP in config

3. **Node not syncing**
   - Check boot nodes are accessible
   - Verify network connectivity
   - Check logs: `journalctl -u arthachain -n 100`

### **📞 Support**
- Discord: https://discord.gg/arthachain
- Telegram: https://t.me/arthachain
- Email: support@arthachain.io

---

**🎊 CONGRATULATIONS! Your ArthaChain testnet is now accessible worldwide!**
