#!/bin/bash

# ArthaChain Production Security Hardening Script
# Run this script on your production server to apply security best practices

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔒 ArthaChain Production Security Hardening${NC}"
echo "============================================="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}❌ This script must be run as root${NC}"
   exit 1
fi

# Update system
echo -e "${YELLOW}📦 Updating system packages...${NC}"
apt update && apt upgrade -y

# Install security tools
echo -e "${YELLOW}🛠️ Installing security tools...${NC}"
apt install -y \
    ufw \
    fail2ban \
    unattended-upgrades \
    aide \
    rkhunter \
    chkrootkit \
    logwatch \
    htop \
    iftop \
    nethogs

# Configure firewall
echo -e "${YELLOW}🔥 Configuring UFW firewall...${NC}"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing

# SSH access (change port if needed)
ufw allow 22/tcp
echo -e "${GREEN}✅ SSH access allowed on port 22${NC}"

# ArthaChain specific ports
ufw allow 30303/tcp comment 'ArthaChain P2P'
echo -e "${GREEN}✅ P2P port 30303 opened${NC}"

# HTTP/HTTPS for API access
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'
echo -e "${GREEN}✅ HTTP/HTTPS ports opened${NC}"

# Enable firewall
ufw --force enable

# Configure fail2ban
echo -e "${YELLOW}🚫 Configuring fail2ban...${NC}"
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 3
EOF

systemctl enable fail2ban
systemctl restart fail2ban
echo -e "${GREEN}✅ Fail2ban configured and started${NC}"

# SSH hardening
echo -e "${YELLOW}🔐 Hardening SSH configuration...${NC}"
SSH_CONFIG="/etc/ssh/sshd_config"

# Backup original config
cp $SSH_CONFIG $SSH_CONFIG.backup

# Apply hardening
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' $SSH_CONFIG
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' $SSH_CONFIG
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' $SSH_CONFIG
sed -i 's/#AuthorizedKeysFile/AuthorizedKeysFile/' $SSH_CONFIG
sed -i 's/X11Forwarding yes/X11Forwarding no/' $SSH_CONFIG
sed -i 's/#MaxAuthTries 6/MaxAuthTries 3/' $SSH_CONFIG
sed -i 's/#ClientAliveInterval 0/ClientAliveInterval 300/' $SSH_CONFIG
sed -i 's/#ClientAliveCountMax 3/ClientAliveCountMax 2/' $SSH_CONFIG

# Add SSH hardening directives
cat >> $SSH_CONFIG << EOF

# Additional hardening
Protocol 2
IgnoreRhosts yes
HostbasedAuthentication no
PermitEmptyPasswords no
ChallengeResponseAuthentication no
KerberosAuthentication no
GSSAPIAuthentication no
UsePAM yes
AllowUsers ubuntu admin  # Add your actual username here
EOF

systemctl restart sshd
echo -e "${GREEN}✅ SSH hardened and restarted${NC}"

# Configure automatic security updates
echo -e "${YELLOW}🔄 Configuring automatic security updates...${NC}"
cat > /etc/apt/apt.conf.d/20auto-upgrades << EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Unattended-Upgrade "1";
EOF

cat > /etc/apt/apt.conf.d/50unattended-upgrades << EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}";
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}ESMApps:\${distro_codename}-apps-security";
    "\${distro_id}ESM:\${distro_codename}-infra-security";
};

Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

systemctl enable unattended-upgrades
echo -e "${GREEN}✅ Automatic security updates configured${NC}"

# Configure kernel parameters for security
echo -e "${YELLOW}⚙️ Hardening kernel parameters...${NC}"
cat > /etc/sysctl.d/99-arthachain-security.conf << EOF
# IP Spoofing protection
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.rp_filter = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Log Martians
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# Ignore ICMP ping requests
net.ipv4.icmp_echo_ignore_all = 1

# Ignore Directed pings
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Disable IPv6 if not needed
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1

# TCP SYN flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# Network performance and security
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr
EOF

sysctl -p /etc/sysctl.d/99-arthachain-security.conf
echo -e "${GREEN}✅ Kernel parameters hardened${NC}"

# Set up log monitoring
echo -e "${YELLOW}📊 Configuring log monitoring...${NC}"
cat > /etc/logwatch/conf/logwatch.conf << EOF
MailTo = root
MailFrom = Logwatch
Print = No
Save = /var/cache/logwatch
Range = yesterday
Detail = Med
Service = All
mailer = "/usr/sbin/sendmail -t"
EOF

# Install and configure AIDE (file integrity monitoring)
echo -e "${YELLOW}🔍 Setting up file integrity monitoring...${NC}"
aide --init
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Create cron job for AIDE
cat > /etc/cron.daily/aide << EOF
#!/bin/bash
/usr/bin/aide --check
EOF
chmod +x /etc/cron.daily/aide

# Docker security configuration
echo -e "${YELLOW}🐳 Hardening Docker configuration...${NC}"
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "5"
  },
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true,
  "seccomp-profile": "/etc/docker/seccomp.json",
  "default-ulimits": {
    "nofile": {
      "Hard": 64000,
      "Name": "nofile",
      "Soft": 64000
    }
  }
}
EOF

# Restart Docker with new configuration
systemctl restart docker
echo -e "${GREEN}✅ Docker security configuration applied${NC}"

# Create monitoring and alerting scripts
echo -e "${YELLOW}📈 Setting up monitoring scripts...${NC}"

# System health check script
cat > /usr/local/bin/arthachain-health-check.sh << 'EOF'
#!/bin/bash

# ArthaChain Health Check Script
HEALTH_URL="http://localhost:3000/api/health"
WEBHOOK_URL="$DISCORD_WEBHOOK_URL"

# Check if ArthaChain is responding
if ! curl -f -s $HEALTH_URL > /dev/null; then
    echo "$(date): ArthaChain health check failed" >> /var/log/arthachain-health.log
    
    # Send alert if webhook is configured
    if [[ -n "$WEBHOOK_URL" ]]; then
        curl -H "Content-Type: application/json" -X POST -d '{"content":"🚨 ArthaChain node health check failed!"}' $WEBHOOK_URL
    fi
    
    # Restart ArthaChain service
    cd /opt/arthachain/deploy && docker-compose restart node
fi
EOF

chmod +x /usr/local/bin/arthachain-health-check.sh

# Add health check to crontab
echo "*/5 * * * * /usr/local/bin/arthachain-health-check.sh" | crontab -

# Resource monitoring script
cat > /usr/local/bin/arthachain-resource-monitor.sh << 'EOF'
#!/bin/bash

# Resource monitoring thresholds
CPU_THRESHOLD=80
MEMORY_THRESHOLD=80
DISK_THRESHOLD=90

# Get current usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d' ' -f1)
MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
DISK_USAGE=$(df / | grep -vE '^Filesystem' | awk '{print $5}' | cut -d'%' -f1)

# Check thresholds and alert
if (( $(echo "$CPU_USAGE > $CPU_THRESHOLD" | bc -l) )); then
    echo "$(date): High CPU usage: $CPU_USAGE%" >> /var/log/arthachain-resources.log
fi

if (( $MEMORY_USAGE > $MEMORY_THRESHOLD )); then
    echo "$(date): High memory usage: $MEMORY_USAGE%" >> /var/log/arthachain-resources.log
fi

if (( $DISK_USAGE > $DISK_THRESHOLD )); then
    echo "$(date): High disk usage: $DISK_USAGE%" >> /var/log/arthachain-resources.log
fi
EOF

chmod +x /usr/local/bin/arthachain-resource-monitor.sh

# Add resource monitoring to crontab
echo "*/10 * * * * /usr/local/bin/arthachain-resource-monitor.sh" | crontab -

# Create backup script
cat > /usr/local/bin/arthachain-backup.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/backup/arthachain"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup blockchain data
docker run --rm -v arthachain_node_data:/data -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/blockchain_data_$DATE.tar.gz -C /data .

# Backup configuration
tar czf $BACKUP_DIR/config_$DATE.tar.gz /opt/arthachain/deploy

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "$(date): Backup completed: blockchain_data_$DATE.tar.gz" >> /var/log/arthachain-backup.log
EOF

chmod +x /usr/local/bin/arthachain-backup.sh

# Add backup to crontab (daily at 2 AM)
echo "0 2 * * * /usr/local/bin/arthachain-backup.sh" | crontab -

# Set up log rotation
echo -e "${YELLOW}🔄 Configuring log rotation...${NC}"
cat > /etc/logrotate.d/arthachain << EOF
/var/log/arthachain*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF

# Final security checks
echo -e "${YELLOW}🔍 Running final security checks...${NC}"

# Check for common security issues
if ! command -v rkhunter &> /dev/null; then
    echo -e "${RED}❌ rkhunter not installed${NC}"
else
    rkhunter --update
    echo -e "${GREEN}✅ rkhunter updated${NC}"
fi

# Generate summary report
echo -e "${BLUE}📋 Security Hardening Summary${NC}"
echo "================================"
echo -e "${GREEN}✅ System packages updated${NC}"
echo -e "${GREEN}✅ UFW firewall configured and enabled${NC}"
echo -e "${GREEN}✅ Fail2ban configured for intrusion prevention${NC}"
echo -e "${GREEN}✅ SSH hardened (root login disabled, key-only auth)${NC}"
echo -e "${GREEN}✅ Automatic security updates enabled${NC}"
echo -e "${GREEN}✅ Kernel security parameters configured${NC}"
echo -e "${GREEN}✅ File integrity monitoring (AIDE) configured${NC}"
echo -e "${GREEN}✅ Docker security hardened${NC}"
echo -e "${GREEN}✅ Health monitoring scripts deployed${NC}"
echo -e "${GREEN}✅ Automated backup system configured${NC}"
echo -e "${GREEN}✅ Log rotation configured${NC}"

echo ""
echo -e "${BLUE}🔧 Manual Steps Required:${NC}"
echo "1. Update SSH AllowUsers in /etc/ssh/sshd_config with your actual username"
echo "2. Set up Discord/Slack webhooks in environment variables for alerts"
echo "3. Configure external backup storage (S3, etc.)"
echo "4. Set up external monitoring service (UptimeRobot, Pingdom, etc.)"
echo "5. Review and test all configurations"
echo "6. Document your security procedures"

echo ""
echo -e "${YELLOW}⚠️  Important Security Notes:${NC}"
echo "- Change default passwords immediately"
echo "- Use strong, unique passwords for all accounts"
echo "- Enable 2FA where possible"
echo "- Regularly review and update security configurations"
echo "- Monitor logs for suspicious activity"
echo "- Keep system and applications updated"

echo ""
echo -e "${GREEN}🎉 Security hardening complete!${NC}"
echo "Your ArthaChain node is now significantly more secure."
echo "Remember to regularly review and update your security measures."
