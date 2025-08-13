# ArthaChain Disaster Recovery Playbook

## 🚨 Emergency Contact Information

**Critical Support Contacts:**
- Primary On-Call Engineer: +1-XXX-XXX-XXXX
- Secondary Engineer: +1-XXX-XXX-XXXX  
- DevOps Team Lead: +1-XXX-XXX-XXXX
- Emergency Escalation: emergency@arthachain.org

**Monitoring & Alert Systems:**
- Grafana Dashboard: https://monitor.arthachain.org
- PagerDuty: https://arthachain.pagerduty.com
- Status Page: https://status.arthachain.org

---

## 📋 Quick Reference - Common Recovery Commands

### 🔍 System Status Check
```bash
# Check overall system health
curl -X GET http://localhost:8080/api/recovery/status

# Check specific component health
curl -X GET http://localhost:8080/api/health

# View active alerts
curl -X GET http://localhost:8080/api/alerts/active
```

### 🔄 Basic Recovery Operations
```bash
# Force leader election
curl -X POST http://localhost:8080/api/recovery/execute \
  -H "Content-Type: application/json" \
  -d '{"operation": "ForceLeaderElection", "force": false}'

# Restart from checkpoint
curl -X POST http://localhost:8080/api/recovery/execute \
  -H "Content-Type: application/json" \
  -d '{"operation": "RestartFromCheckpoint", "force": false}'

# Emergency shutdown
curl -X POST http://localhost:8080/api/recovery/execute \
  -H "Content-Type: application/json" \
  -d '{"operation": "EmergencyShutdown", "force": true}'
```

---

## 🎯 Failure Scenarios & Recovery Procedures

### 1. 🏛️ **CONSENSUS FAILURE**

#### Symptoms:
- No new blocks being produced
- Validator nodes out of sync
- Leader election failures
- Alert: "Consensus Failure" (Emergency)

#### Immediate Actions:
1. **Assess Situation** (2 minutes)
   ```bash
   # Check consensus state
   curl -X GET http://localhost:8080/api/recovery/status | jq '.consensus'
   
   # Check validator status
   curl -X GET http://localhost:8080/api/consensus/validators
   ```

2. **Force Leader Election** (1 minute)
   ```bash
   curl -X POST http://localhost:8080/api/recovery/execute \
     -H "Content-Type: application/json" \
     -d '{"operation": "ForceLeaderElection", "force": true}'
   ```

3. **If Election Fails - Force Failover** (2 minutes)
   ```bash
   # Identify healthy validator
   curl -X GET http://localhost:8080/api/health | jq '.validators'
   
   # Force failover to specific node
   curl -X POST http://localhost:8080/api/recovery/execute \
     -H "Content-Type: application/json" \
     -d '{"operation": "ForceFailover", "target_node": "NODE_ID", "force": true}'
   ```

4. **Last Resort - Full System Recovery** (5-10 minutes)
   ```bash
   curl -X POST http://localhost:8080/api/recovery/execute \
     -H "Content-Type: application/json" \
     -d '{"operation": "FullSystemRecovery", "force": true}'
   ```

#### Escalation Path:
- **0-5 minutes**: Auto-recovery attempts
- **5-10 minutes**: Manual intervention by on-call engineer
- **10+ minutes**: Escalate to senior team

---

### 2. 💾 **STORAGE CORRUPTION**

#### Symptoms:
- Storage integrity checks failing
- Database corruption errors
- Node unable to sync state
- Alert: "Storage Corruption Detected" (Critical)

#### Immediate Actions:
1. **Verify Corruption** (1 minute)
   ```bash
   # Check storage integrity
   curl -X GET http://localhost:8080/api/storage/integrity
   
   # List available backups
   curl -X GET http://localhost:8080/api/recovery/backups
   ```

2. **Restore from Latest Backup** (5-15 minutes)
   ```bash
   # Get latest backup ID
   BACKUP_ID=$(curl -s http://localhost:8080/api/recovery/backups | jq -r '.[0].id')
   
   # Restore from backup
   curl -X POST http://localhost:8080/api/recovery/execute \
     -H "Content-Type: application/json" \
     -d "{\"operation\": \"RestoreFromBackup\", \"backup_id\": \"$BACKUP_ID\", \"force\": true}"
   ```

3. **Verify Recovery** (2 minutes)
   ```bash
   # Check storage health
   curl -X GET http://localhost:8080/api/health | jq '.storage'
   
   # Restart consensus
   curl -X POST http://localhost:8080/api/recovery/execute \
     -H "Content-Type: application/json" \
     -d '{"operation": "ForceLeaderElection", "force": false}'
   ```

#### Advanced Recovery:
If automatic backup restore fails:

1. **Manual State Rebuild** (30+ minutes)
   ```bash
   # Stop node
   sudo systemctl stop arthachain
   
   # Backup corrupted data
   sudo mv /var/lib/arthachain/data /var/lib/arthachain/data.corrupted
   
   # Restore from cloud backup
   aws s3 sync s3://arthachain-backups/latest /var/lib/arthachain/data
   
   # Restart node
   sudo systemctl start arthachain
   ```

---

### 3. 🌐 **NETWORK PARTITION**

#### Symptoms:
- Network partition alerts
- Nodes unable to communicate
- Peer disconnections
- Alert: "Network Partition Detected" (Warning)

#### Immediate Actions:
1. **Assess Partition** (2 minutes)
   ```bash
   # Check network status
   curl -X GET http://localhost:8080/api/recovery/status | jq '.network'
   
   # List active partitions
   curl -X GET http://localhost:8080/api/network/partitions
   ```

2. **Heal Partition** (3-5 minutes)
   ```bash
   # Get partition ID
   PARTITION_ID=$(curl -s http://localhost:8080/api/network/partitions | jq -r 'keys[0]')
   
   # Force healing
   curl -X POST http://localhost:8080/api/recovery/execute \
     -H "Content-Type: application/json" \
     -d "{\"operation\": \"HealPartition\", \"partition_id\": \"$PARTITION_ID\", \"force\": true}"
   ```

3. **Network Reset if Needed** (5 minutes)
   ```bash
   curl -X POST http://localhost:8080/api/recovery/execute \
     -H "Content-Type: application/json" \
     -d '{"operation": "NetworkReset", "force": true}'
   ```

#### Manual Network Recovery:
```bash
# Check network connectivity
ping validator-node-1.arthachain.org
ping validator-node-2.arthachain.org

# Restart network interface
sudo ip link set eth0 down
sudo ip link set eth0 up

# Restart node if needed
sudo systemctl restart arthachain
```

---

### 4. 🔥 **NODE CRASH / TOTAL FAILURE**

#### Symptoms:
- Node completely unresponsive
- Process crashed
- System hardware failure
- All health checks failing

#### Immediate Actions:
1. **Check System Status** (1 minute)
   ```bash
   # Check if process is running
   sudo systemctl status arthachain
   
   # Check system resources
   free -h
   df -h
   top -n 1
   ```

2. **Attempt Service Restart** (2 minutes)
   ```bash
   # Restart service
   sudo systemctl restart arthachain
   
   # Check logs
   sudo journalctl -u arthachain -f --since "5 minutes ago"
   ```

3. **Restore from Checkpoint** (5-10 minutes)
   ```bash
   # If restart fails, restore from checkpoint
   curl -X POST http://localhost:8080/api/recovery/execute \
     -H "Content-Type: application/json" \
     -d '{"operation": "RestartFromCheckpoint", "force": true}'
   ```

4. **Emergency Failover** (if node won't recover)
   ```bash
   # Failover to backup node
   ./scripts/emergency-failover.sh backup-node-1
   
   # Update DNS records
   ./scripts/update-dns.sh backup-node-1
   ```

---

### 5. 🔐 **SECURITY INCIDENT**

#### Symptoms:
- Security alerts triggered
- Unusual network activity
- Authentication failures
- Potential DDoS attack

#### Immediate Actions:
1. **Assess Threat** (2 minutes)
   ```bash
   # Check security alerts
   curl -X GET http://localhost:8080/api/alerts/active | grep -i security
   
   # Check connection statistics
   netstat -an | grep :8080 | wc -l
   ```

2. **Enable Enhanced Security** (1 minute)
   ```bash
   # Enable rate limiting
   curl -X POST http://localhost:8080/api/security/rate-limit/enable
   
   # Enable DDoS protection
   curl -X POST http://localhost:8080/api/security/ddos-protection/enable
   ```

3. **Block Malicious IPs** (ongoing)
   ```bash
   # Block specific IP
   sudo ufw deny from 192.168.1.100
   
   # Block IP range
   sudo ufw deny from 192.168.1.0/24
   ```

4. **Emergency Lockdown** (if severe)
   ```bash
   # Enable maintenance mode
   curl -X POST http://localhost:8080/api/maintenance/enable
   
   # Emergency shutdown if needed
   curl -X POST http://localhost:8080/api/recovery/execute \
     -H "Content-Type: application/json" \
     -d '{"operation": "EmergencyShutdown", "force": true}'
   ```

---

## 🛠️ **Maintenance & Preventive Actions**

### Daily Health Checks
```bash
#!/bin/bash
# daily-health-check.sh

echo "=== Daily ArthaChain Health Check ===" 
date

# System health
curl -s http://localhost:8080/api/health | jq '.'

# Storage integrity
curl -s http://localhost:8080/api/storage/integrity

# Recent alerts
curl -s http://localhost:8080/api/alerts/recent

# Backup status
curl -s http://localhost:8080/api/recovery/backups | jq '.[0:3]'

echo "Health check completed"
```

### Weekly Maintenance
```bash
#!/bin/bash
# weekly-maintenance.sh

# Create manual backup
curl -X POST http://localhost:8080/api/recovery/backup

# Clean old logs
find /var/log/arthachain -name "*.log" -mtime +7 -delete

# Update system
sudo apt update && sudo apt upgrade -y

# Restart service for memory cleanup
sudo systemctl restart arthachain
```

### Monthly Disaster Recovery Test
```bash
#!/bin/bash
# monthly-dr-test.sh

echo "=== Monthly DR Test ==="

# Test backup creation
BACKUP_ID=$(curl -s -X POST http://localhost:8080/api/recovery/backup | jq -r '.data.id')
echo "Test backup created: $BACKUP_ID"

# Test leader election (on test network)
curl -X POST http://testnet:8080/api/recovery/execute \
  -d '{"operation": "ForceLeaderElection", "force": false}'

# Test network partition healing (simulated)
echo "DR test completed successfully"
```

---

## 📊 **Monitoring & Alerting**

### Key Metrics to Monitor:
- **Consensus**: Block production rate, validator participation
- **Storage**: Disk usage, corruption checks, backup status
- **Network**: Peer count, partition detection, bandwidth
- **Performance**: CPU, memory, disk I/O
- **Security**: Authentication attempts, rate limiting triggers

### Alert Response Times:
- **Emergency** (Consensus failure): < 5 minutes
- **Critical** (Storage corruption): < 15 minutes  
- **Warning** (Network partition): < 30 minutes
- **Info** (Performance): < 1 hour

### Escalation Matrix:
1. **Auto-resolution** (0-5 minutes)
2. **On-call engineer** (5-15 minutes)
3. **Team lead** (15-30 minutes)
4. **Management** (30+ minutes)

---

## 🔧 **Tools & Scripts**

### Essential Scripts:
```bash
# Emergency recovery toolkit
/opt/arthachain/scripts/
├── emergency-failover.sh      # Automatic failover to backup
├── health-check.sh           # Comprehensive health check
├── backup-restore.sh         # Manual backup/restore
├── network-reset.sh          # Network connectivity reset
├── log-analyzer.sh          # Parse and analyze logs
└── security-lockdown.sh     # Emergency security measures
```

### Configuration Files:
```bash
# Key configuration locations
/etc/arthachain/
├── node.toml                # Main node configuration
├── consensus.toml           # Consensus parameters
├── storage.toml            # Storage configuration  
├── network.toml            # Network settings
├── security.toml           # Security policies
└── monitoring.toml         # Monitoring configuration
```

---

## 📞 **Emergency Procedures**

### Severity 1 - System Down (Emergency)
1. **Immediately contact on-call engineer**
2. **Execute emergency recovery procedures**
3. **Update status page**
4. **Notify management within 15 minutes**
5. **Begin incident documentation**

### Severity 2 - Degraded Performance (Critical)  
1. **Attempt automatic recovery**
2. **Contact on-call engineer if auto-recovery fails**
3. **Monitor closely for escalation**
4. **Document incident**

### Severity 3 - Minor Issues (Warning)
1. **Auto-remediation where possible**
2. **Monitor for patterns**
3. **Schedule non-urgent maintenance**

### Post-Incident Actions:
1. **Root cause analysis**
2. **Update runbooks**
3. **Improve monitoring**
4. **Conduct team retrospective**

---

## 🔍 **Troubleshooting Guide**

### Common Issues:

#### "Consensus timeout"
- **Cause**: Network latency or validator failure
- **Fix**: Check network connectivity, force leader election

#### "Storage integrity check failed"  
- **Cause**: Disk corruption or hardware failure
- **Fix**: Restore from backup, check disk health

#### "High memory usage"
- **Cause**: Memory leak or heavy load
- **Fix**: Restart service, investigate logs

#### "Peer disconnections"
- **Cause**: Network issues or firewall changes
- **Fix**: Check firewall rules, restart network

---

## 📚 **Additional Resources**

- **Architecture Documentation**: [docs/BLOCKCHAIN_ARCHITECTURE.md](BLOCKCHAIN_ARCHITECTURE.md)
- **Monitoring Guide**: [docs/monitoring.md](monitoring.md)
- **Security Best Practices**: [docs/security.md](security.md)
- **API Documentation**: [docs/api.md](api.md)

---

## ✅ **Recovery Verification Checklist**

After any recovery operation:

- [ ] All health checks passing
- [ ] Consensus producing blocks
- [ ] Storage integrity verified
- [ ] Network connectivity restored
- [ ] Monitoring alerts cleared
- [ ] Performance metrics normal
- [ ] Security systems active
- [ ] Backup systems operational
- [ ] Documentation updated
- [ ] Team notification sent

---

## 🔄 **Continuous Improvement**

This playbook should be:
- **Reviewed monthly** for accuracy
- **Updated after incidents** with lessons learned
- **Tested quarterly** with DR exercises
- **Validated annually** with full system tests

**Last Updated**: {Current Date}
**Version**: 1.0
**Next Review**: {Date + 1 month} 