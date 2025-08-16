# 🚀 Quick Setup Guide for ArthaChain Telegram Faucet Bot

## 📋 Prerequisites

1. **Create a Telegram Bot**:
   - Open Telegram and message [@BotFather](https://t.me/botfather)
   - Send `/newbot` and follow the instructions
   - Choose a name like "ArthaChain Faucet Bot"
   - Choose a username like "ArthaChainFaucetBot"
   - **Save the bot token** you receive (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

2. **ArthaChain Node Running**:
   - Ensure your ArthaChain testnet API is running on `https://api.arthachain.in`
   - Test with: `curl https://api.arthachain.in/api/health`

## ⚡ Quick Start

1. **Set Environment Variable**:
   ```bash
   export TELEGRAM_BOT_TOKEN="your_bot_token_here"
   export FAUCET_API_URL="https://api.arthachain.in"
   ```

2. **Run the Bot**:
   ```bash
   cargo run --bin arthachain_faucet_bot
   ```

3. **Test the Bot**:
   - Open Telegram and search for your bot
   - Send `/start` to see the welcome message
   - Try `/faucet 0x742d35Cc6634C0532925a3b844Bc454e4438f44e`

## 🎯 Bot Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Show welcome message | `/start` |
| `/faucet <address>` | Request 2.0 ARTHA tokens | `/faucet 0x742d35...44e` |
| `/balance <address>` | Check wallet balance | `/balance 0x742d35...44e` |
| `/status` | Get faucet and network status | `/status` |
| `/network` | View network statistics | `/network` |
| `/stats` | View bot usage statistics | `/stats` |
| `/help` | Show help message | `/help` |

## 🛡️ Security Features

- ✅ **Rate Limiting**: 24-hour cooldown per user
- ✅ **Address Validation**: Validates Ethereum-compatible addresses
- ✅ **Error Handling**: Graceful API failure handling
- ✅ **Usage Tracking**: Comprehensive statistics

## 🔧 Production Setup

### Environment Variables
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export FAUCET_API_URL="https://api.arthachain.in"
export RUST_LOG="info"
```

### Run in Background
```bash
nohup cargo run --release --bin arthachain_faucet_bot > bot.log 2>&1 &
```

### Check Logs
```bash
tail -f bot.log
```

## 📊 Expected Output

When running successfully, you should see:
```
🚀 Starting ArthaChain Telegram Faucet Bot...
✅ Bot initialized. Faucet API: https://api.arthachain.in
```

## 🔍 Troubleshooting

### Bot Not Responding
1. Verify bot token is correct
2. Check if bot is started with @BotFather
3. Ensure network connectivity

### Faucet Requests Failing
1. Check if ArthaChain node is running: `curl https://api.arthachain.in/api/health`
2. Verify API endpoints are accessible
3. Check rate limiting (24-hour cooldown)

### Common Issues
- **Invalid Token**: Double-check the bot token from @BotFather
- **API Unreachable**: Ensure ArthaChain node is running on correct port
- **Permission Denied**: Make sure bot has permission to send messages

## 🎉 Success!

Once running, users can:
- Request 2.0 ARTHA tokens instantly
- Check their wallet balance
- View real-time network statistics
- Get faucet status information

The bot integrates seamlessly with your ArthaChain testnet! 🚀