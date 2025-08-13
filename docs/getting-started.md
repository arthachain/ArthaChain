# 🚀 Getting Started with ArthaChain

**Goal**: Get you making transactions and deploying smart contracts in under 10 minutes, even if you've never used blockchain before!

## 🎯 What We'll Do (Step by Step)

By the end of this guide, you'll have:
1. ✅ **Set up your development environment** (2 minutes)
2. ✅ **Connected to ArthaChain testnet** (1 minute)
3. ✅ **Created your first wallet** (1 minute)
4. ✅ **Received free test tokens** (1 minute)
5. ✅ **Made your first transaction** (2 minutes)
6. ✅ **Deployed a smart contract** (3 minutes)

**Total time**: ~10 minutes of pure magic! ✨

## 📋 Before We Start (Prerequisites)

You'll need:
- **💻 A computer** (Windows, Mac, or Linux)
- **🌐 Internet connection**
- **That's it!** We'll install everything else together

### 🤔 Don't Have These Programming Tools? No Problem!

**Option 1: Web Playground (Easiest)**
- Go to [playground.arthachain.online](https://playground.arthachain.online)
- Everything runs in your browser!
- Skip to [Step 4: Get Free Tokens](#-step-4-get-free-test-tokens-1-minute)

**Option 2: Follow Along (Recommended)**
- We'll install everything step by step
- More powerful for serious development

## 🚀 Step 1: Set Up Your Environment (2 minutes)

### 🔧 Install Node.js (If You Don't Have It)

**Windows/Mac Users:**
1. Go to [nodejs.org](https://nodejs.org)
2. Download the **LTS version** (the green button)
3. Run the installer and click "Next" through everything
4. Open **Command Prompt** (Windows) or **Terminal** (Mac)

**Linux Users:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install nodejs npm

# CentOS/RHEL
sudo yum install nodejs npm

# Arch Linux
sudo pacman -S nodejs npm
```

**Verify Installation:**
```bash
# Check if Node.js is installed
node --version
# Should show something like: v18.17.0

# Check if npm is installed  
npm --version
# Should show something like: 9.6.7
```

### 📦 Install ArthaChain CLI

```bash
# Install the ArthaChain command-line tool
npm install -g @arthachain/cli

# Verify it worked
arthachain --version
```

**If you get permission errors:**
```bash
# Mac/Linux: Add sudo
sudo npm install -g @arthachain/cli

# Windows: Run Command Prompt as Administrator
```

🎉 **Great! You now have the ArthaChain CLI installed!**

## 🌐 Step 2: Connect to Testnet (1 minute)

The testnet is like a "practice version" of ArthaChain where you can experiment safely with fake money.

```bash
# Connect to the testnet
arthachain network set testnet

# Check connection
arthachain network status
```

**You should see something like:**
```
✅ Connected to ArthaChain Testnet
📡 RPC: https://testnet.arthachain.online/rpc
🌐 Explorer: https://testnet.arthachain.online
⚡ Block Height: 142,857
🚀 TPS: 450+
```

**Troubleshooting:**
```bash
# If connection fails, try:
arthachain network set testnet --force

# Or manually set the RPC:
arthachain config set rpc-url https://testnet.arthachain.online/rpc
```

## 🔑 Step 3: Create Your Wallet (1 minute)

A wallet is like your digital bank account on the blockchain.

```bash
# Create a new wallet
arthachain wallet create

# You'll see something like:
```

**Example Output:**
```
🎉 Wallet Created Successfully!

📧 Address: artha1xyz123abc456def789...
🔑 Private Key: ********** (HIDDEN - Keep this SECRET!)
📝 Mnemonic: word1 word2 word3 word4 word5 word6...

⚠️  CRITICAL: Save your mnemonic phrase!
   Write it down on paper and store it safely.
   This is the ONLY way to recover your wallet!
```

### 🔐 **SUPER IMPORTANT - Save Your Mnemonic!**

The mnemonic (12-24 words) is like the master key to your wallet. If you lose it, you lose access to your funds **forever**!

**Best Practices:**
1. ✍️ **Write it down on paper** (not digital)
2. 🏠 **Store in a safe place** (not on your computer)
3. 🚫 **Never share it** with anyone
4. 📸 **Don't take photos** of it
5. 💾 **Make multiple copies** and store separately

**Example of saving securely:**
```bash
# Create a secure backup file (optional)
arthachain wallet export backup.json

# This creates an encrypted backup file
# You'll need a password to decrypt it later
```

### 👀 Check Your Wallet

```bash
# View your wallet address
arthachain wallet address

# View your balance (should be 0 for now)
arthachain wallet balance
```

## 💰 Step 4: Get Free Test Tokens (1 minute)

Time to get some free ARTHA tokens to play with!

### 🤖 Method 1: CLI Faucet (Easiest)

```bash
# Request free tokens
arthachain faucet request

# Check your balance
arthachain wallet balance
```

**You should see:**
```
💰 Balance: 1000.0 ARTHA
```

### 🌐 Method 2: Web Faucet

1. Go to [faucet.arthachain.online](https://faucet.arthachain.online)
2. Paste your address (get it with `arthachain wallet address`)
3. Click "Request Tokens"
4. Wait 30 seconds, then check balance

### 💬 Method 3: Discord Bot

1. Join [discord.gg/arthachain](https://discord.gg/arthachain)
2. Go to the `#faucet` channel
3. Type: `/faucet your-wallet-address`
4. The bot will send you tokens!

### 📱 Method 4: Telegram Bot

1. Open [@ArthachainFaucetBot](https://t.me/ArthachainFaucetBot)
2. Send: `/faucet your-wallet-address`
3. Get instant tokens!

**Faucet Limits:**
- 🕒 **1000 ARTHA per 24 hours** per address
- 🌐 **Rate limited** to prevent spam
- 🆓 **Completely free** for developers

## 💸 Step 5: Make Your First Transaction (2 minutes)

Let's send some tokens to another address!

### 🎯 Create a Second Wallet (For Testing)

```bash
# Create a second wallet to send money to
arthachain wallet create --name recipient

# Switch back to your main wallet
arthachain wallet use main
```

### 📤 Send Your First Transaction

```bash
# Get recipient address
RECIPIENT=$(arthachain wallet address --name recipient)

# Send 10 ARTHA tokens
arthachain tx send $RECIPIENT 10

# Or manually:
arthachain tx send artha1abc123def456... 10
```

**What happens:**
```
🚀 Sending Transaction...
💰 Amount: 10 ARTHA  
📧 From: artha1xyz123abc456...
📧 To: artha1abc123def456...
⚡ Gas: 21000
💵 Fee: 0.001 ARTHA

✅ Transaction Sent!
📋 Hash: 0x1a2b3c4d5e6f...
🌐 Explorer: https://testnet.arthachain.online/tx/0x1a2b3c4d5e6f...

⏳ Waiting for confirmation...
✅ Confirmed in block 142,858 (2.3 seconds)
```

### 🔍 Check Transaction Status

```bash
# Check transaction by hash
arthachain tx status 0x1a2b3c4d5e6f...

# Check both wallet balances
arthachain wallet balance
arthachain wallet balance --name recipient
```

**Expected result:**
```
Main wallet: 989.999 ARTHA (1000 - 10 - 0.001 fee)
Recipient wallet: 10.0 ARTHA
```

🎉 **Congratulations! You just made your first blockchain transaction!**

## 🤖 Step 6: Deploy Your First Smart Contract (3 minutes)

Smart contracts are programs that run on the blockchain. Let's deploy one!

### 📝 Create a Simple Smart Contract

ArthaChain supports both **WASM** (Rust) and **Solidity** (Ethereum) contracts. We'll start with a simple example.

**Option A: Simple WASM Contract**

```bash
# Create a new contract project
arthachain contract new my-first-contract

# This creates a folder with example code
cd my-first-contract
```

**The generated contract (`src/lib.rs`):**
```rust
// A simple counter contract
use arthachain_sdk::*;

#[contract]
pub struct Counter {
    value: i32,
}

#[contract_impl]
impl Counter {
    // Initialize the contract
    pub fn new() -> Self {
        Self { value: 0 }
    }
    
    // Increment the counter
    pub fn increment(&mut self) {
        self.value += 1;
    }
    
    // Get current value
    pub fn get_value(&self) -> i32 {
        self.value
    }
    
    // Add a specific amount
    pub fn add(&mut self, amount: i32) {
        self.value += amount;
    }
}
```

**Option B: Solidity Contract (Ethereum Compatible)**

```bash
# Create Solidity contract
arthachain contract new --type solidity my-token
cd my-token
```

**The generated contract (`contracts/Counter.sol`):**
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Counter {
    int public value = 0;
    
    function increment() public {
        value += 1;
    }
    
    function add(int amount) public {
        value += amount;
    }
}
```

### 🔨 Compile Your Contract

**For WASM contracts:**
```bash
# Compile the Rust contract to WASM
arthachain contract build

# This creates: target/wasm32-unknown-unknown/release/my_first_contract.wasm
```

**For Solidity contracts:**
```bash
# Compile the Solidity contract
arthachain contract build --type solidity

# This creates: build/contracts/Counter.json
```

### 🚀 Deploy to Testnet

```bash
# Deploy your contract
arthachain contract deploy

# You'll see:
```

**Example Output:**
```
🚀 Deploying Contract...
📄 Contract: my-first-contract.wasm
💰 Gas Limit: 1,000,000
💵 Gas Price: 0.000001 ARTHA

✅ Contract Deployed!
📋 Contract Address: artha1contract123abc456...
🔍 Transaction Hash: 0x9f8e7d6c5b4a...
🌐 Explorer: https://testnet.arthachain.online/address/artha1contract123abc456...

💾 Saved contract info to: .arthachain/contracts/my-first-contract.json
```

### 🎮 Interact With Your Contract

```bash
# Call the get_value function (read-only)
arthachain contract call artha1contract123abc456... get_value

# Output: 0

# Call the increment function (changes state)
arthachain contract call artha1contract123abc456... increment

# Check the value again
arthachain contract call artha1contract123abc456... get_value

# Output: 1

# Add a specific amount
arthachain contract call artha1contract123abc456... add 5

# Check final value
arthachain contract call artha1contract123abc456... get_value

# Output: 6
```

### 📊 Check Contract in Explorer

1. Go to [testnet.arthachain.online](https://testnet.arthachain.online)
2. Paste your contract address in the search box
3. See all the transactions and state changes!

🎉 **Amazing! You've deployed and interacted with your first smart contract!**

## 🎯 What You've Accomplished

In just 10 minutes, you:

✅ **Set up a complete ArthaChain development environment**  
✅ **Connected to the testnet**  
✅ **Created a secure wallet**  
✅ **Received free test tokens**  
✅ **Made your first transaction**  
✅ **Deployed a smart contract**  
✅ **Interacted with your contract**  

You're now officially a blockchain developer! 🎓

## 🚀 Next Steps

### 👶 **Just Getting Started?**
1. **[🎓 Learn Basic Concepts](./basic-concepts.md)** - Understand key blockchain terms
2. **[🎮 Build Your First dApp](./tutorials/first-dapp.md)** - Create a web app that uses your contract
3. **[💡 Explore Examples](./tutorials/)** - See what others have built

### 👨‍💻 **Ready to Build Seriously?**
1. **[⚙️ Set Up a Local Node](./node-setup.md)** - Run ArthaChain on your computer
2. **[📱 API Reference](./api-reference.md)** - Connect your apps to the blockchain
3. **[🤖 Advanced Contracts](./smart-contracts.md)** - Build complex smart contracts

### 🏢 **Building for Production?**
1. **[🔐 Security Guide](./security.md)** - Keep your apps safe
2. **[📊 Performance Optimization](./performance.md)** - Make your apps lightning fast
3. **[🏗️ Architecture Patterns](./architecture.md)** - Design scalable systems

## 🆘 Common Issues & Solutions

### ❌ **"Command not found: arthachain"**

**Solution:**
```bash
# Make sure npm global packages are in your PATH
echo $PATH | grep npm

# If not, add this to your shell profile:
export PATH=$PATH:$(npm prefix -g)/bin

# Reload your shell
source ~/.bashrc   # or ~/.zshrc
```

### ❌ **"Network connection failed"**

**Solutions:**
```bash
# Try different RPC endpoints
arthachain config set rpc-url https://rpc-backup.arthachain.online

# Check if you're behind a firewall
curl https://testnet.arthachain.online/api/health

# Reset configuration
arthachain config reset
arthachain network set testnet
```

### ❌ **"Faucet rate limited"**

**Solutions:**
1. **Wait 24 hours** for rate limit to reset
2. **Try different methods** (Discord, Telegram, Web)
3. **Ask in Discord** - community members often share tokens
4. **Use a different address** if you have multiple wallets

### ❌ **"Insufficient funds"**

**Check your balance:**
```bash
arthachain wallet balance

# If balance is 0:
arthachain faucet request

# If balance is low:
# Reduce transaction amount or get more from faucet
```

### ❌ **"Contract compilation failed"**

**For WASM contracts:**
```bash
# Make sure Rust is installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add wasm32-unknown-unknown

# Clean and rebuild
arthachain contract clean
arthachain contract build
```

**For Solidity contracts:**
```bash
# Make sure solc is installed
npm install -g solc

# Check contract syntax
arthachain contract verify
arthachain contract build
```

## 🌐 Useful Resources

### 📚 **Learning Resources**
- **[📖 Documentation](./README.md)** - Complete ArthaChain docs
- **[🎮 Interactive Tutorials](./tutorials/)** - Hands-on learning
- **[💡 Code Examples](https://github.com/arthachain/examples)** - Copy-paste code snippets
- **[📺 Video Tutorials](https://youtube.com/@arthachain)** - Visual learning

### 🛠️ **Developer Tools**
- **[🌐 Testnet Explorer](https://testnet.arthachain.online)** - View transactions and contracts
- **[💰 Faucet](https://faucet.arthachain.online)** - Get free test tokens
- **[🎮 Playground](https://playground.arthachain.online)** - Code in your browser
- **[📊 Network Status](https://status.arthachain.online)** - Check network health

### 💬 **Community Support**
- **[💬 Discord](https://discord.gg/arthachain)** - Live chat with developers
- **[📱 Telegram](https://t.me/arthachain)** - Mobile-friendly community
- **[🐙 GitHub](https://github.com/arthachain/blockchain)** - Source code and issues
- **[🐦 Twitter](https://twitter.com/ArthaChain)** - Latest updates and news

### 📧 **Direct Support**
- **👨‍💻 Developer Questions**: [developers@arthachain.com](mailto:developers@arthachain.com)
- **🏢 Enterprise Inquiries**: [enterprise@arthachain.com](mailto:enterprise@arthachain.com)
- **🚨 Security Issues**: [security@arthachain.com](mailto:security@arthachain.com)

## 🎉 Welcome to the ArthaChain Community!

You've just taken your first steps into the future of blockchain technology. ArthaChain's unique combination of:

- **⚡ Real benchmarked performance** (Measured with actual cryptographic operations)
- **🛡️ Quantum-proof security**
- **🧠 AI-powered protection**
- **📱 Mobile-first design**

...makes it the perfect platform for building the next generation of decentralized applications.

**Ready to build something amazing?** The entire ArthaChain ecosystem is here to support you!

---

**🎯 Next**: [🎓 Basic Concepts](./basic-concepts.md) →

**💬 Need Help?** Join our [Discord](https://discord.gg/arthachain) - the community is super friendly and helpful! 