# 🤖 Smart Contracts on ArthaChain

**Welcome to the future of smart contracts!** ArthaChain supports both **WebAssembly (WASM)** and **Solidity** contracts, giving you the best of both worlds.

## 🎯 What You'll Learn

- **🔤 What are smart contracts?** (Simple explanation)
- **⚔️ WASM vs Solidity** - Which to choose and when
- **🦀 WASM Development** - Using Rust for ultra-fast contracts
- **⚡ Solidity Development** - Ethereum-compatible contracts
- **🚀 Deployment & Testing** - Get your contracts running
- **🔧 Advanced Features** - Upgrades, standards, verification
- **💡 Real Examples** - Copy-paste working contracts

## 🤔 What Are Smart Contracts? (Simple Explanation)

Think of smart contracts like **magical vending machines** on the blockchain:

```
🏪 Traditional Vending Machine:
├── 💰 You put coins in
├── 🔘 You press a button  
├── 🍫 You get a snack
└── 💸 Keep your change

🤖 Smart Contract (Magical Vending Machine):
├── 💰 You send cryptocurrency
├── 📋 Contract executes automatically
├── 🎁 You get tokens/services/digital items
├── 🔄 Everything happens instantly
├── 🌍 Works from anywhere in the world
├── 🛡️ No one can cheat or stop it
└── 📖 Everyone can see the code
```

**Key Differences from Regular Programs:**
- **🌍 Global Access**: Anyone in the world can use them
- **🛡️ Unstoppable**: Once deployed, they run forever
- **💰 Handle Money**: They can receive, store, and send cryptocurrency
- **📖 Transparent**: All code is visible to everyone
- **🔒 Immutable**: Can't be changed (unless designed to be upgradeable)

## ⚔️ WASM vs Solidity: The Ultimate Comparison

ArthaChain supports **both** WASM and Solidity contracts. Here's when to use each:

### 🦀 **WASM (WebAssembly) Contracts**

**Best for:** High-performance applications, complex logic, new projects

```
🚀 WASM Advantages:
├── ⚡ Ultra-fast execution (near-native speed)
├── 🧠 Any programming language (Rust, C++, AssemblyScript)
├── 🔒 Memory safety (Rust prevents many bugs)
├── 📦 Smaller contract sizes
├── 🎯 Modern development experience
├── 🔧 Advanced type safety
└── 💪 Complex computations possible
```

**Supported Languages:**
- **🦀 Rust** (Recommended - best performance & safety)
- **🔷 AssemblyScript** (TypeScript-like, easier learning curve)
- **⚙️ C/C++** (Maximum performance, complex projects)
- **📜 More coming soon!**

### ⚡ **Solidity Contracts**

**Best for:** Ethereum migration, DeFi protocols, existing Solidity expertise

```
⚡ Solidity Advantages:
├── 🔄 Ethereum compatibility (migrate existing contracts)
├── 🛠️ Huge ecosystem (Hardhat, Remix, OpenZeppelin)
├── 👥 Large developer community
├── 📚 Lots of learning resources
├── 🧰 Mature tooling and libraries
├── 💰 DeFi standards (ERC20, ERC721, etc.)
└── 🔌 MetaMask integration
```

### 🎯 **Which Should You Choose?**

| Use Case | Recommended | Why |
|----------|------------|-----|
| **🎮 Games** | WASM (Rust) | Need high performance for game logic |
| **💰 DeFi** | Solidity | Ecosystem compatibility, established patterns |
| **🖼️ NFTs** | Either | Solidity for compatibility, WASM for innovation |
| **🏢 Enterprise** | WASM (Rust) | Security, performance, maintainability |
| **🧮 Complex Math** | WASM (Rust) | Better performance for heavy computations |
| **🚀 New Project** | WASM (Rust) | Future-proof, better development experience |
| **🔄 Ethereum Migration** | Solidity | Direct migration, less rewriting |

## 🦀 WASM Smart Contracts (Rust)

### 🚀 Quick Start

```bash
# Create a new WASM contract project
arthachain contract new my-awesome-contract --type wasm
cd my-awesome-contract

# The project structure:
my-awesome-contract/
├── Cargo.toml          # Rust dependencies
├── src/
│   └── lib.rs         # Your contract code
├── tests/             # Unit tests
└── README.md          # Documentation
```

### 📝 Basic Contract Template

```rust
// src/lib.rs - A simple counter contract
use arthachain_sdk::*;
use serde::{Deserialize, Serialize};

// Define the contract state
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Counter {
    pub value: i32,
    pub owner: Address,
}

// Contract implementation
#[contract_impl]
impl Counter {
    // Constructor - called when contract is deployed
    #[constructor]
    pub fn new() -> Self {
        Self {
            value: 0,
            owner: msg_sender(), // Get the address that deployed the contract
        }
    }

    // Public function - anyone can call
    #[public]
    pub fn increment(&mut self) {
        self.value += 1;
        
        // Emit an event
        emit_event("Incremented", &self.value);
    }

    // Public function with parameters
    #[public]
    pub fn add(&mut self, amount: i32) {
        // Only owner can add large amounts
        if amount > 100 {
            require(msg_sender() == self.owner, "Only owner can add large amounts");
        }
        
        self.value += amount;
        emit_event("Added", &amount);
    }

    // Read-only function (doesn't change state)
    #[view]
    pub fn get_value(&self) -> i32 {
        self.value
    }

    // Read-only function
    #[view]
    pub fn get_owner(&self) -> Address {
        self.owner
    }

    // Only owner can call this
    #[public]
    pub fn reset(&mut self) {
        require(msg_sender() == self.owner, "Only owner can reset");
        self.value = 0;
        emit_event("Reset", &0);
    }
}
```

### 🎮 Advanced Example: Gaming Contract

```rust
// A simple RPG character contract
use arthachain_sdk::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Character {
    pub name: String,
    pub level: u32,
    pub health: u32,
    pub mana: u32,
    pub experience: u32,
    pub equipment: HashMap<String, String>, // slot -> item_id
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GameContract {
    pub characters: HashMap<Address, Character>,
    pub items: HashMap<String, Item>,
    pub admins: Vec<Address>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Item {
    pub name: String,
    pub rarity: String,
    pub stats: HashMap<String, i32>, // stat_name -> bonus
}

#[contract_impl]
impl GameContract {
    #[constructor]
    pub fn new() -> Self {
        let mut contract = Self {
            characters: HashMap::new(),
            items: HashMap::new(),
            admins: vec![msg_sender()],
        };
        
        // Add some starter items
        contract.create_starter_items();
        contract
    }

    #[public]
    pub fn create_character(&mut self, name: String) {
        let player = msg_sender();
        require(!self.characters.contains_key(&player), "Character already exists");
        require(name.len() > 2 && name.len() < 20, "Name must be 3-19 characters");

        let character = Character {
            name,
            level: 1,
            health: 100,
            mana: 50,
            experience: 0,
            equipment: HashMap::new(),
        };

        self.characters.insert(player, character);
        emit_event("CharacterCreated", &player);
    }

    #[public]
    pub fn gain_experience(&mut self, amount: u32) {
        let player = msg_sender();
        let character = self.characters.get_mut(&player)
            .expect("Character not found");

        character.experience += amount;
        
        // Check for level up
        let required_exp = character.level * 100;
        if character.experience >= required_exp {
            character.level += 1;
            character.health += 20;
            character.mana += 10;
            
            emit_event("LevelUp", &character.level);
        }
        
        emit_event("ExperienceGained", &amount);
    }

    #[public]
    pub fn equip_item(&mut self, item_id: String, slot: String) {
        let player = msg_sender();
        require(self.items.contains_key(&item_id), "Item does not exist");
        
        let character = self.characters.get_mut(&player)
            .expect("Character not found");
            
        character.equipment.insert(slot, item_id.clone());
        emit_event("ItemEquipped", &item_id);
    }

    #[view]
    pub fn get_character(&self, player: Address) -> Option<Character> {
        self.characters.get(&player).cloned()
    }

    #[view]
    pub fn get_leaderboard(&self) -> Vec<(Address, u32)> {
        let mut players: Vec<_> = self.characters.iter()
            .map(|(addr, char)| (*addr, char.level))
            .collect();
        
        players.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by level descending
        players.truncate(10); // Top 10
        players
    }

    // Admin function to create items
    #[public]
    pub fn create_item(&mut self, id: String, name: String, rarity: String) {
        require(self.admins.contains(&msg_sender()), "Only admins can create items");
        
        let item = Item {
            name,
            rarity,
            stats: HashMap::new(),
        };
        
        self.items.insert(id, item);
        emit_event("ItemCreated", &id);
    }

    fn create_starter_items(&mut self) {
        // Create some basic items
        self.items.insert("sword_1".to_string(), Item {
            name: "Iron Sword".to_string(),
            rarity: "Common".to_string(),
            stats: [("attack".to_string(), 10)].into_iter().collect(),
        });
        
        self.items.insert("shield_1".to_string(), Item {
            name: "Wooden Shield".to_string(),
            rarity: "Common".to_string(),
            stats: [("defense".to_string(), 5)].into_iter().collect(),
        });
    }
}
```

### 🔧 Contract Development Workflow

```bash
# 1. Create project
arthachain contract new my-contract --type wasm
cd my-contract

# 2. Edit your contract
nano src/lib.rs

# 3. Add dependencies to Cargo.toml if needed
[dependencies]
arthachain-sdk = "1.0"
serde = { version = "1.0", features = ["derive"] }

# 4. Build the contract
arthachain contract build

# 5. Run unit tests
cargo test

# 6. Deploy to testnet
arthachain contract deploy --network testnet

# 7. Interact with deployed contract
arthachain contract call <contract-address> increment
arthachain contract call <contract-address> get_value
```

### 🧪 Unit Testing

```rust
// tests/contract_tests.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_increment() {
        let mut counter = Counter::new();
        assert_eq!(counter.get_value(), 0);
        
        counter.increment();
        assert_eq!(counter.get_value(), 1);
        
        counter.add(5);
        assert_eq!(counter.get_value(), 6);
    }

    #[test]
    fn test_character_creation() {
        let mut game = GameContract::new();
        
        // Mock the message sender
        set_msg_sender(Address::from("player1"));
        
        game.create_character("Hero".to_string());
        let character = game.get_character(Address::from("player1")).unwrap();
        
        assert_eq!(character.name, "Hero");
        assert_eq!(character.level, 1);
        assert_eq!(character.health, 100);
    }
}
```

## ⚡ Solidity Smart Contracts

### 🚀 Quick Start

```bash
# Create a new Solidity contract project
arthachain contract new my-token --type solidity
cd my-token

# Project structure:
my-token/
├── contracts/         # Solidity source files
│   └── MyToken.sol
├── scripts/          # Deployment scripts
│   └── deploy.js
├── test/            # JavaScript tests
│   └── MyToken.js
├── hardhat.config.js # Hardhat configuration
└── package.json     # Node.js dependencies
```

### 📝 Basic ERC20 Token Contract

```solidity
// contracts/MyToken.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract MyToken is ERC20, Ownable, Pausable {
    uint256 public constant MAX_SUPPLY = 1000000 * 10**18; // 1 million tokens
    uint256 public mintPrice = 0.001 ether; // Price per token in ARTHA
    
    mapping(address => bool) public whitelist;
    mapping(address => uint256) public purchaseCount;
    
    event TokensPurchased(address buyer, uint256 amount);
    event PriceUpdated(uint256 newPrice);
    event WhitelistUpdated(address user, bool status);

    constructor() ERC20("MyToken", "MTK") {
        // Mint 100,000 tokens to the deployer
        _mint(msg.sender, 100000 * 10**18);
    }

    /**
     * @dev Purchase tokens by sending ARTHA
     */
    function buyTokens(uint256 tokenAmount) external payable whenNotPaused {
        require(tokenAmount > 0, "Amount must be greater than 0");
        require(totalSupply() + tokenAmount <= MAX_SUPPLY, "Would exceed max supply");
        
        uint256 cost = tokenAmount * mintPrice / 10**18;
        require(msg.value >= cost, "Insufficient ARTHA sent");
        
        // Whitelist members get 10% discount
        if (whitelist[msg.sender]) {
            cost = cost * 90 / 100;
        }
        
        // Limit purchases for non-whitelisted users
        if (!whitelist[msg.sender]) {
            require(purchaseCount[msg.sender] + tokenAmount <= 1000 * 10**18, "Purchase limit exceeded");
        }
        
        _mint(msg.sender, tokenAmount);
        purchaseCount[msg.sender] += tokenAmount;
        
        // Refund excess ARTHA
        if (msg.value > cost) {
            payable(msg.sender).transfer(msg.value - cost);
        }
        
        emit TokensPurchased(msg.sender, tokenAmount);
    }

    /**
     * @dev Add or remove users from whitelist (only owner)
     */
    function updateWhitelist(address user, bool status) external onlyOwner {
        whitelist[user] = status;
        emit WhitelistUpdated(user, status);
    }

    /**
     * @dev Update mint price (only owner)
     */
    function updateMintPrice(uint256 newPrice) external onlyOwner {
        require(newPrice > 0, "Price must be greater than 0");
        mintPrice = newPrice;
        emit PriceUpdated(newPrice);
    }

    /**
     * @dev Withdraw contract balance (only owner)
     */
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        payable(owner()).transfer(balance);
    }

    /**
     * @dev Pause/unpause the contract (only owner)
     */
    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Burn tokens from your balance
     */
    function burn(uint256 amount) external {
        _burn(msg.sender, amount);
    }

    /**
     * @dev Get token purchase cost
     */
    function getPurchaseCost(uint256 tokenAmount) external view returns (uint256) {
        uint256 cost = tokenAmount * mintPrice / 10**18;
        if (whitelist[msg.sender]) {
            cost = cost * 90 / 100; // 10% discount for whitelist
        }
        return cost;
    }
}
```

### 🎮 Advanced Example: NFT Marketplace

```solidity
// contracts/NFTMarketplace.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract NFTMarketplace is ERC721URIStorage, Ownable, ReentrancyGuard {
    using Counters for Counters.Counter;
    
    Counters.Counter private _tokenIds;
    Counters.Counter private _itemsSold;

    uint256 public listingPrice = 0.025 ether; // Fee to list an NFT
    
    struct MarketItem {
        uint256 tokenId;
        address payable seller;
        address payable owner;
        uint256 price;
        bool sold;
        uint256 category; // 0: Art, 1: Gaming, 2: Music, 3: Collectibles
    }

    mapping(uint256 => MarketItem) private idToMarketItem;
    mapping(uint256 => bool) public tokenExists;
    mapping(address => uint256[]) public ownerTokens;

    event MarketItemCreated(
        uint256 indexed tokenId,
        address seller,
        address owner,
        uint256 price,
        bool sold
    );

    event MarketItemSold(
        uint256 indexed tokenId,
        address seller,
        address buyer,
        uint256 price
    );

    constructor() ERC721("ArthaChain NFT", "ANFT") {}

    /**
     * @dev Create a new NFT and list it on the marketplace
     */
    function createToken(
        string memory tokenURI,
        uint256 price,
        uint256 category
    ) public payable returns (uint256) {
        require(msg.value == listingPrice, "Must pay listing fee");
        require(price > 0, "Price must be greater than 0");
        require(category <= 3, "Invalid category");

        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();

        _mint(msg.sender, newTokenId);
        _setTokenURI(newTokenId, tokenURI);
        
        createMarketItem(newTokenId, price, category);
        
        tokenExists[newTokenId] = true;
        
        return newTokenId;
    }

    function createMarketItem(uint256 tokenId, uint256 price, uint256 category) private {
        idToMarketItem[tokenId] = MarketItem(
            tokenId,
            payable(msg.sender),
            payable(address(this)), // Marketplace owns it until sold
            price,
            false,
            category
        );

        _transfer(msg.sender, address(this), tokenId);

        emit MarketItemCreated(
            tokenId,
            msg.sender,
            address(this),
            price,
            false
        );
    }

    /**
     * @dev Purchase an NFT from the marketplace
     */
    function createMarketSale(uint256 tokenId) public payable nonReentrant {
        uint256 price = idToMarketItem[tokenId].price;
        address seller = idToMarketItem[tokenId].seller;
        
        require(msg.value == price, "Must pay the asking price");
        require(!idToMarketItem[tokenId].sold, "Item already sold");

        idToMarketItem[tokenId].owner = payable(msg.sender);
        idToMarketItem[tokenId].sold = true;
        _itemsSold.increment();

        _transfer(address(this), msg.sender, tokenId);
        
        // Add to buyer's token list
        ownerTokens[msg.sender].push(tokenId);
        
        // Remove from seller's token list
        _removeTokenFromOwner(seller, tokenId);

        // Pay seller (marketplace takes 2.5% fee)
        uint256 marketplaceFee = price * 25 / 1000; // 2.5%
        uint256 sellerAmount = price - marketplaceFee;
        
        payable(seller).transfer(sellerAmount);
        payable(owner()).transfer(marketplaceFee);

        emit MarketItemSold(tokenId, seller, msg.sender, price);
    }

    /**
     * @dev Resell an NFT you own
     */
    function resellToken(uint256 tokenId, uint256 price) public payable {
        require(idToMarketItem[tokenId].owner == msg.sender, "Only owner can resell");
        require(msg.value == listingPrice, "Must pay listing fee");
        require(price > 0, "Price must be greater than 0");

        idToMarketItem[tokenId].sold = false;
        idToMarketItem[tokenId].price = price;
        idToMarketItem[tokenId].seller = payable(msg.sender);
        idToMarketItem[tokenId].owner = payable(address(this));
        
        _itemsSold.decrement();

        _transfer(msg.sender, address(this), tokenId);
    }

    /**
     * @dev Get all unsold market items
     */
    function fetchMarketItems() public view returns (MarketItem[] memory) {
        uint256 itemCount = _tokenIds.current();
        uint256 unsoldItemCount = _tokenIds.current() - _itemsSold.current();
        uint256 currentIndex = 0;

        MarketItem[] memory items = new MarketItem[](unsoldItemCount);
        
        for (uint256 i = 0; i < itemCount; i++) {
            if (idToMarketItem[i + 1].owner == address(this)) {
                uint256 currentId = i + 1;
                MarketItem storage currentItem = idToMarketItem[currentId];
                items[currentIndex] = currentItem;
                currentIndex += 1;
            }
        }
        
        return items;
    }

    /**
     * @dev Get NFTs owned by the caller
     */
    function fetchMyNFTs() public view returns (MarketItem[] memory) {
        uint256 totalItemCount = _tokenIds.current();
        uint256 itemCount = 0;
        uint256 currentIndex = 0;

        // Count items owned by caller
        for (uint256 i = 0; i < totalItemCount; i++) {
            if (idToMarketItem[i + 1].owner == msg.sender) {
                itemCount += 1;
            }
        }

        MarketItem[] memory items = new MarketItem[](itemCount);
        
        for (uint256 i = 0; i < totalItemCount; i++) {
            if (idToMarketItem[i + 1].owner == msg.sender) {
                uint256 currentId = i + 1;
                MarketItem storage currentItem = idToMarketItem[currentId];
                items[currentIndex] = currentItem;
                currentIndex += 1;
            }
        }
        
        return items;
    }

    /**
     * @dev Get NFTs listed by the caller
     */
    function fetchItemsListed() public view returns (MarketItem[] memory) {
        uint256 totalItemCount = _tokenIds.current();
        uint256 itemCount = 0;
        uint256 currentIndex = 0;

        for (uint256 i = 0; i < totalItemCount; i++) {
            if (idToMarketItem[i + 1].seller == msg.sender) {
                itemCount += 1;
            }
        }

        MarketItem[] memory items = new MarketItem[](itemCount);
        
        for (uint256 i = 0; i < totalItemCount; i++) {
            if (idToMarketItem[i + 1].seller == msg.sender) {
                uint256 currentId = i + 1;
                MarketItem storage currentItem = idToMarketItem[currentId];
                items[currentIndex] = currentItem;
                currentIndex += 1;
            }
        }
        
        return items;
    }

    function _removeTokenFromOwner(address owner, uint256 tokenId) private {
        uint256[] storage tokens = ownerTokens[owner];
        for (uint256 i = 0; i < tokens.length; i++) {
            if (tokens[i] == tokenId) {
                tokens[i] = tokens[tokens.length - 1];
                tokens.pop();
                break;
            }
        }
    }

    /**
     * @dev Update listing price (only owner)
     */
    function updateListingPrice(uint256 _listingPrice) public onlyOwner {
        listingPrice = _listingPrice;
    }

    /**
     * @dev Get listing price
     */
    function getListingPrice() public view returns (uint256) {
        return listingPrice;
    }
}
```

### 🔧 Solidity Development Workflow

```bash
# 1. Create project
arthachain contract new my-nft-marketplace --type solidity
cd my-nft-marketplace

# 2. Install dependencies
npm install

# 3. Configure Hardhat for ArthaChain
# Edit hardhat.config.js:
```

```javascript
// hardhat.config.js
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    arthachain_testnet: {
      url: "https://testnet.arthachain.online/rpc",
      chainId: 1337,
      accounts: [process.env.PRIVATE_KEY] // Your private key
    },
    arthachain_mainnet: {
      url: "https://api.arthachain.com/rpc", 
      chainId: 1338,
      accounts: [process.env.PRIVATE_KEY]
    }
  },
  etherscan: {
    apiKey: {
      arthachain_testnet: "your-api-key",
      arthachain_mainnet: "your-api-key"
    }
  }
};
```

```bash
# 4. Compile contracts
npx hardhat compile

# 5. Run tests
npx hardhat test

# 6. Deploy to testnet
npx hardhat run scripts/deploy.js --network arthachain_testnet

# 7. Verify contract (optional)
npx hardhat verify --network arthachain_testnet <contract-address>
```

### 🧪 JavaScript Testing

```javascript
// test/NFTMarketplace.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("NFTMarketplace", function () {
  let marketplace, owner, seller, buyer;
  
  beforeEach(async function () {
    [owner, seller, buyer] = await ethers.getSigners();
    
    const NFTMarketplace = await ethers.getContractFactory("NFTMarketplace");
    marketplace = await NFTMarketplace.deploy();
    await marketplace.deployed();
  });

  it("Should create and sell NFTs", async function () {
    const listingPrice = await marketplace.getListingPrice();
    const auctionPrice = ethers.utils.parseUnits("1", "ether");

    // Create NFT
    await marketplace.connect(seller).createToken(
      "https://example.com/token/1",
      auctionPrice,
      0, // Art category
      { value: listingPrice }
    );

    // Buy NFT
    await marketplace.connect(buyer).createMarketSale(1, {
      value: auctionPrice
    });

    // Check ownership
    const owner = await marketplace.ownerOf(1);
    expect(owner).to.equal(buyer.address);
  });

  it("Should return correct market items", async function () {
    const listingPrice = await marketplace.getListingPrice();
    const auctionPrice = ethers.utils.parseUnits("1", "ether");

    // Create two NFTs
    await marketplace.connect(seller).createToken(
      "https://example.com/token/1",
      auctionPrice,
      0,
      { value: listingPrice }
    );
    
    await marketplace.connect(seller).createToken(
      "https://example.com/token/2", 
      auctionPrice,
      1,
      { value: listingPrice }
    );

    // Get market items
    const items = await marketplace.fetchMarketItems();
    expect(items.length).to.equal(2);
  });
});
```

## 🚀 Deployment & Management

### 🌐 Deploy to Different Networks

```bash
# Deploy to testnet
arthachain contract deploy --network testnet

# Deploy to mainnet (be careful!)
arthachain contract deploy --network mainnet

# Deploy with specific gas settings
arthachain contract deploy --gas-limit 1000000 --gas-price 10

# Deploy and verify
arthachain contract deploy --verify --network testnet
```

### 📊 Contract Interaction

```bash
# Call read-only functions (no gas cost)
arthachain contract call <address> get_value
arthachain contract call <address> balanceOf 0x123...

# Call state-changing functions (costs gas)
arthachain contract call <address> increment --gas 50000
arthachain contract call <address> transfer 0x123... 100 --value 0.01

# Listen to events
arthachain contract events <address> --filter "Transfer"

# Get contract info
arthachain contract info <address>
```

### 🔄 Contract Upgrades

**WASM Upgradeable Pattern:**
```rust
// Upgradeable contract pattern
use arthachain_sdk::*;

#[derive(Serialize, Deserialize)]
pub struct UpgradeableContract {
    pub version: u32,
    pub admin: Address,
    pub logic_contract: Address,
    pub data: ContractData,
}

#[contract_impl]
impl UpgradeableContract {
    #[public]
    pub fn upgrade(&mut self, new_logic_address: Address) {
        require(msg_sender() == self.admin, "Only admin can upgrade");
        self.logic_contract = new_logic_address;
        self.version += 1;
        emit_event("ContractUpgraded", &self.version);
    }

    #[public]
    pub fn delegate_call(&self, function_name: String, args: Vec<u8>) -> Vec<u8> {
        // Delegate call to logic contract
        delegate_call(self.logic_contract, function_name, args)
    }
}
```

**Solidity Proxy Pattern:**
```solidity
// Using OpenZeppelin's upgradeable contracts
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

contract MyUpgradeableContract is Initializable, OwnableUpgradeable {
    uint256 public value;

    function initialize() public initializer {
        __Ownable_init();
        value = 0;
    }

    function setValue(uint256 _value) public onlyOwner {
        value = _value;
    }

    // This function can be added in v2
    function getValue() public view returns (uint256) {
        return value;
    }
}
```

## 📋 Token Standards & Compliance

### 🪙 ERC20 Token Standard

```solidity
// Standard ERC20 implementation
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract StandardToken is ERC20 {
    constructor(uint256 initialSupply) ERC20("MyToken", "MTK") {
        _mint(msg.sender, initialSupply);
    }
}
```

### 🖼️ ERC721 NFT Standard

```solidity
// Standard NFT implementation
import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";

contract StandardNFT is ERC721URIStorage {
    uint256 private _tokenIds;

    constructor() ERC721("MyNFT", "MNFT") {}

    function mint(address to, string memory tokenURI) public returns (uint256) {
        _tokenIds++;
        _safeMint(to, _tokenIds);
        _setTokenURI(_tokenIds, tokenURI);
        return _tokenIds;
    }
}
```

### 🎮 ERC1155 Multi-Token Standard

```solidity
// Gaming tokens (weapons, items, etc.)
import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";

contract GameItems is ERC1155 {
    uint256 public constant SWORD = 0;
    uint256 public constant SHIELD = 1;
    uint256 public constant POTION = 2;

    constructor() ERC1155("https://game.example/api/item/{id}.json") {
        _mint(msg.sender, SWORD, 10**18, "");
        _mint(msg.sender, SHIELD, 10**27, "");
        _mint(msg.sender, POTION, 10**9, "");
    }
}
```

## 🔐 Security Best Practices

### 🛡️ Common Vulnerabilities & Fixes

**1. Reentrancy Attacks**
```solidity
// ❌ Vulnerable code
function withdraw() public {
    uint amount = balances[msg.sender];
    msg.sender.call{value: amount}(""); // External call before state change
    balances[msg.sender] = 0; // State change after external call
}

// ✅ Secure code
function withdraw() public nonReentrant {
    uint amount = balances[msg.sender];
    balances[msg.sender] = 0; // State change before external call
    msg.sender.call{value: amount}("");
}
```

**2. Integer Overflow/Underflow**
```solidity
// ✅ Use SafeMath or Solidity 0.8+
pragma solidity ^0.8.0; // Built-in overflow protection

// Or explicitly use SafeMath for older versions
using SafeMath for uint256;

function safeAdd(uint256 a, uint256 b) public pure returns (uint256) {
    return a.add(b); // Will revert on overflow
}
```

**3. Access Control**
```rust
// WASM example
#[public]
pub fn admin_only_function(&mut self) {
    require(msg_sender() == self.admin, "Only admin can call this");
    // ... rest of function
}
```

### 🧪 Testing & Auditing

```bash
# Run security analysis
arthachain contract audit my-contract.wasm
arthachain contract test --security-checks

# Generate test coverage report  
arthachain contract coverage

# Run formal verification (for critical contracts)
arthachain contract verify --formal my-contract.wasm
```

## 💡 Performance Tips

### ⚡ Gas Optimization

**WASM Optimization:**
```rust
// Use references instead of cloning large data
#[view]
pub fn get_large_data(&self) -> &Vec<BigStruct> {
    &self.data // Return reference, not clone
}

// Batch operations
#[public]
pub fn batch_update(&mut self, updates: Vec<(u32, u32)>) {
    for (id, value) in updates {
        self.data.insert(id, value);
    }
    // Single event for all updates
    emit_event("BatchUpdate", &updates.len());
}
```

**Solidity Optimization:**
```solidity
// Pack structs efficiently
struct User {
    uint128 balance;    // 16 bytes
    uint128 timestamp;  // 16 bytes = 32 bytes total (1 storage slot)
}

// Use events instead of storage for logs
event UserAction(address user, uint256 amount, uint256 timestamp);

function recordAction(uint256 amount) external {
    emit UserAction(msg.sender, amount, block.timestamp);
    // Don't store in contract storage unless necessary
}
```

### 📊 Storage Optimization

```rust
// Use efficient data structures
use std::collections::BTreeMap; // More gas-efficient for ordered data
use std::collections::HashMap;  // Better for random access

// Minimize storage writes
#[public]
pub fn efficient_update(&mut self, new_value: u32) {
    if self.value != new_value { // Only write if changed
        self.value = new_value;
        emit_event("ValueChanged", &new_value);
    }
}
```

## 🎯 What's Next?

Congratulations! You now know how to build smart contracts on ArthaChain. Here's what to explore next:

### 👶 **Beginner Next Steps**
1. **[🎮 Build a dApp Tutorial](./tutorials/first-dapp.md)** - Create a web interface for your contract
2. **[💰 DeFi Examples](./tutorials/defi-basics.md)** - Build decentralized finance applications
3. **[🎨 NFT Projects](./tutorials/nft-marketplace.md)** - Create and trade NFTs

### 👨‍💻 **Advanced Topics**
1. **[⚛️ Quantum Features](./quantum-resistance.md)** - Use quantum-resistant cryptography
2. **[🧠 AI Integration](./ai-features.md)** - Integrate with ArthaChain's AI engine
3. **[🌐 Cross-Shard](./cross-shard.md)** - Build applications that use multiple shards

### 🏢 **Production Ready**
1. **[🔐 Security Guide](./security.md)** - Secure your contracts for production
2. **[📊 Monitoring](./monitoring.md)** - Monitor your deployed contracts
3. **[🚀 Scaling](./scaling.md)** - Optimize for high-traffic applications

## 🆘 Need Help?

### 💬 **Community Support**
- **[💬 Discord](https://discord.gg/arthachain)** - Live chat with developers
- **[📱 Telegram](https://t.me/arthachain_dev)** - Developer support group
- **[🐙 GitHub](https://github.com/arthachain/blockchain)** - Source code and issues

### 📧 **Direct Support**
- **🤖 Smart Contract Questions**: [contracts@arthachain.com](mailto:contracts@arthachain.com)
- **🔐 Security Concerns**: [security@arthachain.com](mailto:security@arthachain.com)
- **🏢 Enterprise Development**: [enterprise@arthachain.com](mailto:enterprise@arthachain.com)

---

**🎯 Next**: [📱 API Reference](./api-reference.md) →

**💬 Questions?** Join our [Discord](https://discord.gg/arthachain) - we love helping developers build amazing things! 