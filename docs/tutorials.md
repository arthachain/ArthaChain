# 💡 ArthaChain Tutorials & Examples

**Learn by building! Step-by-step tutorials from your first transaction to advanced dApps.** Perfect for developers at every level.

## 🎯 What You'll Find Here

- **👶 Beginner Tutorials** - Your first steps with ArthaChain
- **👨‍💻 Intermediate Projects** - Build real applications
- **🚀 Advanced Examples** - Enterprise-grade development
- **🎮 Gaming dApps** - NFTs, tokens, and game mechanics
- **💰 DeFi Projects** - Decentralized finance applications
- **🤖 AI Integration** - Use ArthaChain's AI features
- **📱 Mobile Development** - Build for mobile validators

## 👶 Beginner Tutorials

Perfect if you're new to blockchain development.

### 🪙 **Tutorial 1: Your First Token (30 minutes)**

Let's create a simple token that you can send to friends!

**What you'll build:** A basic ERC20 token with your name

**Prerequisites:** Completed [Getting Started](./getting-started.md)

#### Step 1: Create the Token Contract

```solidity
// contracts/MyFirstToken.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract MyFirstToken is ERC20 {
    constructor() ERC20("My First Token", "MFT") {
        // Mint 1 million tokens to yourself
        _mint(msg.sender, 1000000 * 10**18);
    }
    
    // Allow anyone to get free tokens (for learning)
    function getFreeTokens() public {
        require(balanceOf(msg.sender) < 1000 * 10**18, "You already have enough tokens");
        _mint(msg.sender, 100 * 10**18); // Give 100 tokens
    }
}
```

#### Step 2: Deploy Your Token

```bash
# Create new project
arthachain contract new my-first-token --type solidity
cd my-first-token

# Copy the contract code above to contracts/MyFirstToken.sol

# Compile
npx hardhat compile

# Deploy to testnet
npx hardhat run scripts/deploy.js --network arthachain_testnet
```

#### Step 3: Interact with Your Token

```javascript
// scripts/interact.js
const { ethers } = require("hardhat");

async function main() {
    // Get the deployed contract
    const contractAddress = "YOUR_TOKEN_ADDRESS_HERE";
    const MyFirstToken = await ethers.getContractFactory("MyFirstToken");
    const token = MyFirstToken.attach(contractAddress);
    
    // Check your balance
    const [deployer] = await ethers.getSigners();
    const balance = await token.balanceOf(deployer.address);
    console.log(`Your balance: ${ethers.utils.formatEther(balance)} MFT`);
    
    // Get some free tokens
    const freeTx = await token.getFreeTokens();
    await freeTx.wait();
    console.log("Got free tokens!");
    
    // Check balance again
    const newBalance = await token.balanceOf(deployer.address);
    console.log(`New balance: ${ethers.utils.formatEther(newBalance)} MFT`);
}

main().catch(console.error);
```

**🎉 Congratulations!** You've created your first token!

### 🖼️ **Tutorial 2: Simple NFT Collection (45 minutes)**

Create unique digital collectibles!

**What you'll build:** A collection of personalized NFTs

#### Step 1: NFT Contract

```solidity
// contracts/MyNFTCollection.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract MyNFTCollection is ERC721 {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;
    
    // Store metadata for each NFT
    mapping(uint256 => string) public tokenNames;
    mapping(uint256 => string) public tokenDescriptions;
    mapping(uint256 => string) public tokenColors;
    
    constructor() ERC721("My NFT Collection", "MNC") {}
    
    function mintNFT(
        address to,
        string memory name,
        string memory description,
        string memory color
    ) public returns (uint256) {
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _mint(to, newTokenId);
        
        // Store metadata
        tokenNames[newTokenId] = name;
        tokenDescriptions[newTokenId] = description;
        tokenColors[newTokenId] = color;
        
        return newTokenId;
    }
    
    function getNFTData(uint256 tokenId) public view returns (
        string memory name,
        string memory description,
        string memory color,
        address owner
    ) {
        require(_exists(tokenId), "NFT does not exist");
        
        return (
            tokenNames[tokenId],
            tokenDescriptions[tokenId],
            tokenColors[tokenId],
            ownerOf(tokenId)
        );
    }
    
    function getTotalSupply() public view returns (uint256) {
        return _tokenIds.current();
    }
}
```

#### Step 2: Mint Your First NFT

```javascript
// scripts/mint-nft.js
async function main() {
    const MyNFTCollection = await ethers.getContractFactory("MyNFTCollection");
    const nft = MyNFTCollection.attach("YOUR_CONTRACT_ADDRESS");
    
    const [deployer] = await ethers.getSigners();
    
    // Mint your first NFT
    const mintTx = await nft.mintNFT(
        deployer.address,
        "My First NFT",
        "This is my very first NFT on ArthaChain!",
        "blue"
    );
    
    const receipt = await mintTx.wait();
    console.log("NFT minted! Transaction hash:", receipt.transactionHash);
    
    // Get the NFT data
    const totalSupply = await nft.getTotalSupply();
    const nftData = await nft.getNFTData(totalSupply);
    
    console.log("NFT Data:", {
        id: totalSupply.toString(),
        name: nftData.name,
        description: nftData.description,
        color: nftData.color,
        owner: nftData.owner
    });
}

main().catch(console.error);
```

### 🎮 **Tutorial 3: Simple Game (60 minutes)**

Build a basic blockchain game!

**What you'll build:** A simple rock-paper-scissors game with betting

```solidity
// contracts/RockPaperScissors.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract RockPaperScissors {
    enum Choice { None, Rock, Paper, Scissors }
    enum GameState { Waiting, Playing, Finished }
    
    struct Game {
        address player1;
        address player2;
        uint256 betAmount;
        bytes32 player1Choice; // Hidden choice
        Choice player2Choice;
        GameState state;
        address winner;
    }
    
    mapping(uint256 => Game) public games;
    uint256 public gameCounter;
    
    event GameCreated(uint256 gameId, address player1, uint256 betAmount);
    event GameJoined(uint256 gameId, address player2);
    event GameFinished(uint256 gameId, address winner, uint256 winnings);
    
    function createGame(bytes32 hashedChoice) external payable {
        require(msg.value > 0, "Bet amount must be greater than 0");
        
        gameCounter++;
        games[gameCounter] = Game({
            player1: msg.sender,
            player2: address(0),
            betAmount: msg.value,
            player1Choice: hashedChoice,
            player2Choice: Choice.None,
            state: GameState.Waiting,
            winner: address(0)
        });
        
        emit GameCreated(gameCounter, msg.sender, msg.value);
    }
    
    function joinGame(uint256 gameId, Choice choice) external payable {
        Game storage game = games[gameId];
        require(game.state == GameState.Waiting, "Game not available");
        require(msg.sender != game.player1, "Cannot play against yourself");
        require(msg.value == game.betAmount, "Incorrect bet amount");
        require(choice != Choice.None, "Must choose rock, paper, or scissors");
        
        game.player2 = msg.sender;
        game.player2Choice = choice;
        game.state = GameState.Playing;
        
        emit GameJoined(gameId, msg.sender);
    }
    
    function revealAndFinish(uint256 gameId, Choice choice, uint256 nonce) external {
        Game storage game = games[gameId];
        require(game.state == GameState.Playing, "Game not in playing state");
        require(msg.sender == game.player1, "Only player1 can reveal");
        
        // Verify the choice matches the hash
        bytes32 hash = keccak256(abi.encodePacked(choice, nonce, msg.sender));
        require(hash == game.player1Choice, "Invalid choice or nonce");
        
        // Determine winner
        address winner = determineWinner(choice, game.player2Choice);
        game.winner = winner;
        game.state = GameState.Finished;
        
        // Pay out winnings
        if (winner != address(0)) {
            uint256 winnings = game.betAmount * 2;
            payable(winner).transfer(winnings);
            emit GameFinished(gameId, winner, winnings);
        } else {
            // Tie - refund both players
            payable(game.player1).transfer(game.betAmount);
            payable(game.player2).transfer(game.betAmount);
            emit GameFinished(gameId, address(0), 0);
        }
    }
    
    function determineWinner(Choice choice1, Choice choice2) internal pure returns (address) {
        if (choice1 == choice2) return address(0); // Tie
        
        if ((choice1 == Choice.Rock && choice2 == Choice.Scissors) ||
            (choice1 == Choice.Paper && choice2 == Choice.Rock) ||
            (choice1 == Choice.Scissors && choice2 == Choice.Paper)) {
            return address(1); // Player 1 wins (placeholder)
        } else {
            return address(2); // Player 2 wins (placeholder)
        }
    }
    
    // Helper function to create hash
    function createHash(Choice choice, uint256 nonce) external view returns (bytes32) {
        return keccak256(abi.encodePacked(choice, nonce, msg.sender));
    }
}
```

## 👨‍💻 Intermediate Projects

Ready to build more complex applications.

### 💰 **Project 1: DeFi Yield Farm (2 hours)**

Build a yield farming protocol where users can stake tokens and earn rewards!

#### Core Contract

```solidity
// contracts/YieldFarm.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract YieldFarm is ReentrancyGuard, Ownable {
    IERC20 public stakingToken;
    IERC20 public rewardToken;
    
    uint256 public rewardRate = 100; // 100 tokens per second
    uint256 public lastUpdateTime;
    uint256 public rewardPerTokenStored;
    
    mapping(address => uint256) public userRewardPerTokenPaid;
    mapping(address => uint256) public rewards;
    mapping(address => uint256) private _balances;
    
    uint256 private _totalSupply;
    
    event Staked(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);
    event RewardPaid(address indexed user, uint256 reward);
    
    constructor(address _stakingToken, address _rewardToken) {
        stakingToken = IERC20(_stakingToken);
        rewardToken = IERC20(_rewardToken);
    }
    
    modifier updateReward(address account) {
        rewardPerTokenStored = rewardPerToken();
        lastUpdateTime = block.timestamp;
        
        if (account != address(0)) {
            rewards[account] = earned(account);
            userRewardPerTokenPaid[account] = rewardPerTokenStored;
        }
        _;
    }
    
    function rewardPerToken() public view returns (uint256) {
        if (_totalSupply == 0) {
            return rewardPerTokenStored;
        }
        return rewardPerTokenStored + 
            (((block.timestamp - lastUpdateTime) * rewardRate * 1e18) / _totalSupply);
    }
    
    function earned(address account) public view returns (uint256) {
        return ((_balances[account] * 
            (rewardPerToken() - userRewardPerTokenPaid[account])) / 1e18) + 
            rewards[account];
    }
    
    function stake(uint256 amount) external nonReentrant updateReward(msg.sender) {
        require(amount > 0, "Cannot stake 0");
        _totalSupply += amount;
        _balances[msg.sender] += amount;
        stakingToken.transferFrom(msg.sender, address(this), amount);
        emit Staked(msg.sender, amount);
    }
    
    function withdraw(uint256 amount) external nonReentrant updateReward(msg.sender) {
        require(amount > 0, "Cannot withdraw 0");
        require(_balances[msg.sender] >= amount, "Insufficient balance");
        _totalSupply -= amount;
        _balances[msg.sender] -= amount;
        stakingToken.transfer(msg.sender, amount);
        emit Withdrawn(msg.sender, amount);
    }
    
    function getReward() external nonReentrant updateReward(msg.sender) {
        uint256 reward = rewards[msg.sender];
        if (reward > 0) {
            rewards[msg.sender] = 0;
            rewardToken.transfer(msg.sender, reward);
            emit RewardPaid(msg.sender, reward);
        }
    }
    
    function exit() external {
        withdraw(_balances[msg.sender]);
        getReward();
    }
    
    // View functions
    function balanceOf(address account) external view returns (uint256) {
        return _balances[account];
    }
    
    function totalSupply() external view returns (uint256) {
        return _totalSupply;
    }
}
```

#### Frontend Integration

```javascript
// Frontend interaction with yield farm
class YieldFarmDApp {
    constructor(provider, contractAddress, abi) {
        this.contract = new ethers.Contract(contractAddress, abi, provider);
        this.signer = provider.getSigner();
    }
    
    async stakeTokens(amount) {
        try {
            const contractWithSigner = this.contract.connect(this.signer);
            const tx = await contractWithSigner.stake(
                ethers.utils.parseEther(amount.toString())
            );
            
            console.log('Staking transaction sent:', tx.hash);
            await tx.wait();
            console.log('Tokens staked successfully!');
            
            this.updateUI();
        } catch (error) {
            console.error('Error staking tokens:', error);
        }
    }
    
    async withdrawTokens(amount) {
        try {
            const contractWithSigner = this.contract.connect(this.signer);
            const tx = await contractWithSigner.withdraw(
                ethers.utils.parseEther(amount.toString())
            );
            
            await tx.wait();
            console.log('Tokens withdrawn successfully!');
            this.updateUI();
        } catch (error) {
            console.error('Error withdrawing tokens:', error);
        }
    }
    
    async claimRewards() {
        try {
            const contractWithSigner = this.contract.connect(this.signer);
            const tx = await contractWithSigner.getReward();
            
            await tx.wait();
            console.log('Rewards claimed successfully!');
            this.updateUI();
        } catch (error) {
            console.error('Error claiming rewards:', error);
        }
    }
    
    async getUserStats(userAddress) {
        const [stakedBalance, earnedRewards] = await Promise.all([
            this.contract.balanceOf(userAddress),
            this.contract.earned(userAddress)
        ]);
        
        return {
            staked: ethers.utils.formatEther(stakedBalance),
            earned: ethers.utils.formatEther(earnedRewards)
        };
    }
    
    async updateUI() {
        const user = await this.signer.getAddress();
        const stats = await this.getUserStats(user);
        
        document.getElementById('staked-amount').textContent = stats.staked;
        document.getElementById('earned-rewards').textContent = stats.earned;
    }
}
```

### 🎮 **Project 2: NFT Trading Game (3 hours)**

Create a game where players collect, trade, and battle with NFT creatures!

#### Game Contracts

```solidity
// contracts/CreatureNFT.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract CreatureNFT is ERC721 {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;
    
    struct Creature {
        string name;
        uint256 strength;
        uint256 speed;
        uint256 intelligence;
        uint256 level;
        uint256 experience;
        string element; // fire, water, earth, air
    }
    
    mapping(uint256 => Creature) public creatures;
    mapping(address => uint256[]) public ownerCreatures;
    
    event CreatureMinted(uint256 tokenId, address owner, string name);
    event CreatureLevelUp(uint256 tokenId, uint256 newLevel);
    
    constructor() ERC721("Battle Creatures", "CREATURE") {}
    
    function mintCreature(
        address to,
        string memory name,
        string memory element
    ) external returns (uint256) {
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _mint(to, newTokenId);
        
        // Random stats (in production, use Chainlink VRF for true randomness)
        uint256 randomSeed = uint256(keccak256(abi.encodePacked(
            block.timestamp, block.difficulty, to, newTokenId
        )));
        
        creatures[newTokenId] = Creature({
            name: name,
            strength: (randomSeed % 50) + 50,    // 50-99
            speed: ((randomSeed / 100) % 50) + 50,
            intelligence: ((randomSeed / 10000) % 50) + 50,
            level: 1,
            experience: 0,
            element: element
        });
        
        ownerCreatures[to].push(newTokenId);
        
        emit CreatureMinted(newTokenId, to, name);
        return newTokenId;
    }
    
    function addExperience(uint256 tokenId, uint256 exp) external {
        require(_exists(tokenId), "Creature does not exist");
        require(ownerOf(tokenId) == msg.sender, "Not the owner");
        
        creatures[tokenId].experience += exp;
        
        // Level up logic
        uint256 expNeeded = creatures[tokenId].level * 100;
        if (creatures[tokenId].experience >= expNeeded) {
            creatures[tokenId].level++;
            creatures[tokenId].experience = 0;
            
            // Increase stats on level up
            creatures[tokenId].strength += 5;
            creatures[tokenId].speed += 5;
            creatures[tokenId].intelligence += 5;
            
            emit CreatureLevelUp(tokenId, creatures[tokenId].level);
        }
    }
    
    function getCreaturesByOwner(address owner) external view returns (uint256[] memory) {
        return ownerCreatures[owner];
    }
    
    function getCreatureStats(uint256 tokenId) external view returns (
        string memory name,
        uint256 strength,
        uint256 speed,
        uint256 intelligence,
        uint256 level,
        string memory element
    ) {
        require(_exists(tokenId), "Creature does not exist");
        Creature memory creature = creatures[tokenId];
        
        return (
            creature.name,
            creature.strength,
            creature.speed,
            creature.intelligence,
            creature.level,
            creature.element
        );
    }
}
```

```solidity
// contracts/BattleArena.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./CreatureNFT.sol";

contract BattleArena {
    CreatureNFT public creatureContract;
    
    struct Battle {
        address player1;
        address player2;
        uint256 creature1Id;
        uint256 creature2Id;
        address winner;
        bool finished;
        uint256 wager;
    }
    
    mapping(uint256 => Battle) public battles;
    uint256 public battleCounter;
    
    event BattleCreated(uint256 battleId, address player1, uint256 creature1Id, uint256 wager);
    event BattleJoined(uint256 battleId, address player2, uint256 creature2Id);
    event BattleFinished(uint256 battleId, address winner, uint256 winnings);
    
    constructor(address _creatureContract) {
        creatureContract = CreatureNFT(_creatureContract);
    }
    
    function createBattle(uint256 creatureId) external payable {
        require(creatureContract.ownerOf(creatureId) == msg.sender, "Not your creature");
        require(msg.value > 0, "Must wager something");
        
        battleCounter++;
        battles[battleCounter] = Battle({
            player1: msg.sender,
            player2: address(0),
            creature1Id: creatureId,
            creature2Id: 0,
            winner: address(0),
            finished: false,
            wager: msg.value
        });
        
        emit BattleCreated(battleCounter, msg.sender, creatureId, msg.value);
    }
    
    function joinBattle(uint256 battleId, uint256 creatureId) external payable {
        Battle storage battle = battles[battleId];
        require(!battle.finished, "Battle already finished");
        require(battle.player2 == address(0), "Battle already has two players");
        require(msg.value == battle.wager, "Incorrect wager amount");
        require(creatureContract.ownerOf(creatureId) == msg.sender, "Not your creature");
        
        battle.player2 = msg.sender;
        battle.creature2Id = creatureId;
        
        emit BattleJoined(battleId, msg.sender, creatureId);
        
        // Automatically resolve battle
        _resolveBattle(battleId);
    }
    
    function _resolveBattle(uint256 battleId) internal {
        Battle storage battle = battles[battleId];
        
        // Get creature stats
        (, uint256 str1, uint256 spd1, uint256 int1,,) = creatureContract.getCreatureStats(battle.creature1Id);
        (, uint256 str2, uint256 spd2, uint256 int2,,) = creatureContract.getCreatureStats(battle.creature2Id);
        
        // Calculate battle power (simple formula)
        uint256 power1 = str1 + spd1 + int1;
        uint256 power2 = str2 + spd2 + int2;
        
        // Add some randomness
        uint256 random = uint256(keccak256(abi.encodePacked(
            block.timestamp, block.difficulty, battleId
        )));
        
        power1 += (random % 50);
        power2 += ((random / 100) % 50);
        
        // Determine winner
        address winner;
        if (power1 > power2) {
            winner = battle.player1;
        } else if (power2 > power1) {
            winner = battle.player2;
        } else {
            // Tie - refund both players
            payable(battle.player1).transfer(battle.wager);
            payable(battle.player2).transfer(battle.wager);
            battle.finished = true;
            emit BattleFinished(battleId, address(0), 0);
            return;
        }
        
        battle.winner = winner;
        battle.finished = true;
        
        // Pay winner
        uint256 winnings = battle.wager * 2;
        payable(winner).transfer(winnings);
        
        // Give experience to both creatures
        if (winner == battle.player1) {
            creatureContract.addExperience(battle.creature1Id, 50); // Winner gets more
            creatureContract.addExperience(battle.creature2Id, 25);
        } else {
            creatureContract.addExperience(battle.creature2Id, 50);
            creatureContract.addExperience(battle.creature1Id, 25);
        }
        
        emit BattleFinished(battleId, winner, winnings);
    }
}
```

## 🚀 Advanced Examples

Enterprise-grade applications and complex systems.

### 🏢 **Enterprise Project: Supply Chain Tracking (4+ hours)**

Build a complete supply chain tracking system with quantum-resistant features!

#### Core Architecture

```solidity
// contracts/SupplyChain.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract SupplyChain is AccessControl, ReentrancyGuard {
    bytes32 public constant MANUFACTURER_ROLE = keccak256("MANUFACTURER_ROLE");
    bytes32 public constant DISTRIBUTOR_ROLE = keccak256("DISTRIBUTOR_ROLE");
    bytes32 public constant RETAILER_ROLE = keccak256("RETAILER_ROLE");
    bytes32 public constant AUDITOR_ROLE = keccak256("AUDITOR_ROLE");
    
    enum ProductState { 
        Manufactured, 
        InTransit, 
        Distributed, 
        InStore, 
        Sold, 
        Recalled 
    }
    
    struct Product {
        uint256 id;
        string name;
        string batchNumber;
        address manufacturer;
        uint256 manufacturedDate;
        ProductState state;
        mapping(ProductState => uint256) stateTimestamps;
        mapping(ProductState => address) stateActors;
        string[] certificates; // Quality certificates
        bytes32 quantumSignature; // Quantum-resistant signature
    }
    
    struct TransferEvent {
        uint256 productId;
        address from;
        address to;
        uint256 timestamp;
        string location;
        string notes;
        bytes32 signature;
    }
    
    mapping(uint256 => Product) public products;
    mapping(uint256 => TransferEvent[]) public productHistory;
    mapping(string => uint256[]) public batchProducts;
    mapping(address => bool) public verifiedEntities;
    
    uint256 public productCounter;
    
    event ProductManufactured(uint256 productId, string name, address manufacturer);
    event ProductTransferred(uint256 productId, address from, address to, string location);
    event ProductStateChanged(uint256 productId, ProductState newState);
    event QualityCertificateAdded(uint256 productId, string certificate);
    event ProductRecalled(uint256 productId, string reason);
    
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }
    
    modifier onlyVerifiedEntity() {
        require(verifiedEntities[msg.sender], "Entity not verified");
        _;
    }
    
    function addVerifiedEntity(address entity, bytes32 role) external onlyRole(DEFAULT_ADMIN_ROLE) {
        verifiedEntities[entity] = true;
        _grantRole(role, entity);
    }
    
    function manufactureProduct(
        string memory name,
        string memory batchNumber,
        string memory initialCertificate
    ) external onlyRole(MANUFACTURER_ROLE) onlyVerifiedEntity returns (uint256) {
        productCounter++;
        uint256 productId = productCounter;
        
        Product storage product = products[productId];
        product.id = productId;
        product.name = name;
        product.batchNumber = batchNumber;
        product.manufacturer = msg.sender;
        product.manufacturedDate = block.timestamp;
        product.state = ProductState.Manufactured;
        product.stateTimestamps[ProductState.Manufactured] = block.timestamp;
        product.stateActors[ProductState.Manufactured] = msg.sender;
        
        if (bytes(initialCertificate).length > 0) {
            product.certificates.push(initialCertificate);
        }
        
        // Generate quantum-resistant signature
        product.quantumSignature = generateQuantumSignature(productId, msg.sender);
        
        batchProducts[batchNumber].push(productId);
        
        // Add initial event to history
        productHistory[productId].push(TransferEvent({
            productId: productId,
            from: address(0),
            to: msg.sender,
            timestamp: block.timestamp,
            location: "Manufacturing Facility",
            notes: "Product manufactured",
            signature: product.quantumSignature
        }));
        
        emit ProductManufactured(productId, name, msg.sender);
        return productId;
    }
    
    function transferProduct(
        uint256 productId,
        address to,
        string memory location,
        string memory notes
    ) external onlyVerifiedEntity nonReentrant {
        require(products[productId].id != 0, "Product does not exist");
        require(canTransferProduct(productId, msg.sender), "Cannot transfer this product");
        
        Product storage product = products[productId];
        ProductState newState = determineNewState(to);
        
        // Update product state
        product.state = newState;
        product.stateTimestamps[newState] = block.timestamp;
        product.stateActors[newState] = to;
        
        // Add transfer event
        bytes32 transferSignature = generateTransferSignature(productId, msg.sender, to);
        productHistory[productId].push(TransferEvent({
            productId: productId,
            from: msg.sender,
            to: to,
            timestamp: block.timestamp,
            location: location,
            notes: notes,
            signature: transferSignature
        }));
        
        emit ProductTransferred(productId, msg.sender, to, location);
        emit ProductStateChanged(productId, newState);
    }
    
    function addQualityCertificate(
        uint256 productId,
        string memory certificate
    ) external onlyRole(AUDITOR_ROLE) {
        require(products[productId].id != 0, "Product does not exist");
        
        products[productId].certificates.push(certificate);
        emit QualityCertificateAdded(productId, certificate);
    }
    
    function recallProduct(
        uint256 productId,
        string memory reason
    ) external onlyRole(MANUFACTURER_ROLE) {
        require(products[productId].id != 0, "Product does not exist");
        require(products[productId].manufacturer == msg.sender, "Not the manufacturer");
        
        products[productId].state = ProductState.Recalled;
        products[productId].stateTimestamps[ProductState.Recalled] = block.timestamp;
        products[productId].stateActors[ProductState.Recalled] = msg.sender;
        
        emit ProductRecalled(productId, reason);
    }
    
    function getProductHistory(uint256 productId) external view returns (TransferEvent[] memory) {
        return productHistory[productId];
    }
    
    function getProductsByBatch(string memory batchNumber) external view returns (uint256[] memory) {
        return batchProducts[batchNumber];
    }
    
    function getProductCertificates(uint256 productId) external view returns (string[] memory) {
        return products[productId].certificates;
    }
    
    function verifyProductAuthenticity(uint256 productId) external view returns (bool) {
        if (products[productId].id == 0) return false;
        
        // Verify quantum signature
        bytes32 expectedSignature = generateQuantumSignature(
            productId, 
            products[productId].manufacturer
        );
        
        return products[productId].quantumSignature == expectedSignature;
    }
    
    // Internal functions
    function canTransferProduct(uint256 productId, address from) internal view returns (bool) {
        Product storage product = products[productId];
        
        if (product.state == ProductState.Recalled) return false;
        
        // Check if sender is current holder based on last transfer
        TransferEvent[] storage history = productHistory[productId];
        if (history.length == 0) return false;
        
        return history[history.length - 1].to == from;
    }
    
    function determineNewState(address to) internal view returns (ProductState) {
        if (hasRole(DISTRIBUTOR_ROLE, to)) return ProductState.Distributed;
        if (hasRole(RETAILER_ROLE, to)) return ProductState.InStore;
        return ProductState.InTransit;
    }
    
    function generateQuantumSignature(uint256 productId, address signer) internal pure returns (bytes32) {
        // In production, this would use actual quantum-resistant cryptography
        return keccak256(abi.encodePacked("QUANTUM_SIG", productId, signer));
    }
    
    function generateTransferSignature(
        uint256 productId, 
        address from, 
        address to
    ) internal view returns (bytes32) {
        return keccak256(abi.encodePacked(productId, from, to, block.timestamp));
    }
}
```

### 📱 **Mobile Integration Example**

```javascript
// React Native app for supply chain tracking
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity } from 'react-native';
import { ethers } from 'ethers';

const SupplyChainTracker = () => {
    const [products, setProducts] = useState([]);
    const [contract, setContract] = useState(null);
    
    useEffect(() => {
        initializeContract();
    }, []);
    
    const initializeContract = async () => {
        try {
            // Connect to ArthaChain
            const provider = new ethers.providers.JsonRpcProvider(
                'https://testnet.arthachain.online/rpc'
            );
            
            const contractInstance = new ethers.Contract(
                CONTRACT_ADDRESS,
                CONTRACT_ABI,
                provider
            );
            
            setContract(contractInstance);
            loadProducts();
        } catch (error) {
            console.error('Failed to initialize contract:', error);
        }
    };
    
    const loadProducts = async () => {
        if (!contract) return;
        
        try {
            // Listen for new products
            contract.on('ProductManufactured', (productId, name, manufacturer) => {
                const newProduct = {
                    id: productId.toString(),
                    name,
                    manufacturer,
                    state: 'Manufactured'
                };
                
                setProducts(prev => [...prev, newProduct]);
            });
            
            // Load existing products (in production, you'd paginate this)
            const productCount = await contract.productCounter();
            const productPromises = [];
            
            for (let i = 1; i <= productCount; i++) {
                productPromises.push(contract.products(i));
            }
            
            const productData = await Promise.all(productPromises);
            const formattedProducts = productData.map((product, index) => ({
                id: (index + 1).toString(),
                name: product.name,
                manufacturer: product.manufacturer,
                state: getStateString(product.state)
            }));
            
            setProducts(formattedProducts);
        } catch (error) {
            console.error('Failed to load products:', error);
        }
    };
    
    const getStateString = (state) => {
        const states = ['Manufactured', 'InTransit', 'Distributed', 'InStore', 'Sold', 'Recalled'];
        return states[state] || 'Unknown';
    };
    
    const trackProduct = async (productId) => {
        try {
            const history = await contract.getProductHistory(productId);
            
            // Navigate to tracking details screen
            navigation.navigate('ProductTracking', { 
                productId, 
                history: history.map(event => ({
                    from: event.from,
                    to: event.to,
                    timestamp: new Date(event.timestamp * 1000),
                    location: event.location,
                    notes: event.notes
                }))
            });
        } catch (error) {
            console.error('Failed to track product:', error);
        }
    };
    
    const renderProduct = ({ item }) => (
        <TouchableOpacity 
            style={styles.productCard}
            onPress={() => trackProduct(item.id)}
        >
            <Text style={styles.productName}>{item.name}</Text>
            <Text style={styles.productId}>ID: {item.id}</Text>
            <Text style={styles.productState}>{item.state}</Text>
            <Text style={styles.manufacturer}>
                Manufacturer: {item.manufacturer.substring(0, 10)}...
            </Text>
        </TouchableOpacity>
    );
    
    return (
        <View style={styles.container}>
            <Text style={styles.title}>Supply Chain Tracker</Text>
            <FlatList
                data={products}
                renderItem={renderProduct}
                keyExtractor={(item) => item.id}
                style={styles.productList}
            />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5',
        padding: 16,
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 16,
        textAlign: 'center',
    },
    productCard: {
        backgroundColor: 'white',
        padding: 16,
        marginBottom: 8,
        borderRadius: 8,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
        elevation: 3,
    },
    productName: {
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 4,
    },
    productId: {
        fontSize: 14,
        color: '#666',
        marginBottom: 4,
    },
    productState: {
        fontSize: 16,
        color: '#007AFF',
        marginBottom: 4,
    },
    manufacturer: {
        fontSize: 12,
        color: '#999',
    },
    productList: {
        flex: 1,
    },
});

export default SupplyChainTracker;
```

## 🤖 AI Integration Examples

Using ArthaChain's built-in AI features.

### 🧠 **AI-Powered Smart Contract**

```rust
// WASM contract with AI integration
use arthachain_sdk::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct AIContract {
    owner: Address,
    ai_enabled: bool,
    fraud_threshold: f64,
    transaction_history: Vec<TransactionRecord>,
    ai_model_version: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TransactionRecord {
    from: Address,
    to: Address,
    amount: u128,
    timestamp: u64,
    ai_risk_score: f64,
    approved: bool,
}

#[contract_impl]
impl AIContract {
    #[constructor]
    pub fn new() -> Self {
        Self {
            owner: msg_sender(),
            ai_enabled: true,
            fraud_threshold: 0.7,
            transaction_history: Vec::new(),
            ai_model_version: "v2.1.0".to_string(),
        }
    }
    
    #[public]
    pub async fn smart_transfer(&mut self, to: Address, amount: u128) -> bool {
        let sender = msg_sender();
        
        if self.ai_enabled {
            // Get AI risk analysis
            let risk_analysis = self.analyze_transaction_risk(&sender, &to, amount).await;
            
            if risk_analysis.risk_score > self.fraud_threshold {
                emit_event("TransactionBlocked", &(sender, to, amount, risk_analysis.risk_score));
                return false;
            }
            
            // Store transaction record with AI analysis
            let record = TransactionRecord {
                from: sender,
                to,
                amount,
                timestamp: block_timestamp(),
                ai_risk_score: risk_analysis.risk_score,
                approved: true,
            };
            
            self.transaction_history.push(record);
        }
        
        // Execute transfer
        self.execute_transfer(sender, to, amount);
        emit_event("TransferCompleted", &(sender, to, amount));
        true
    }
    
    async fn analyze_transaction_risk(
        &self, 
        from: &Address, 
        to: &Address, 
        amount: u128
    ) -> RiskAnalysis {
        // Call ArthaChain's AI fraud detection API
        let api_result = call_ai_api("fraud_detection", &TransactionData {
            from: from.clone(),
            to: to.clone(),
            amount,
            sender_history: self.get_sender_history(from),
            recipient_history: self.get_recipient_history(to),
            network_context: get_network_context(),
        }).await;
        
        match api_result {
            Ok(analysis) => analysis,
            Err(_) => RiskAnalysis {
                risk_score: 0.5, // Default to medium risk if AI unavailable
                confidence: 0.0,
                factors: vec!["AI_UNAVAILABLE".to_string()],
            }
        }
    }
    
    fn get_sender_history(&self, address: &Address) -> Vec<TransactionRecord> {
        self.transaction_history
            .iter()
            .filter(|record| &record.from == address)
            .take(10) // Last 10 transactions
            .cloned()
            .collect()
    }
    
    #[public]
    pub fn get_ai_insights(&self, address: Address) -> AIInsights {
        let user_transactions: Vec<_> = self.transaction_history
            .iter()
            .filter(|record| record.from == address || record.to == address)
            .collect();
        
        let avg_risk_score = if user_transactions.is_empty() {
            0.0
        } else {
            user_transactions.iter()
                .map(|record| record.ai_risk_score)
                .sum::<f64>() / user_transactions.len() as f64
        };
        
        let risk_trend = self.calculate_risk_trend(&user_transactions);
        
        AIInsights {
            average_risk_score: avg_risk_score,
            total_transactions: user_transactions.len(),
            risk_trend,
            recommendations: self.generate_recommendations(avg_risk_score),
        }
    }
    
    fn calculate_risk_trend(&self, transactions: &[&TransactionRecord]) -> String {
        if transactions.len() < 2 {
            return "INSUFFICIENT_DATA".to_string();
        }
        
        let recent_avg = transactions.iter()
            .rev()
            .take(5)
            .map(|record| record.ai_risk_score)
            .sum::<f64>() / 5.0;
            
        let older_avg = transactions.iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|record| record.ai_risk_score)
            .sum::<f64>() / 5.0;
        
        if recent_avg > older_avg + 0.1 {
            "INCREASING_RISK".to_string()
        } else if recent_avg < older_avg - 0.1 {
            "DECREASING_RISK".to_string()
        } else {
            "STABLE".to_string()
        }
    }
    
    #[public]
    pub fn update_ai_settings(&mut self, threshold: f64, enabled: bool) {
        require(msg_sender() == self.owner, "Only owner can update AI settings");
        require(threshold >= 0.0 && threshold <= 1.0, "Invalid threshold");
        
        self.fraud_threshold = threshold;
        self.ai_enabled = enabled;
        
        emit_event("AISettingsUpdated", &(threshold, enabled));
    }
}

#[derive(Serialize, Deserialize)]
struct RiskAnalysis {
    risk_score: f64,
    confidence: f64,
    factors: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct AIInsights {
    average_risk_score: f64,
    total_transactions: usize,
    risk_trend: String,
    recommendations: Vec<String>,
}
```

## 📚 Tutorial Templates

### 🛠️ **Quick Start Templates**

```bash
# Create project from template
arthachain create --template [template-name]

# Available templates:
arthachain create --template erc20-token
arthachain create --template nft-collection  
arthachain create --template defi-staking
arthachain create --template gaming-nft
arthachain create --template dao-governance
arthachain create --template supply-chain
arthachain create --template ai-powered-contract
```

### 📖 **Learning Path Recommendations**

```
🎓 Recommended Learning Path:

👶 Beginner (Week 1-2):
├── Day 1-2: Complete [Getting Started](./getting-started.md)
├── Day 3-4: Tutorial 1 (Your First Token)
├── Day 5-6: Tutorial 2 (Simple NFT)
├── Day 7-10: Tutorial 3 (Rock Paper Scissors Game)
└── Day 11-14: Practice and experiment

👨‍💻 Intermediate (Week 3-6):
├── Week 3: DeFi Yield Farm project
├── Week 4: NFT Trading Game project
├── Week 5: Custom project (your idea!)
└── Week 6: Security and optimization

🚀 Advanced (Month 2+):
├── Enterprise supply chain project
├── AI integration projects
├── Mobile app development
├── Cross-chain integration
└── Contribute to ArthaChain core
```

## 🎯 What's Next?

### 📚 **More Learning Resources**
1. **[🎥 Video Tutorials](https://youtube.com/@arthachain)** - Visual learning
2. **[📝 Blog Posts](https://blog.arthachain.com)** - Weekly tutorials
3. **[🎮 Interactive Playground](https://playground.arthachain.online)** - Learn by doing
4. **[📚 Advanced Patterns](./advanced-patterns.md)** - Pro developer techniques

### 🤝 **Community & Support**
1. **[💬 Developer Discord](https://discord.gg/arthachain-dev)** - Get help with tutorials
2. **[📱 Telegram Group](https://t.me/arthachain_tutorials)** - Quick questions
3. **[🎓 Study Groups](https://study.arthachain.com)** - Learn with others
4. **[🏆 Hackathons](https://hackathons.arthachain.com)** - Build and win prizes

### 💡 **Project Ideas**
1. **🏥 Healthcare Records** - Secure patient data management
2. **🎓 Education Certificates** - Tamper-proof academic credentials
3. **🏛️ Voting Systems** - Transparent, secure elections
4. **🌱 Carbon Credits** - Environmental impact tracking
5. **🎨 Digital Art Platform** - Create your own OpenSea
6. **🚗 Vehicle History** - Used car transparency
7. **💊 Pharmaceutical Tracking** - Drug authenticity verification

---

**🎯 Next**: [⚛️ Advanced Topics](./advanced-topics.md) →

**💬 Questions?** Join our [Tutorial Discord](https://discord.gg/arthachain-tutorials) - we love helping you learn! 