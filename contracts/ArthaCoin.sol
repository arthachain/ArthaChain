// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts-upgradeable/token/ERC20/ERC20Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "./interfaces/ICycleManager.sol";
import "./interfaces/IBurnManager.sol";
import "./interfaces/IAntiWhaleManager.sol";

/**
 * @title ArthaCoin (ARTHA)
 * @notice Native Layer 1 token for ArthaChain with advanced tokenomics
 * @dev Upgradeable ERC20 token with emission cycles, burn mechanics, and anti-whale protection
 * 
 * Key Features:
 * - Zero initial supply, emissions through 3-year cycles
 * - Progressive burn rates (40% → 96% over 17+ years)
 * - Anti-whale protection (max 1.5% holding, 0.5% transfer)
 * - Multi-pool allocation system for emissions
 * - Fully upgradeable with UUPS pattern
 */
contract ArthaCoin is 
    Initializable,
    ERC20Upgradeable, 
    AccessControlUpgradeable, 
    UUPSUpgradeable 
{
    // ==================== ROLES ====================
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
    bytes32 public constant MANAGER_ROLE = keccak256("MANAGER_ROLE");

    // ==================== STATE VARIABLES ====================
    
    /// @notice Manager contracts for modular functionality
    ICycleManager public cycleManager;
    IBurnManager public burnManager;
    IAntiWhaleManager public antiWhaleManager;
    
    /// @notice Treasury addresses for different allocations
    address public validatorsPool;
    address public stakingRewardsPool;
    address public ecosystemGrantsPool;
    address public marketingWallet;
    address public developersPool;
    address public daoGovernancePool;
    address public treasuryReserve;
    
    /// @notice Deployment timestamp for calculating cycles and burn rates
    uint256 public immutable deploymentTime;
    
    /// @notice Total amount burned throughout the token's lifetime
    uint256 public totalBurned;
    
    /// @notice Mapping to track if address is exempt from burn (like pools)
    mapping(address => bool) public burnExempt;
    
    // ==================== EVENTS ====================
    event TokensBurned(address indexed from, uint256 amount, uint256 burnRate);
    event CycleEmissionMinted(uint256 indexed cycle, uint256 totalAmount);
    event ManagerContractUpdated(string indexed managerType, address indexed newAddress);
    event PoolAddressUpdated(string indexed poolName, address indexed newAddress);
    event BurnExemptionUpdated(address indexed account, bool exempt);
    event AntiWhaleViolation(address indexed account, string violationType, uint256 amount, uint256 limit);

    // ==================== CUSTOM ERRORS ====================
    error ZeroAddress();
    error InvalidManager();
    error OnlyManagerContract();
    error TransferExceedsLimit(uint256 amount, uint256 limit);
    error HoldingExceedsLimit(uint256 balance, uint256 limit);
    error ManagerNotSet();

    // ==================== MODIFIERS ====================
    modifier onlyManager() {
        if (!hasRole(MANAGER_ROLE, msg.sender)) {
            revert OnlyManagerContract();
        }
        _;
    }

    modifier validAddress(address addr) {
        if (addr == address(0)) {
            revert ZeroAddress();
        }
        _;
    }

    // ==================== CONSTRUCTOR & INITIALIZER ====================
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        deploymentTime = block.timestamp;
        _disableInitializers();
    }

    /**
     * @notice Initializes the ArthaCoin contract
     * @param _admin Address to receive admin roles
     * @param _validatorsPool Address for validator rewards
     * @param _stakingRewardsPool Address for staking rewards
     * @param _ecosystemGrantsPool Address for ecosystem grants
     * @param _marketingWallet Address for marketing funds
     * @param _developersPool Address for developer rewards
     * @param _daoGovernancePool Address for DAO governance
     * @param _treasuryReserve Address for treasury reserve
     */
    function initialize(
        address _admin,
        address _validatorsPool,
        address _stakingRewardsPool,
        address _ecosystemGrantsPool,
        address _marketingWallet,
        address _developersPool,
        address _daoGovernancePool,
        address _treasuryReserve
    ) public initializer {
        __ERC20_init("ArthaCoin", "ARTHA");
        __AccessControl_init();
        __UUPSUpgradeable_init();

        // Grant roles to admin
        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(MINTER_ROLE, _admin);
        _grantRole(GOVERNANCE_ROLE, _admin);
        _grantRole(UPGRADER_ROLE, _admin);
        _grantRole(MANAGER_ROLE, _admin);

        // Set pool addresses
        _setPoolAddresses(
            _validatorsPool,
            _stakingRewardsPool,
            _ecosystemGrantsPool,
            _marketingWallet,
            _developersPool,
            _daoGovernancePool,
            _treasuryReserve
        );

        // Make pools burn-exempt
        burnExempt[_validatorsPool] = true;
        burnExempt[_stakingRewardsPool] = true;
        burnExempt[_ecosystemGrantsPool] = true;
        burnExempt[_marketingWallet] = true;
        burnExempt[_developersPool] = true;
        burnExempt[_daoGovernancePool] = true;
        burnExempt[_treasuryReserve] = true;
    }

    // ==================== EMISSION FUNCTIONS ====================

    /**
     * @notice Mints tokens for the next emission cycle
     * @dev Can only be called by MINTER_ROLE (usually CycleManager)
     * @return totalMinted Total amount of tokens minted
     */
    function mintNextCycle() external onlyRole(MINTER_ROLE) returns (uint256 totalMinted) {
        if (address(cycleManager) == address(0)) {
            revert ManagerNotSet();
        }

        // Get current cycle info from CycleManager
        (uint256 currentCycle, uint256 emissionAmount, bool canMint) = cycleManager.getCurrentCycleInfo();
        
        require(canMint, "ArthaCoin: Cannot mint yet");

        // Calculate allocations (percentages are in basis points, e.g., 4500 = 45%)
        uint256 validatorsAmount = (emissionAmount * 4500) / 10000; // 45%
        uint256 stakingAmount = (emissionAmount * 2000) / 10000;    // 20%
        uint256 ecosystemAmount = (emissionAmount * 1000) / 10000;  // 10%
        uint256 marketingAmount = (emissionAmount * 1000) / 10000;  // 10%
        uint256 developersAmount = (emissionAmount * 500) / 10000;  // 5%
        uint256 daoAmount = (emissionAmount * 500) / 10000;         // 5%
        uint256 treasuryAmount = (emissionAmount * 500) / 10000;    // 5%

        // Mint to respective pools
        _mint(validatorsPool, validatorsAmount);
        _mint(stakingRewardsPool, stakingAmount);
        _mint(ecosystemGrantsPool, ecosystemAmount);
        _mint(marketingWallet, marketingAmount);
        _mint(developersPool, developersAmount);
        _mint(daoGovernancePool, daoAmount);
        _mint(treasuryReserve, treasuryAmount);

        totalMinted = emissionAmount;

        // Update cycle in CycleManager
        cycleManager.completeCycle();

        emit CycleEmissionMinted(currentCycle, totalMinted);
    }

    /**
     * @notice Emergency mint function for governance
     * @param to Address to mint tokens to
     * @param amount Amount of tokens to mint
     */
    function emergencyMint(address to, uint256 amount) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
        validAddress(to) 
    {
        _mint(to, amount);
    }

    // ==================== BURN FUNCTIONS ====================

    /**
     * @notice Burns tokens from an account
     * @param account Account to burn from
     * @param amount Amount to burn
     */
    function burn(address account, uint256 amount) external onlyManager {
        _burn(account, amount);
        totalBurned += amount;
    }

    /**
     * @notice Burns tokens from caller's account
     * @param amount Amount to burn
     */
    function burnSelf(uint256 amount) external {
        _burn(msg.sender, amount);
        totalBurned += amount;
    }

    // ==================== TRANSFER OVERRIDES ====================

    /**
     * @notice Override transfer to implement burn-on-transfer and anti-whale
     * @param from Sender address
     * @param to Recipient address
     * @param amount Transfer amount
     */
    function _transfer(address from, address to, uint256 amount) internal override {
        // Skip checks for minting/burning or if managers not set
        if (from == address(0) || to == address(0) || 
            address(antiWhaleManager) == address(0) || 
            address(burnManager) == address(0)) {
            super._transfer(from, to, amount);
            return;
        }

        // Anti-whale checks
        antiWhaleManager.validateTransfer(from, to, amount, totalSupply());

        // Calculate burn amount if not exempt
        uint256 burnAmount = 0;
        if (!burnExempt[from] && !burnExempt[to]) {
            uint256 burnRate = burnManager.getCurrentBurnRate();
            burnAmount = (amount * burnRate) / 10000; // burnRate is in basis points
        }

        // Execute burn if applicable
        if (burnAmount > 0) {
            _burn(from, burnAmount);
            totalBurned += burnAmount;
            emit TokensBurned(from, burnAmount, burnManager.getCurrentBurnRate());
        }

        // Execute transfer (reduced by burn amount)
        uint256 transferAmount = amount - burnAmount;
        super._transfer(from, to, transferAmount);
    }

    // ==================== MANAGER FUNCTIONS ====================

    /**
     * @notice Sets the CycleManager contract
     * @param _cycleManager Address of the CycleManager contract
     */
    function setCycleManager(address _cycleManager) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
        validAddress(_cycleManager)
    {
        cycleManager = ICycleManager(_cycleManager);
        _grantRole(MANAGER_ROLE, _cycleManager);
        emit ManagerContractUpdated("CycleManager", _cycleManager);
    }

    /**
     * @notice Sets the BurnManager contract
     * @param _burnManager Address of the BurnManager contract
     */
    function setBurnManager(address _burnManager) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
        validAddress(_burnManager)
    {
        burnManager = IBurnManager(_burnManager);
        _grantRole(MANAGER_ROLE, _burnManager);
        emit ManagerContractUpdated("BurnManager", _burnManager);
    }

    /**
     * @notice Sets the AntiWhaleManager contract
     * @param _antiWhaleManager Address of the AntiWhaleManager contract
     */
    function setAntiWhaleManager(address _antiWhaleManager) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
        validAddress(_antiWhaleManager)
    {
        antiWhaleManager = IAntiWhaleManager(_antiWhaleManager);
        _grantRole(MANAGER_ROLE, _antiWhaleManager);
        emit ManagerContractUpdated("AntiWhaleManager", _antiWhaleManager);
    }

    // ==================== POOL MANAGEMENT ====================

    /**
     * @notice Updates pool addresses (governance only)
     */
    function updatePoolAddresses(
        address _validatorsPool,
        address _stakingRewardsPool,
        address _ecosystemGrantsPool,
        address _marketingWallet,
        address _developersPool,
        address _daoGovernancePool,
        address _treasuryReserve
    ) external onlyRole(GOVERNANCE_ROLE) {
        _setPoolAddresses(
            _validatorsPool,
            _stakingRewardsPool,
            _ecosystemGrantsPool,
            _marketingWallet,
            _developersPool,
            _daoGovernancePool,
            _treasuryReserve
        );
    }

    /**
     * @notice Sets burn exemption status for an address
     * @param account Address to update
     * @param exempt Whether address should be exempt from burns
     */
    function setBurnExempt(address account, bool exempt) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
    {
        burnExempt[account] = exempt;
        emit BurnExemptionUpdated(account, exempt);
    }

    // ==================== VIEW FUNCTIONS ====================

    /**
     * @notice Gets current burn rate from BurnManager
     * @return Current burn rate in basis points (e.g., 4000 = 40%)
     */
    function getCurrentBurnRate() external view returns (uint256) {
        if (address(burnManager) == address(0)) return 0;
        return burnManager.getCurrentBurnRate();
    }

    /**
     * @notice Gets current cycle information
     * @return cycle Current emission cycle
     * @return emissionAmount Amount to be minted in current cycle
     * @return canMint Whether minting is currently allowed
     */
    function getCurrentCycleInfo() external view returns (uint256 cycle, uint256 emissionAmount, bool canMint) {
        if (address(cycleManager) == address(0)) return (0, 0, false);
        return cycleManager.getCurrentCycleInfo();
    }

    /**
     * @notice Gets the maximum transferable amount (0.5% of total supply)
     * @return Maximum transfer amount
     */
    function getMaxTransferAmount() external view returns (uint256) {
        if (address(antiWhaleManager) == address(0)) return type(uint256).max;
        return antiWhaleManager.getMaxTransferAmount(totalSupply());
    }

    /**
     * @notice Gets the maximum holdable amount (1.5% of total supply)
     * @return Maximum holding amount
     */
    function getMaxHoldingAmount() external view returns (uint256) {
        if (address(antiWhaleManager) == address(0)) return type(uint256).max;
        return antiWhaleManager.getMaxHoldingAmount(totalSupply());
    }

    /**
     * @notice Returns years since deployment
     * @return Years elapsed since contract deployment
     */
    function getYearsSinceDeployment() external view returns (uint256) {
        return (block.timestamp - deploymentTime) / 365 days;
    }

    // ==================== INTERNAL FUNCTIONS ====================

    /**
     * @notice Internal function to set all pool addresses
     */
    function _setPoolAddresses(
        address _validatorsPool,
        address _stakingRewardsPool,
        address _ecosystemGrantsPool,
        address _marketingWallet,
        address _developersPool,
        address _daoGovernancePool,
        address _treasuryReserve
    ) internal {
        if (_validatorsPool == address(0) || _stakingRewardsPool == address(0) ||
            _ecosystemGrantsPool == address(0) || _marketingWallet == address(0) ||
            _developersPool == address(0) || _daoGovernancePool == address(0) ||
            _treasuryReserve == address(0)) {
            revert ZeroAddress();
        }

        validatorsPool = _validatorsPool;
        stakingRewardsPool = _stakingRewardsPool;
        ecosystemGrantsPool = _ecosystemGrantsPool;
        marketingWallet = _marketingWallet;
        developersPool = _developersPool;
        daoGovernancePool = _daoGovernancePool;
        treasuryReserve = _treasuryReserve;

        emit PoolAddressUpdated("ValidatorsPool", _validatorsPool);
        emit PoolAddressUpdated("StakingRewardsPool", _stakingRewardsPool);
        emit PoolAddressUpdated("EcosystemGrantsPool", _ecosystemGrantsPool);
        emit PoolAddressUpdated("MarketingWallet", _marketingWallet);
        emit PoolAddressUpdated("DevelopersPool", _developersPool);
        emit PoolAddressUpdated("DaoGovernancePool", _daoGovernancePool);
        emit PoolAddressUpdated("TreasuryReserve", _treasuryReserve);
    }

    // ==================== UPGRADE AUTHORIZATION ====================

    /**
     * @notice Authorizes contract upgrades
     * @param newImplementation Address of new implementation
     */
    function _authorizeUpgrade(address newImplementation) 
        internal 
        override 
        onlyRole(UPGRADER_ROLE) 
    {}

    // ==================== DECIMALS ====================

    /**
     * @notice Returns the number of decimals (18)
     */
    function decimals() public pure override returns (uint8) {
        return 18;
    }
} 