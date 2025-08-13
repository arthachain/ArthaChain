// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "./interfaces/IAntiWhaleManager.sol";

/**
 * @title AntiWhaleManager
 * @notice Manages anti-whale protection for ArthaCoin
 * @dev Enforces maximum holding (1.5% of supply) and transfer limits (0.5% of supply)
 * 
 * Anti-Whale Rules:
 * - No wallet can hold more than 1.5% of total supply
 * - No single transaction can transfer more than 0.5% of total supply
 * - Certain addresses can be exempted (pools, contracts, etc.)
 */
contract AntiWhaleManager is 
    Initializable,
    AccessControlUpgradeable, 
    UUPSUpgradeable,
    IAntiWhaleManager 
{
    // ==================== ROLES ====================
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
    bytes32 public constant TOKEN_CONTRACT_ROLE = keccak256("TOKEN_CONTRACT_ROLE");

    // ==================== CONSTANTS ====================
    
    /// @notice Maximum holding percentage (1.5% = 150 basis points)
    uint256 public constant MAX_HOLDING_PERCENTAGE = 150; // 1.5%
    
    /// @notice Maximum transfer percentage (0.5% = 50 basis points)
    uint256 public constant MAX_TRANSFER_PERCENTAGE = 50; // 0.5%
    
    /// @notice Basis points denominator (10000 = 100%)
    uint256 public constant BASIS_POINTS = 10000;

    // ==================== STATE VARIABLES ====================
    
    /// @notice Mapping of addresses exempt from anti-whale rules
    mapping(address => bool) public override isExempt;
    
    /// @notice Override percentages for holding limit (0 = use default)
    uint256 public holdingLimitOverride;
    
    /// @notice Override percentages for transfer limit (0 = use default)
    uint256 public transferLimitOverride;
    
    /// @notice Whether overrides are active
    bool public overridesActive;
    
    /// @notice Grace period for new holders (in seconds)
    uint256 public gracePeriod;
    
    /// @notice Mapping of when each address first received tokens
    mapping(address => uint256) public firstReceiveTime;

    // ==================== EVENTS ====================
    event ExemptionUpdated(address indexed account, bool exempt);
    event LimitsOverridden(uint256 holdingLimit, uint256 transferLimit, bool active);
    event GracePeriodUpdated(uint256 newGracePeriod);
    event AntiWhaleViolationDetected(
        address indexed account, 
        string violationType, 
        uint256 amount, 
        uint256 limit
    );

    // ==================== CUSTOM ERRORS ====================
    error TransferExceedsLimit(uint256 amount, uint256 limit);
    error HoldingExceedsLimit(uint256 balance, uint256 limit);
    error InvalidPercentage(uint256 percentage);
    error ZeroAddress();

    // ==================== MODIFIERS ====================
    modifier onlyTokenContract() {
        require(hasRole(TOKEN_CONTRACT_ROLE, msg.sender), "Only token contract");
        _;
    }

    modifier validAddress(address addr) {
        if (addr == address(0)) revert ZeroAddress();
        _;
    }

    // ==================== CONSTRUCTOR & INITIALIZER ====================
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /**
     * @notice Initializes the AntiWhaleManager contract
     * @param _admin Address to receive admin roles
     * @param _tokenContract Address of the ArthaCoin token contract
     */
    function initialize(address _admin, address _tokenContract) public initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();

        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(GOVERNANCE_ROLE, _admin);
        _grantRole(UPGRADER_ROLE, _admin);
        _grantRole(TOKEN_CONTRACT_ROLE, _tokenContract);

        // Set default grace period to 24 hours
        gracePeriod = 24 hours;
    }

    // ==================== CORE FUNCTIONS ====================

    /**
     * @notice Validates a transfer against anti-whale rules
     * @param from Sender address
     * @param to Recipient address
     * @param amount Transfer amount
     * @param totalSupply Current total supply
     */
    function validateTransfer(
        address from, 
        address to, 
        uint256 amount, 
        uint256 totalSupply
    ) external view override {
        // Skip validation for exempt addresses
        if (isExempt[from] || isExempt[to]) {
            return;
        }

        // Check transfer amount limit
        uint256 maxTransfer = getMaxTransferAmount(totalSupply);
        if (amount > maxTransfer) {
            revert TransferExceedsLimit(amount, maxTransfer);
        }

        // Check that recipient won't exceed holding limit after transfer
        // Note: We can't get the actual balance here since this is called from _transfer
        // The token contract should check this separately or pass the balance
        uint256 maxHolding = getMaxHoldingAmount(totalSupply);
        
        // Emit event for monitoring (view function can't emit, but we can revert with info)
        if (amount > maxTransfer) {
            revert TransferExceedsLimit(amount, maxTransfer);
        }
    }

    /**
     * @notice Validates that an account's balance doesn't exceed holding limits
     * @param account Account to check
     * @param balance Account's token balance
     * @param totalSupply Current total supply
     */
    function validateHolding(
        address account, 
        uint256 balance, 
        uint256 totalSupply
    ) external view {
        // Skip validation for exempt addresses
        if (isExempt[account]) {
            return;
        }

        // Check if within grace period
        if (firstReceiveTime[account] != 0 && 
            block.timestamp < firstReceiveTime[account] + gracePeriod) {
            return; // Within grace period
        }

        uint256 maxHolding = getMaxHoldingAmount(totalSupply);
        if (balance > maxHolding) {
            revert HoldingExceedsLimit(balance, maxHolding);
        }
    }

    /**
     * @notice Records first receive time for an address (called by token contract)
     * @param account Address receiving tokens for the first time
     */
    function recordFirstReceive(address account) external onlyTokenContract {
        if (firstReceiveTime[account] == 0) {
            firstReceiveTime[account] = block.timestamp;
        }
    }

    // ==================== VIEW FUNCTIONS ====================

    /**
     * @notice Gets maximum transfer amount (0.5% of total supply)
     * @param totalSupply Current total supply
     * @return Maximum transfer amount
     */
    function getMaxTransferAmount(uint256 totalSupply) public view override returns (uint256) {
        uint256 percentage = overridesActive && transferLimitOverride > 0 
            ? transferLimitOverride 
            : MAX_TRANSFER_PERCENTAGE;
            
        return (totalSupply * percentage) / BASIS_POINTS;
    }

    /**
     * @notice Gets maximum holding amount (1.5% of total supply)
     * @param totalSupply Current total supply
     * @return Maximum holding amount
     */
    function getMaxHoldingAmount(uint256 totalSupply) public view override returns (uint256) {
        uint256 percentage = overridesActive && holdingLimitOverride > 0 
            ? holdingLimitOverride 
            : MAX_HOLDING_PERCENTAGE;
            
        return (totalSupply * percentage) / BASIS_POINTS;
    }

    /**
     * @notice Gets current limit percentages
     * @return holdingPercent Current holding limit percentage (basis points)
     * @return transferPercent Current transfer limit percentage (basis points)
     */
    function getCurrentLimits() external view returns (uint256 holdingPercent, uint256 transferPercent) {
        holdingPercent = overridesActive && holdingLimitOverride > 0 
            ? holdingLimitOverride 
            : MAX_HOLDING_PERCENTAGE;
            
        transferPercent = overridesActive && transferLimitOverride > 0 
            ? transferLimitOverride 
            : MAX_TRANSFER_PERCENTAGE;
    }

    /**
     * @notice Checks if address is within grace period
     * @param account Address to check
     * @return Whether address is within grace period
     */
    function isWithinGracePeriod(address account) external view returns (bool) {
        if (firstReceiveTime[account] == 0) return false;
        return block.timestamp < firstReceiveTime[account] + gracePeriod;
    }

    /**
     * @notice Gets time remaining in grace period
     * @param account Address to check
     * @return Seconds remaining in grace period (0 if expired or never received)
     */
    function getGracePeriodRemaining(address account) external view returns (uint256) {
        if (firstReceiveTime[account] == 0) return 0;
        
        uint256 graceEnd = firstReceiveTime[account] + gracePeriod;
        if (block.timestamp >= graceEnd) return 0;
        
        return graceEnd - block.timestamp;
    }

    // ==================== GOVERNANCE FUNCTIONS ====================

    /**
     * @notice Sets exemption status for an address
     * @param account Address to update
     * @param exempt Whether to exempt from anti-whale rules
     */
    function setExempt(address account, bool exempt) 
        external 
        override 
        onlyRole(GOVERNANCE_ROLE) 
        validAddress(account)
    {
        isExempt[account] = exempt;
        emit ExemptionUpdated(account, exempt);
    }

    /**
     * @notice Sets exemption status for multiple addresses
     * @param accounts Array of addresses to update
     * @param exempt Whether to exempt from anti-whale rules
     */
    function setExemptBatch(address[] calldata accounts, bool exempt) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
    {
        for (uint256 i = 0; i < accounts.length; i++) {
            if (accounts[i] != address(0)) {
                isExempt[accounts[i]] = exempt;
                emit ExemptionUpdated(accounts[i], exempt);
            }
        }
    }

    /**
     * @notice Sets override limits for holding and transfer percentages
     * @param holdingLimit New holding limit percentage (basis points, 0 = use default)
     * @param transferLimit New transfer limit percentage (basis points, 0 = use default)
     * @param active Whether to activate the overrides
     */
    function setLimitOverrides(
        uint256 holdingLimit, 
        uint256 transferLimit, 
        bool active
    ) external onlyRole(GOVERNANCE_ROLE) {
        if (holdingLimit > BASIS_POINTS || transferLimit > BASIS_POINTS) {
            revert InvalidPercentage(holdingLimit > BASIS_POINTS ? holdingLimit : transferLimit);
        }

        holdingLimitOverride = holdingLimit;
        transferLimitOverride = transferLimit;
        overridesActive = active;

        emit LimitsOverridden(holdingLimit, transferLimit, active);
    }

    /**
     * @notice Updates the grace period for new token holders
     * @param newGracePeriod New grace period in seconds
     */
    function setGracePeriod(uint256 newGracePeriod) external onlyRole(GOVERNANCE_ROLE) {
        gracePeriod = newGracePeriod;
        emit GracePeriodUpdated(newGracePeriod);
    }

    /**
     * @notice Emergency function to disable all anti-whale protections
     */
    function emergencyDisableProtections() external onlyRole(GOVERNANCE_ROLE) {
        holdingLimitOverride = BASIS_POINTS; // 100% = no limit
        transferLimitOverride = BASIS_POINTS; // 100% = no limit
        overridesActive = true;
        
        emit LimitsOverridden(BASIS_POINTS, BASIS_POINTS, true);
    }

    /**
     * @notice Re-enables default anti-whale protections
     */
    function enableDefaultProtections() external onlyRole(GOVERNANCE_ROLE) {
        overridesActive = false;
        emit LimitsOverridden(0, 0, false);
    }

    /**
     * @notice Updates token contract address (governance only)
     * @param newTokenContract New token contract address
     */
    function updateTokenContract(address newTokenContract) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
        validAddress(newTokenContract)
    {
        // Revoke role from old contract and grant to new one
        address[] memory members = new address[](getRoleMemberCount(TOKEN_CONTRACT_ROLE));
        for (uint256 i = 0; i < members.length; i++) {
            members[i] = getRoleMember(TOKEN_CONTRACT_ROLE, i);
        }
        
        for (uint256 i = 0; i < members.length; i++) {
            _revokeRole(TOKEN_CONTRACT_ROLE, members[i]);
        }
        
        _grantRole(TOKEN_CONTRACT_ROLE, newTokenContract);
    }

    // ==================== UTILITY FUNCTIONS ====================

    /**
     * @notice Calculates whale percentage for a given amount vs total supply
     * @param amount Amount to check
     * @param totalSupply Current total supply
     * @return Percentage in basis points
     */
    function calculateWhalePercentage(uint256 amount, uint256 totalSupply) 
        external 
        pure 
        returns (uint256) 
    {
        if (totalSupply == 0) return 0;
        return (amount * BASIS_POINTS) / totalSupply;
    }

    /**
     * @notice Simulates anti-whale checks for given parameters
     * @param account Account to simulate
     * @param transferAmount Transfer amount
     * @param resultingBalance Resulting balance after transfer
     * @param totalSupply Current total supply
     * @return canTransfer Whether transfer would be allowed
     * @return canHold Whether resulting balance would be allowed
     * @return reason Reason if not allowed
     */
    function simulateAntiWhaleCheck(
        address account,
        uint256 transferAmount,
        uint256 resultingBalance,
        uint256 totalSupply
    ) external view returns (
        bool canTransfer,
        bool canHold,
        string memory reason
    ) {
        // Check exemption
        if (isExempt[account]) {
            return (true, true, "Exempt address");
        }

        // Check transfer limit
        uint256 maxTransfer = getMaxTransferAmount(totalSupply);
        if (transferAmount > maxTransfer) {
            return (false, true, "Transfer exceeds limit");
        }

        // Check holding limit (considering grace period)
        bool inGracePeriod = firstReceiveTime[account] != 0 && 
            block.timestamp < firstReceiveTime[account] + gracePeriod;
            
        if (!inGracePeriod) {
            uint256 maxHolding = getMaxHoldingAmount(totalSupply);
            if (resultingBalance > maxHolding) {
                return (true, false, "Holding would exceed limit");
            }
        }

        return (true, true, "Transfer allowed");
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
} 