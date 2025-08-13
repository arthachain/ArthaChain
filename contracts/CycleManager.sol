// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "./interfaces/ICycleManager.sol";

/**
 * @title CycleManager
 * @notice Manages emission cycles for ArthaCoin with 3-year cycles and 5% increases
 * @dev Handles the complex emission schedule: 50M → 52.5M → 55.125M etc., capped at 129.093M after year 30
 * 
 * Emission Schedule:
 * - Cycle 0 (Years 1-3): 50,000,000 ARTHA
 * - Cycle 1 (Years 4-6): 52,500,000 ARTHA (+5%)
 * - Cycle 2 (Years 7-9): 55,125,000 ARTHA (+5%)
 * - ...continues with 5% increases until year 30
 * - After year 30: Fixed 129,093,000 ARTHA per cycle
 */
contract CycleManager is 
    Initializable,
    AccessControlUpgradeable, 
    UUPSUpgradeable,
    ICycleManager 
{
    // ==================== ROLES ====================
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
    bytes32 public constant TOKEN_CONTRACT_ROLE = keccak256("TOKEN_CONTRACT_ROLE");

    // ==================== CONSTANTS ====================
    
    /// @notice Initial emission amount for cycle 0 (50 million ARTHA)
    uint256 public constant INITIAL_EMISSION = 50_000_000 * 1e18;
    
    /// @notice Cycle length in seconds (3 years)
    uint256 public constant CYCLE_LENGTH = 3 * 365 days;
    
    /// @notice Increase rate per cycle (5% = 500 basis points)
    uint256 public constant INCREASE_RATE = 500; // 5% in basis points
    
    /// @notice Maximum emission amount after year 30 (129.093 million ARTHA)
    uint256 public constant MAX_EMISSION = 129_093_000 * 1e18;
    
    /// @notice Cycle when max emission is reached (cycle 10 = year 30)
    uint256 public constant MAX_EMISSION_CYCLE = 10;

    // ==================== STATE VARIABLES ====================
    
    /// @notice Deployment timestamp (start of cycle 0)
    uint256 public immutable deploymentTime;
    
    /// @notice Current emission cycle (0-based)
    uint256 public currentCycle;
    
    /// @notice Timestamp when current cycle was last minted
    uint256 public lastMintTime;
    
    /// @notice Mapping of cycle number to emission amount
    mapping(uint256 => uint256) public cycleEmissions;
    
    /// @notice Whether each cycle has been minted
    mapping(uint256 => bool) public cycleMinted;

    // ==================== EVENTS ====================
    event CycleCompleted(uint256 indexed cycle, uint256 emissionAmount, uint256 timestamp);
    event EmissionCalculated(uint256 indexed cycle, uint256 emissionAmount);
    event CycleSkipped(uint256 indexed cycle, string reason);

    // ==================== CUSTOM ERRORS ====================
    error CycleNotReady(uint256 currentTime, uint256 nextMintTime);
    error CycleAlreadyMinted(uint256 cycle);
    error InvalidCycle(uint256 cycle);
    error OnlyTokenContract();

    // ==================== MODIFIERS ====================
    modifier onlyTokenContract() {
        if (!hasRole(TOKEN_CONTRACT_ROLE, msg.sender)) {
            revert OnlyTokenContract();
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
     * @notice Initializes the CycleManager contract
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

        // Pre-calculate all emission amounts for gas efficiency
        _precalculateEmissions();
    }

    // ==================== CORE FUNCTIONS ====================

    /**
     * @notice Gets current cycle information
     * @return cycle Current emission cycle number
     * @return emissionAmount Amount of tokens to be minted this cycle
     * @return canMint Whether minting is currently allowed
     */
    function getCurrentCycleInfo() external view override returns (uint256 cycle, uint256 emissionAmount, bool canMint) {
        cycle = _getCurrentCycleNumber();
        emissionAmount = cycleEmissions[cycle];
        canMint = _canMintCycle(cycle);
    }

    /**
     * @notice Completes the current cycle (called by token contract after minting)
     */
    function completeCycle() external override onlyTokenContract {
        uint256 cycle = _getCurrentCycleNumber();
        
        if (cycleMinted[cycle]) {
            revert CycleAlreadyMinted(cycle);
        }

        if (!_canMintCycle(cycle)) {
            revert CycleNotReady(block.timestamp, _getCycleStartTime(cycle));
        }

        cycleMinted[cycle] = true;
        lastMintTime = block.timestamp;
        currentCycle = cycle;

        emit CycleCompleted(cycle, cycleEmissions[cycle], block.timestamp);
    }

    /**
     * @notice Gets the next cycle emission amount
     * @return Next cycle's emission amount in wei
     */
    function getNextEmissionAmount() external view override returns (uint256) {
        uint256 nextCycle = _getCurrentCycleNumber() + 1;
        return cycleEmissions[nextCycle];
    }

    /**
     * @notice Gets time until next cycle
     * @return Seconds until next cycle can be minted
     */
    function getTimeUntilNextCycle() external view override returns (uint256) {
        uint256 currentCycleNum = _getCurrentCycleNumber();
        uint256 nextCycleStart = _getCycleStartTime(currentCycleNum + 1);
        
        if (block.timestamp >= nextCycleStart) {
            return 0;
        }
        
        return nextCycleStart - block.timestamp;
    }

    // ==================== VIEW FUNCTIONS ====================

    /**
     * @notice Gets emission amount for a specific cycle
     * @param cycle Cycle number (0-based)
     * @return Emission amount for that cycle
     */
    function getEmissionForCycle(uint256 cycle) external view returns (uint256) {
        return cycleEmissions[cycle];
    }

    /**
     * @notice Checks if a cycle has been minted
     * @param cycle Cycle number to check
     * @return Whether the cycle has been minted
     */
    function isCycleMinted(uint256 cycle) external view returns (bool) {
        return cycleMinted[cycle];
    }

    /**
     * @notice Gets the current cycle number based on time elapsed
     * @return Current cycle number (0-based)
     */
    function getCurrentCycleNumber() external view returns (uint256) {
        return _getCurrentCycleNumber();
    }

    /**
     * @notice Gets the start time for a specific cycle
     * @param cycle Cycle number
     * @return Timestamp when the cycle starts
     */
    function getCycleStartTime(uint256 cycle) external view returns (uint256) {
        return _getCycleStartTime(cycle);
    }

    /**
     * @notice Gets years since deployment
     * @return Years elapsed since deployment
     */
    function getYearsSinceDeployment() external view returns (uint256) {
        return (block.timestamp - deploymentTime) / 365 days;
    }

    // ==================== GOVERNANCE FUNCTIONS ====================

    /**
     * @notice Emergency function to mark a cycle as minted (governance only)
     * @param cycle Cycle to mark as minted
     */
    function emergencyMarkCycleMinted(uint256 cycle) external onlyRole(GOVERNANCE_ROLE) {
        cycleMinted[cycle] = true;
        emit CycleSkipped(cycle, "Emergency governance action");
    }

    /**
     * @notice Updates token contract address (governance only)
     * @param newTokenContract New token contract address
     */
    function updateTokenContract(address newTokenContract) external onlyRole(GOVERNANCE_ROLE) {
        _revokeRole(TOKEN_CONTRACT_ROLE, getRoleMember(TOKEN_CONTRACT_ROLE, 0));
        _grantRole(TOKEN_CONTRACT_ROLE, newTokenContract);
    }

    // ==================== INTERNAL FUNCTIONS ====================

    /**
     * @notice Pre-calculates emission amounts for all cycles for gas efficiency
     */
    function _precalculateEmissions() internal {
        uint256 emission = INITIAL_EMISSION;
        
        // Calculate emissions for cycles 0-10 (years 1-33)
        for (uint256 i = 0; i <= MAX_EMISSION_CYCLE; i++) {
            cycleEmissions[i] = emission;
            emit EmissionCalculated(i, emission);
            
            // Increase by 5% for next cycle, but don't exceed max
            if (i < MAX_EMISSION_CYCLE) {
                emission = (emission * (10000 + INCREASE_RATE)) / 10000;
                if (emission > MAX_EMISSION) {
                    emission = MAX_EMISSION;
                }
            }
        }
        
        // Set max emission for all subsequent cycles
        for (uint256 i = MAX_EMISSION_CYCLE + 1; i <= 50; i++) { // Pre-calculate up to cycle 50 (150 years)
            cycleEmissions[i] = MAX_EMISSION;
            emit EmissionCalculated(i, MAX_EMISSION);
        }
    }

    /**
     * @notice Gets current cycle number based on time elapsed
     * @return Current cycle number
     */
    function _getCurrentCycleNumber() internal view returns (uint256) {
        return (block.timestamp - deploymentTime) / CYCLE_LENGTH;
    }

    /**
     * @notice Gets the start time for a specific cycle
     * @param cycle Cycle number
     * @return Timestamp when the cycle starts
     */
    function _getCycleStartTime(uint256 cycle) internal view returns (uint256) {
        return deploymentTime + (cycle * CYCLE_LENGTH);
    }

    /**
     * @notice Checks if a cycle can be minted
     * @param cycle Cycle number to check
     * @return Whether the cycle can be minted
     */
    function _canMintCycle(uint256 cycle) internal view returns (bool) {
        // Cycle must not have been minted yet
        if (cycleMinted[cycle]) {
            return false;
        }
        
        // Must be at or past the cycle start time
        if (block.timestamp < _getCycleStartTime(cycle)) {
            return false;
        }
        
        // Must be the current or a past cycle (don't allow minting future cycles)
        if (cycle > _getCurrentCycleNumber()) {
            return false;
        }
        
        return true;
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