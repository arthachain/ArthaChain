// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "./interfaces/IBurnManager.sol";

/**
 * @title BurnManager
 * @notice Manages progressive burn rates for ArthaCoin over time
 * @dev Implements increasing burn rates from 40% to 96% over 17+ years
 * 
 * Burn Rate Schedule:
 * - Years 1-2: 40%
 * - Years 3-4: 47%
 * - Years 5-6: 54%
 * - Years 7-8: 61%
 * - Years 9-10: 68%
 * - Years 11-12: 75%
 * - Years 13-14: 82%
 * - Years 15-16: 89%
 * - Year 17+: 96%
 */
contract BurnManager is 
    Initializable,
    AccessControlUpgradeable, 
    UUPSUpgradeable,
    IBurnManager 
{
    // ==================== ROLES ====================
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");

    // ==================== STATE VARIABLES ====================
    
    /// @notice Deployment timestamp for calculating time-based burn rates
    uint256 public immutable deploymentTime;
    
    /// @notice Array of burn rates for each 2-year period (in basis points)
    /// Index 0 = Years 1-2, Index 1 = Years 3-4, etc.
    uint256[9] public burnRateSchedule;
    
    /// @notice Emergency burn rate override (0 = use schedule, >0 = override)
    uint256 public emergencyBurnRateOverride;
    
    /// @notice Whether emergency override is active
    bool public emergencyOverrideActive;

    // ==================== EVENTS ====================
    event BurnRateCalculated(uint256 year, uint256 burnRate);
    event EmergencyBurnRateSet(uint256 newRate, bool active);
    event BurnRateScheduleUpdated(uint256[9] newSchedule);

    // ==================== CUSTOM ERRORS ====================
    error InvalidBurnRate(uint256 rate);
    error InvalidYear(uint256 year);

    // ==================== CONSTRUCTOR & INITIALIZER ====================
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        deploymentTime = block.timestamp;
        _disableInitializers();
    }

    /**
     * @notice Initializes the BurnManager contract
     * @param _admin Address to receive admin roles
     */
    function initialize(address _admin) public initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();

        _grantRole(DEFAULT_ADMIN_ROLE, _admin);
        _grantRole(GOVERNANCE_ROLE, _admin);
        _grantRole(UPGRADER_ROLE, _admin);

        // Initialize burn rate schedule (in basis points)
        burnRateSchedule[0] = 4000; // Years 1-2: 40%
        burnRateSchedule[1] = 4700; // Years 3-4: 47%
        burnRateSchedule[2] = 5400; // Years 5-6: 54%
        burnRateSchedule[3] = 6100; // Years 7-8: 61%
        burnRateSchedule[4] = 6800; // Years 9-10: 68%
        burnRateSchedule[5] = 7500; // Years 11-12: 75%
        burnRateSchedule[6] = 8200; // Years 13-14: 82%
        burnRateSchedule[7] = 8900; // Years 15-16: 89%
        burnRateSchedule[8] = 9600; // Year 17+: 96%

        emit BurnRateScheduleUpdated(burnRateSchedule);
    }

    // ==================== CORE FUNCTIONS ====================

    /**
     * @notice Gets current burn rate based on time elapsed since deployment
     * @return Current burn rate in basis points (e.g., 4000 = 40%)
     */
    function getCurrentBurnRate() external view override returns (uint256) {
        // Check for emergency override first
        if (emergencyOverrideActive) {
            return emergencyBurnRateOverride;
        }

        uint256 yearsSinceDeployment = getYearsSinceDeployment();
        return getBurnRateForYear(yearsSinceDeployment);
    }

    /**
     * @notice Gets burn rate for a specific year since deployment
     * @param year Year since deployment (0-based, so year 0 = first year)
     * @return Burn rate for that year in basis points
     */
    function getBurnRateForYear(uint256 year) public view override returns (uint256) {
        // Calculate which 2-year period this year falls into
        uint256 periodIndex = year / 2;
        
        // If beyond our schedule, use the maximum burn rate (96%)
        if (periodIndex >= burnRateSchedule.length) {
            return burnRateSchedule[burnRateSchedule.length - 1]; // 96%
        }
        
        return burnRateSchedule[periodIndex];
    }

    /**
     * @notice Gets years since deployment
     * @return Number of years elapsed since deployment (0-based)
     */
    function getYearsSinceDeployment() public view override returns (uint256) {
        return (block.timestamp - deploymentTime) / 365 days;
    }

    // ==================== VIEW FUNCTIONS ====================

    /**
     * @notice Gets the complete burn rate schedule
     * @return Array of burn rates for each 2-year period
     */
    function getBurnRateSchedule() external view returns (uint256[9] memory) {
        return burnRateSchedule;
    }

    /**
     * @notice Gets burn rate for next year
     * @return Burn rate that will be active next year
     */
    function getNextYearBurnRate() external view returns (uint256) {
        uint256 nextYear = getYearsSinceDeployment() + 1;
        return getBurnRateForYear(nextYear);
    }

    /**
     * @notice Gets burn rate progression for next N years
     * @param yearsAhead Number of years to look ahead
     * @return Array of burn rates for the next N years
     */
    function getBurnRateProgression(uint256 yearsAhead) external view returns (uint256[] memory) {
        uint256[] memory progression = new uint256[](yearsAhead);
        uint256 currentYear = getYearsSinceDeployment();
        
        for (uint256 i = 0; i < yearsAhead; i++) {
            progression[i] = getBurnRateForYear(currentYear + i + 1);
        }
        
        return progression;
    }

    /**
     * @notice Gets detailed information about current burn state
     * @return year Current year since deployment
     * @return burnRate Current burn rate in basis points
     * @return nextBurnRate Next period's burn rate
     * @return yearsUntilNextIncrease Years until next burn rate increase
     */
    function getBurnRateInfo() external view returns (
        uint256 year,
        uint256 burnRate,
        uint256 nextBurnRate,
        uint256 yearsUntilNextIncrease
    ) {
        year = getYearsSinceDeployment();
        burnRate = getBurnRateForYear(year);
        
        // Calculate next burn rate
        uint256 currentPeriod = year / 2;
        uint256 nextPeriod = currentPeriod + 1;
        
        if (nextPeriod >= burnRateSchedule.length) {
            nextBurnRate = burnRateSchedule[burnRateSchedule.length - 1];
            yearsUntilNextIncrease = 0; // No more increases
        } else {
            nextBurnRate = burnRateSchedule[nextPeriod];
            yearsUntilNextIncrease = ((currentPeriod + 1) * 2) - year;
        }
    }

    // ==================== GOVERNANCE FUNCTIONS ====================

    /**
     * @notice Sets emergency burn rate override (governance only)
     * @param newRate New burn rate in basis points (0-10000)
     * @param active Whether to activate the override
     */
    function setEmergencyBurnRate(uint256 newRate, bool active) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
    {
        if (newRate > 10000) {
            revert InvalidBurnRate(newRate);
        }

        emergencyBurnRateOverride = newRate;
        emergencyOverrideActive = active;

        emit EmergencyBurnRateSet(newRate, active);
    }

    /**
     * @notice Updates the burn rate schedule (governance only)
     * @param newSchedule New burn rate schedule (9 values for 9 periods)
     */
    function updateBurnRateSchedule(uint256[9] calldata newSchedule) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
    {
        // Validate that rates are in ascending order and within bounds
        for (uint256 i = 0; i < newSchedule.length; i++) {
            if (newSchedule[i] > 10000) {
                revert InvalidBurnRate(newSchedule[i]);
            }
            
            // Ensure rates are non-decreasing (except for emergency cases)
            if (i > 0 && newSchedule[i] < newSchedule[i - 1]) {
                revert InvalidBurnRate(newSchedule[i]);
            }
        }

        burnRateSchedule = newSchedule;
        emit BurnRateScheduleUpdated(newSchedule);
    }

    /**
     * @notice Disables emergency burn rate override (governance only)
     */
    function disableEmergencyOverride() external onlyRole(GOVERNANCE_ROLE) {
        emergencyOverrideActive = false;
        emit EmergencyBurnRateSet(0, false);
    }

    // ==================== UTILITY FUNCTIONS ====================

    /**
     * @notice Calculates burn amount for a given transfer amount
     * @param transferAmount Amount being transferred
     * @return Burn amount based on current burn rate
     */
    function calculateBurnAmount(uint256 transferAmount) external view returns (uint256) {
        uint256 burnRate = emergencyOverrideActive ? 
            emergencyBurnRateOverride : 
            getBurnRateForYear(getYearsSinceDeployment());
            
        return (transferAmount * burnRate) / 10000;
    }

    /**
     * @notice Simulates burn amounts for different transfer sizes
     * @param transferAmounts Array of transfer amounts to simulate
     * @return Array of corresponding burn amounts
     */
    function simulateBurnAmounts(uint256[] calldata transferAmounts) 
        external 
        view 
        returns (uint256[] memory) 
    {
        uint256[] memory burnAmounts = new uint256[](transferAmounts.length);
        uint256 currentBurnRate = emergencyOverrideActive ? 
            emergencyBurnRateOverride : 
            getBurnRateForYear(getYearsSinceDeployment());
        
        for (uint256 i = 0; i < transferAmounts.length; i++) {
            burnAmounts[i] = (transferAmounts[i] * currentBurnRate) / 10000;
        }
        
        return burnAmounts;
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