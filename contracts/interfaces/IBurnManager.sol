// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IBurnManager
 * @notice Interface for managing burn rates over time
 */
interface IBurnManager {
    /**
     * @notice Gets current burn rate based on time elapsed
     * @return Current burn rate in basis points (e.g., 4000 = 40%)
     */
    function getCurrentBurnRate() external view returns (uint256);
    
    /**
     * @notice Gets burn rate for a specific year
     * @param year Year since deployment (0-based)
     * @return Burn rate for that year in basis points
     */
    function getBurnRateForYear(uint256 year) external pure returns (uint256);
    
    /**
     * @notice Gets years since deployment
     * @return Number of years elapsed since deployment
     */
    function getYearsSinceDeployment() external view returns (uint256);
} 