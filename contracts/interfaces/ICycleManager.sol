// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ICycleManager
 * @notice Interface for managing emission cycles
 */
interface ICycleManager {
    /**
     * @notice Gets current cycle information
     * @return cycle Current emission cycle number
     * @return emissionAmount Amount of tokens to be minted this cycle
     * @return canMint Whether minting is currently allowed
     */
    function getCurrentCycleInfo() external view returns (uint256 cycle, uint256 emissionAmount, bool canMint);
    
    /**
     * @notice Completes the current cycle
     */
    function completeCycle() external;
    
    /**
     * @notice Gets the next cycle emission amount
     * @return Next cycle's emission amount
     */
    function getNextEmissionAmount() external view returns (uint256);
    
    /**
     * @notice Gets time until next cycle
     * @return Seconds until next cycle can be minted
     */
    function getTimeUntilNextCycle() external view returns (uint256);
} 