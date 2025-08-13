// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IAntiWhaleManager
 * @notice Interface for anti-whale protection mechanisms
 */
interface IAntiWhaleManager {
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
    ) external view;
    
    /**
     * @notice Gets maximum transfer amount (0.5% of total supply)
     * @param totalSupply Current total supply
     * @return Maximum transfer amount
     */
    function getMaxTransferAmount(uint256 totalSupply) external pure returns (uint256);
    
    /**
     * @notice Gets maximum holding amount (1.5% of total supply)
     * @param totalSupply Current total supply
     * @return Maximum holding amount
     */
    function getMaxHoldingAmount(uint256 totalSupply) external pure returns (uint256);
    
    /**
     * @notice Checks if address is exempt from anti-whale rules
     * @param account Address to check
     * @return Whether address is exempt
     */
    function isExempt(address account) external view returns (bool);
    
    /**
     * @notice Sets exemption status for an address
     * @param account Address to update
     * @param exempt Whether to exempt from anti-whale rules
     */
    function setExempt(address account, bool exempt) external;
} 