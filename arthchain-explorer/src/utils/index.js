import { ethers } from 'ethers';
import { format, formatDistanceToNow } from 'date-fns';

/**
 * Format wei value to ETH with specified decimal places
 * @param {string} wei - Wei value as string
 * @param {number} decimals - Number of decimal places (default: 4)
 * @returns {string} Formatted ETH value
 */
export const formatEther = (wei, decimals = 4) => {
  if (!wei) return '0';
  try {
    const ether = ethers.formatEther(wei);
    return parseFloat(ether).toFixed(decimals);
  } catch (error) {
    console.error('Error formatting ether:', error);
    return '0';
  }
};

/**
 * Format wei value to Gwei
 * @param {string} wei - Wei value as string
 * @returns {string} Formatted Gwei value
 */
export const formatGwei = (wei) => {
  if (!wei) return '0';
  try {
    const gwei = ethers.formatUnits(wei, 'gwei');
    return parseFloat(gwei).toFixed(2);
  } catch (error) {
    console.error('Error formatting gwei:', error);
    return '0';
  }
};

/**
 * Truncate address or hash for display
 * @param {string} value - Address or hash to truncate
 * @param {number} startChars - Number of characters to show at start (default: 6)
 * @param {number} endChars - Number of characters to show at end (default: 4)
 * @returns {string} Truncated value
 */
export const truncateHash = (value, startChars = 6, endChars = 4) => {
  if (!value) return '';
  if (value.length <= startChars + endChars) return value;
  return `${value.slice(0, startChars)}...${value.slice(-endChars)}`;
};

/**
 * Format timestamp to readable date
 * @param {number|string} timestamp - Unix timestamp
 * @returns {string} Formatted date string
 */
export const formatDate = (timestamp) => {
  if (!timestamp) return '';
  const date = new Date(parseInt(timestamp) * 1000);
  return format(date, 'MMM dd, yyyy HH:mm:ss');
};

/**
 * Format timestamp to readable date (alias for formatDate)
 * @param {number|string} timestamp - Unix timestamp
 * @returns {string} Formatted date string
 */
export const formatTimestamp = (timestamp) => {
  return formatDate(timestamp);
};

/**
 * Format timestamp to relative time (e.g., "2 hours ago")
 * @param {number|string} timestamp - Unix timestamp
 * @returns {string} Relative time string
 */
export const formatRelativeTime = (timestamp) => {
  if (!timestamp) return '';
  const date = new Date(parseInt(timestamp) * 1000);
  return formatDistanceToNow(date, { addSuffix: true });
};

/**
 * Format large numbers with appropriate suffixes (K, M, B)
 * @param {number} num - Number to format
 * @param {number} decimals - Number of decimal places (default: 1)
 * @returns {string} Formatted number string
 */
export const formatNumber = (num, decimals = 1) => {
  if (!num || num === 0) return '0';
  
  const absNum = Math.abs(num);
  const sign = num < 0 ? '-' : '';
  
  if (absNum >= 1e9) {
    return sign + (absNum / 1e9).toFixed(decimals) + 'B';
  }
  if (absNum >= 1e6) {
    return sign + (absNum / 1e6).toFixed(decimals) + 'M';
  }
  if (absNum >= 1e3) {
    return sign + (absNum / 1e3).toFixed(decimals) + 'K';
  }
  
  return sign + absNum.toLocaleString();
};

/**
 * Validate Ethereum address
 * @param {string} address - Address to validate
 * @returns {boolean} True if valid address
 */
export const isValidAddress = (address) => {
  try {
    return ethers.isAddress(address);
  } catch {
    return false;
  }
};

/**
 * Validate transaction hash
 * @param {string} hash - Hash to validate
 * @returns {boolean} True if valid hash
 */
export const isValidHash = (hash) => {
  return /^0x[a-fA-F0-9]{64}$/.test(hash);
};

/**
 * Validate block number
 * @param {string} blockNumber - Block number to validate
 * @returns {boolean} True if valid block number
 */
export const isValidBlockNumber = (blockNumber) => {
  const num = parseInt(blockNumber);
  return !isNaN(num) && num >= 0;
};

/**
 * Determine search type based on input
 * @param {string} input - Search input
 * @returns {string} Search type (address, transaction, block, or unknown)
 */
export const getSearchType = (input) => {
  if (!input) return 'unknown';
  
  const trimmed = input.trim();
  
  if (isValidAddress(trimmed)) {
    return 'address';
  }
  
  if (isValidHash(trimmed)) {
    return 'transaction';
  }
  
  if (isValidBlockNumber(trimmed)) {
    return 'block';
  }
  
  return 'unknown';
};

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} True if successful
 */
export const copyToClipboard = async (text) => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
};

/**
 * Format bytes to human readable format
 * @param {number} bytes - Number of bytes
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted bytes string
 */
export const formatBytes = (bytes, decimals = 2) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

/**
 * Calculate gas fee in ETH
 * @param {string} gasUsed - Gas used
 * @param {string} gasPrice - Gas price in wei
 * @returns {string} Gas fee in ETH
 */
export const calculateGasFee = (gasUsed, gasPrice) => {
  if (!gasUsed || !gasPrice) return '0';
  
  try {
    const fee = BigInt(gasUsed) * BigInt(gasPrice);
    return ethers.formatEther(fee.toString());
  } catch (error) {
    console.error('Error calculating gas fee:', error);
    return '0';
  }
};

/**
 * Generate random color for charts
 * @returns {string} Hex color string
 */
export const generateRandomColor = () => {
  const colors = [
    '#00ff88', '#8b5cf6', '#fbbf24', '#10b981', '#ef4444',
    '#06b6d4', '#f59e0b', '#84cc16', '#ec4899', '#6366f1'
  ];
  return colors[Math.floor(Math.random() * colors.length)];
};

/**
 * Debounce function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

/**
 * Throttle function
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in milliseconds
 * @returns {Function} Throttled function
 */
export const throttle = (func, limit) => {
  let inThrottle;
  return function executedFunction(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

