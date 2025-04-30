import { ethers } from 'ethers';

/**
 * Validates if the provided string is a valid Ethereum address
 * @param address Address to validate
 * @returns True if valid address
 */
export function isValidAddress(address: string): boolean {
  return ethers.isAddress(address);
}

/**
 * Converts wei to ether
 * @param wei Amount in wei
 * @returns Amount in ether
 */
export function weiToEther(wei: string | bigint): string {
  return ethers.formatEther(wei);
}

/**
 * Converts ether to wei
 * @param ether Amount in ether
 * @returns Amount in wei
 */
export function etherToWei(ether: string): bigint {
  return ethers.parseEther(ether);
}

/**
 * Normalizes a transaction hash by ensuring it has the 0x prefix
 * @param hash Transaction hash
 * @returns Normalized hash
 */
export function normalizeHash(hash: string): string {
  return hash.startsWith('0x') ? hash : `0x${hash}`;
}

/**
 * Shortens an address for display purposes
 * @param address Full address
 * @param chars Number of characters to keep at start and end
 * @returns Shortened address
 */
export function shortenAddress(address: string, chars = 4): string {
  if (!address) return '';
  const normalized = normalizeHash(address);
  return `${normalized.substring(0, chars + 2)}...${normalized.substring(42 - chars)}`;
}

/**
 * Adds a delay (sleep) using promises
 * @param ms Milliseconds to wait
 * @returns Promise that resolves after the delay
 */
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Retry a function with exponential backoff
 * @param fn Function to retry
 * @param maxRetries Maximum number of retries
 * @param initialDelay Initial delay in milliseconds
 * @returns Result of the function
 */
export async function retry<T>(
  fn: () => Promise<T>,
  maxRetries = 5,
  initialDelay = 500
): Promise<T> {
  let lastError: Error;
  let retryCount = 0;
  let delayMs = initialDelay;

  while (retryCount < maxRetries) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      retryCount++;
      
      if (retryCount >= maxRetries) {
        break;
      }
      
      await delay(delayMs);
      delayMs *= 1.5; // Exponential backoff
    }
  }
  
  throw lastError!;
}

/**
 * Converts a hex string to a UTF-8 string
 * @param hex Hex string
 * @returns UTF-8 string
 */
export function hexToUtf8(hex: string): string {
  return Buffer.from(hex.startsWith('0x') ? hex.slice(2) : hex, 'hex').toString('utf8');
}

/**
 * Converts a UTF-8 string to a hex string
 * @param str UTF-8 string
 * @returns Hex string
 */
export function utf8ToHex(str: string): string {
  return '0x' + Buffer.from(str, 'utf8').toString('hex');
} 