// Type definitions for blockchain explorer

/**
 * @typedef {Object} Block
 * @property {string} hash - Block hash
 * @property {number} number - Block number
 * @property {string} timestamp - Block timestamp
 * @property {number} transactionCount - Number of transactions in block
 * @property {string} miner - Block miner/validator address
 * @property {string} gasUsed - Gas used in block
 * @property {string} gasLimit - Gas limit for block
 * @property {string} size - Block size in bytes
 * @property {Transaction[]} transactions - Array of transactions in block
 */

/**
 * @typedef {Object} Transaction
 * @property {string} hash - Transaction hash
 * @property {string} blockHash - Hash of containing block
 * @property {number} blockNumber - Number of containing block
 * @property {string} from - Sender address
 * @property {string} to - Recipient address
 * @property {string} value - Transaction value in wei
 * @property {string} gas - Gas limit
 * @property {string} gasPrice - Gas price
 * @property {string} gasUsed - Gas used
 * @property {string} status - Transaction status (success/failed)
 * @property {string} timestamp - Transaction timestamp
 * @property {string} input - Transaction input data
 * @property {number} nonce - Transaction nonce
 * @property {number} transactionIndex - Index in block
 */

/**
 * @typedef {Object} Address
 * @property {string} address - The address
 * @property {string} balance - Address balance in wei
 * @property {number} transactionCount - Number of transactions
 * @property {string} type - Address type (EOA/Contract)
 * @property {string} label - Optional address label
 * @property {Token[]} tokens - Token holdings
 */

/**
 * @typedef {Object} Token
 * @property {string} address - Token contract address
 * @property {string} name - Token name
 * @property {string} symbol - Token symbol
 * @property {number} decimals - Token decimals
 * @property {string} totalSupply - Total token supply
 * @property {string} balance - Token balance for address
 * @property {string} value - USD value of balance
 */

/**
 * @typedef {Object} SmartContract
 * @property {string} address - Contract address
 * @property {string} name - Contract name
 * @property {string} sourceCode - Contract source code
 * @property {string} abi - Contract ABI
 * @property {boolean} verified - Whether contract is verified
 * @property {string} compiler - Compiler version used
 * @property {string} creationTx - Creation transaction hash
 */

/**
 * @typedef {Object} NetworkStats
 * @property {number} blockHeight - Current block height
 * @property {string} hashRate - Network hash rate
 * @property {string} difficulty - Current difficulty
 * @property {number} avgBlockTime - Average block time in seconds
 * @property {number} pendingTxCount - Pending transaction count
 * @property {string} totalSupply - Total token supply
 * @property {number} activeAddresses - Number of active addresses
 */

/**
 * @typedef {Object} SearchResult
 * @property {string} type - Result type (block/transaction/address/token)
 * @property {string} value - Search result value
 * @property {string} label - Display label
 * @property {Object} data - Additional result data
 */

export {};

