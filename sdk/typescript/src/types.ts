/**
 * JSON-RPC response structure
 */
export interface RpcResponse<T> {
  jsonrpc: string;
  id: number;
  result?: T;
  error?: {
    code: number;
    message: string;
  };
}

/**
 * Block information
 */
export interface BlockInfo {
  number: string;
  hash: string;
  parentHash: string;
  timestamp: string;
  transactions: string[];
  size: string;
  gasUsed: string;
  gasLimit: string;
}

/**
 * Transaction information
 */
export interface TransactionInfo {
  hash: string;
  blockHash: string;
  blockNumber: string;
  from: string;
  to: string | null;
  value: string;
  gasPrice: string;
  gas: string;
  input: string;
  nonce: string;
}

/**
 * Contract receipt after deployment or call
 */
export interface ContractReceipt {
  blockHash: string;
  blockNumber: string;
  contractAddress: string | null;
  from: string;
  to: string | null;
  gasUsed: string;
  status: string;
  transactionHash: string;
  logs: ContractLog[];
}

/**
 * Contract log/event
 */
export interface ContractLog {
  address: string;
  topics: string[];
  data: string;
  blockNumber: string;
  transactionHash: string;
  logIndex: string;
}

/**
 * Signed transaction ready to send
 */
export interface SignedTransaction {
  hash: string;
  from: string;
  serialized: string;
  raw?: string;
}

/**
 * Contract metadata
 */
export interface ContractMetadata {
  name: string;
  version: string;
  author: string;
  hash: string;
  functions: FunctionMetadata[];
}

/**
 * Function metadata
 */
export interface FunctionMetadata {
  name: string;
  inputs: ParameterMetadata[];
  outputs: ParameterMetadata[];
  isView: boolean;
  isPayable: boolean;
}

/**
 * Parameter metadata
 */
export interface ParameterMetadata {
  name: string;
  typeName: string;
}

/**
 * Transaction options
 */
export interface TransactionOptions {
  gasLimit?: number;
  gasPrice?: string;
  nonce?: number;
  value?: string;
  data?: string;
}

/**
 * Contract parameter for encoding
 */
export interface ContractParameter {
  name: string;
  type: string;
  value: any;
}

export interface BlockchainConfig {
  networkUrl: string;
  chainId: number;
  contractAddresses?: {
    [key: string]: string;
  };
}

export interface TransactionResponse {
  hash: string;
  blockNumber?: number;
  confirmations: number;
  from: string;
  to?: string;
  data: string;
  value: string;
  gasLimit: string;
  gasPrice: string;
  timestamp?: number;
}

export interface Block {
  hash: string;
  parentHash: string;
  number: number;
  timestamp: number;
  transactions: string[];
}

export interface Balance {
  address: string;
  balance: string;
  tokenBalances?: {
    [tokenAddress: string]: string;
  };
}

export interface ContractEventFilter {
  address?: string;
  topics?: Array<string | string[] | null>;
  fromBlock?: number | string;
  toBlock?: number | string;
}

export interface ContractEvent {
  address: string;
  blockNumber: number;
  transactionHash: string;
  logIndex: number;
  event: string;
  args: any;
}

export interface ContractMethod {
  name: string;
  inputs: {
    name: string;
    type: string;
    indexed?: boolean;
  }[];
  outputs?: {
    name: string;
    type: string;
  }[];
  stateMutability: 'pure' | 'view' | 'nonpayable' | 'payable';
  type: 'function' | 'event' | 'constructor' | 'fallback';
}

export interface ContractInterface {
  name?: string;
  address: string;
  abi: ContractMethod[];
} 