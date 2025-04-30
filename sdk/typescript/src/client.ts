import axios, { AxiosInstance } from 'axios';
import { ethers } from 'ethers';
import { Contract } from './contract';
import { Wallet } from './wallet';
import { 
  BlockInfo, 
  TransactionInfo, 
  ContractReceipt, 
  SignedTransaction,
  RpcResponse
} from './types';

/**
 * Main client for interacting with the blockchain
 */
export class BlockchainClient {
  private endpoint: string;
  private http: AxiosInstance;
  private wallet?: Wallet;
  private nextId: number = 1;

  /**
   * Create a new blockchain client
   * @param endpoint RPC endpoint URL
   */
  constructor(endpoint: string) {
    this.endpoint = endpoint;
    this.http = axios.create({
      baseURL: endpoint,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  /**
   * Set a wallet for signing transactions
   * @param wallet Wallet instance or private key
   * @returns Client instance for chaining
   */
  withWallet(wallet: Wallet | string): BlockchainClient {
    if (typeof wallet === 'string') {
      this.wallet = new Wallet(wallet);
    } else {
      this.wallet = wallet;
    }
    return this;
  }

  /**
   * Get the current wallet
   */
  getWallet(): Wallet | undefined {
    return this.wallet;
  }

  /**
   * Create a contract instance
   * @param address Contract address
   * @returns Contract instance
   */
  contract(address: string): Contract {
    return new Contract(this, address);
  }

  /**
   * Deploy a new contract
   * @param bytecode Contract bytecode
   * @param args Constructor arguments
   * @param gasLimit Gas limit for deployment
   * @returns Contract receipt with address
   */
  async deployContract(
    bytecode: string | Uint8Array, 
    args?: string | Uint8Array,
    gasLimit: number = 10000000
  ): Promise<ContractReceipt> {
    if (!this.wallet) {
      throw new Error('Wallet required for contract deployment');
    }

    // Convert bytecode to hex if needed
    const bytecodeHex = typeof bytecode === 'string' 
      ? bytecode.startsWith('0x') ? bytecode : `0x${bytecode}`
      : `0x${Buffer.from(bytecode).toString('hex')}`;

    // Convert args to hex if needed
    const argsHex = !args ? '0x' : (
      typeof args === 'string'
        ? args.startsWith('0x') ? args : `0x${args}`
        : `0x${Buffer.from(args).toString('hex')}`
    );

    // Create transaction for contract deployment
    const tx = {
      to: null,
      data: bytecodeHex,
      args: argsHex,
      gasLimit,
    };

    // Sign transaction
    const signedTx = await this.wallet.signTransaction(tx);

    // Send transaction
    return this.sendTransaction(signedTx);
  }

  /**
   * Send a signed transaction
   * @param transaction Signed transaction
   * @returns Transaction receipt
   */
  async sendTransaction(transaction: SignedTransaction): Promise<ContractReceipt> {
    const response = await this.http.post('', {
      jsonrpc: '2.0',
      id: this.nextId++,
      method: 'wasm_sendRawTransaction',
      params: [transaction.serialized]
    });

    const data = response.data as RpcResponse<ContractReceipt>;
    if (data.error) {
      throw new Error(`RPC error (${data.error.code}): ${data.error.message}`);
    }

    return data.result as ContractReceipt;
  }

  /**
   * Get transaction by hash
   * @param txHash Transaction hash
   * @returns Transaction info
   */
  async getTransaction(txHash: string): Promise<TransactionInfo> {
    const response = await this.http.post('', {
      jsonrpc: '2.0',
      id: this.nextId++,
      method: 'eth_getTransactionByHash',
      params: [txHash]
    });

    const data = response.data as RpcResponse<TransactionInfo>;
    if (data.error) {
      throw new Error(`RPC error (${data.error.code}): ${data.error.message}`);
    }

    return data.result as TransactionInfo;
  }

  /**
   * Get latest block
   * @returns Block info
   */
  async getLatestBlock(): Promise<BlockInfo> {
    const response = await this.http.post('', {
      jsonrpc: '2.0',
      id: this.nextId++,
      method: 'eth_getBlockByNumber',
      params: ['latest', false]
    });

    const data = response.data as RpcResponse<BlockInfo>;
    if (data.error) {
      throw new Error(`RPC error (${data.error.code}): ${data.error.message}`);
    }

    return data.result as BlockInfo;
  }

  /**
   * Get block by hash
   * @param blockHash Block hash
   * @returns Block info
   */
  async getBlockByHash(blockHash: string): Promise<BlockInfo> {
    const response = await this.http.post('', {
      jsonrpc: '2.0',
      id: this.nextId++,
      method: 'eth_getBlockByHash',
      params: [blockHash, false]
    });

    const data = response.data as RpcResponse<BlockInfo>;
    if (data.error) {
      throw new Error(`RPC error (${data.error.code}): ${data.error.message}`);
    }

    return data.result as BlockInfo;
  }

  /**
   * Get block by number
   * @param blockNumber Block number
   * @returns Block info
   */
  async getBlockByNumber(blockNumber: number): Promise<BlockInfo> {
    const hexBlockNumber = ethers.toBeHex(blockNumber);
    
    const response = await this.http.post('', {
      jsonrpc: '2.0',
      id: this.nextId++,
      method: 'eth_getBlockByNumber',
      params: [hexBlockNumber, false]
    });

    const data = response.data as RpcResponse<BlockInfo>;
    if (data.error) {
      throw new Error(`RPC error (${data.error.code}): ${data.error.message}`);
    }

    return data.result as BlockInfo;
  }

  /**
   * Get account balance
   * @param address Account address
   * @returns Balance in smallest denomination
   */
  async getBalance(address: string): Promise<bigint> {
    const response = await this.http.post('', {
      jsonrpc: '2.0',
      id: this.nextId++,
      method: 'eth_getBalance',
      params: [address, 'latest']
    });

    const data = response.data as RpcResponse<string>;
    if (data.error) {
      throw new Error(`RPC error (${data.error.code}): ${data.error.message}`);
    }

    return BigInt(data.result as string);
  }

  /**
   * Get account nonce
   * @param address Account address
   * @returns Nonce
   */
  async getNonce(address: string): Promise<number> {
    const response = await this.http.post('', {
      jsonrpc: '2.0',
      id: this.nextId++,
      method: 'eth_getTransactionCount',
      params: [address, 'latest']
    });

    const data = response.data as RpcResponse<string>;
    if (data.error) {
      throw new Error(`RPC error (${data.error.code}): ${data.error.message}`);
    }

    return parseInt(data.result as string, 16);
  }

  /**
   * Make a raw JSON-RPC call
   * @param method RPC method name
   * @param params RPC parameters
   * @returns RPC result
   */
  async rpcCall<T>(method: string, params: any[]): Promise<T> {
    const response = await this.http.post('', {
      jsonrpc: '2.0',
      id: this.nextId++,
      method,
      params
    });

    const data = response.data as RpcResponse<T>;
    if (data.error) {
      throw new Error(`RPC error (${data.error.code}): ${data.error.message}`);
    }

    return data.result as T;
  }
} 