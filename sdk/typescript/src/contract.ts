import { ethers } from 'ethers';
import { BlockchainClient } from './client';
import { 
  ContractMetadata, 
  FunctionMetadata, 
  ContractParameter, 
  ContractReceipt,
  TransactionOptions,
} from './types';

/**
 * WASM Contract interface
 */
export class Contract {
  private client: BlockchainClient;
  private address: string;
  private metadata?: ContractMetadata;
  private functions: Map<string, FunctionMetadata> = new Map();

  /**
   * Create a contract instance
   * @param client Blockchain client
   * @param address Contract address
   */
  constructor(client: BlockchainClient, address: string) {
    this.client = client;
    this.address = address;
  }

  /**
   * Get contract address
   */
  getAddress(): string {
    return this.address;
  }

  /**
   * Initialize contract metadata
   * Fetches contract metadata if not already loaded
   */
  async init(): Promise<void> {
    if (!this.metadata) {
      await this.loadMetadata();
    }
  }

  /**
   * Load contract metadata
   */
  private async loadMetadata(): Promise<void> {
    this.metadata = await this.client.rpcCall<ContractMetadata>(
      'wasm_getContractMetadata',
      [this.address]
    );

    // Index functions by name for quick lookup
    for (const func of this.metadata.functions) {
      this.functions.set(func.name, func);
    }
  }

  /**
   * Call a view function (read-only, doesn't modify state)
   * @param functionName Function name
   * @param args Function arguments
   * @returns Function result
   */
  async callView(functionName: string, ...args: any[]): Promise<any> {
    await this.init();

    // Get function metadata
    const func = this.functions.get(functionName);
    if (!func) {
      throw new Error(`Function not found: ${functionName}`);
    }

    if (!func.isView) {
      throw new Error(`Function is not a view function: ${functionName}`);
    }

    // Encode arguments
    const encodedArgs = this.encodeArgs(func, args);

    // Make RPC call
    const resultHex = await this.client.rpcCall<string>(
      'wasm_callReadOnlyFunction',
      [this.address, functionName, encodedArgs]
    );

    // Decode result
    return this.decodeResult(resultHex, func.outputs);
  }

  /**
   * Call a function that modifies state (sends a transaction)
   * @param functionName Function name
   * @param args Function arguments
   * @param options Transaction options
   * @returns Transaction receipt
   */
  async call(functionName: string, args: any[] = [], options: TransactionOptions = {}): Promise<ContractReceipt> {
    await this.init();
    
    const wallet = this.client.getWallet();
    if (!wallet) {
      throw new Error('Wallet required for contract calls');
    }

    // Get function metadata
    const func = this.functions.get(functionName);
    if (!func) {
      throw new Error(`Function not found: ${functionName}`);
    }

    // If payable function and no value provided, ensure value is 0
    if (func.isPayable && options.value === undefined) {
      options.value = "0";
    }

    // If not payable but value provided, throw error
    if (!func.isPayable && options.value && options.value !== "0") {
      throw new Error(`Cannot send value to non-payable function: ${functionName}`);
    }

    // Encode arguments
    const encodedArgs = this.encodeArgs(func, args);

    // Build transaction data
    const data = this.buildCallData(functionName, encodedArgs);

    // Create transaction
    const tx = {
      to: this.address,
      data,
      value: options.value !== undefined ? this.toBigIntHex(options.value) : '0x0',
      gasLimit: options.gasLimit || 1000000,
      gasPrice: options.gasPrice !== undefined ? this.toBigIntHex(options.gasPrice) : undefined,
      nonce: options.nonce,
    };

    // Sign and send transaction
    const signedTx = await wallet.signTransaction(tx);
    return await this.client.sendTransaction(signedTx);
  }

  /**
   * Estimate gas for a contract call
   * @param functionName Function name
   * @param args Function arguments
   * @param value Amount of tokens to send with call
   * @returns Estimated gas amount
   */
  async estimateGas(functionName: string, args: any[] = [], value: number | string | bigint = 0): Promise<bigint> {
    await this.init();

    // Get function metadata
    const func = this.functions.get(functionName);
    if (!func) {
      throw new Error(`Function not found: ${functionName}`);
    }

    // Encode arguments
    const encodedArgs = this.encodeArgs(func, args);

    // Make RPC call
    const gasHex = await this.client.rpcCall<string>(
      'wasm_estimateGas',
      [this.address, functionName, encodedArgs, this.toBigIntHex(value)]
    );

    return BigInt(gasHex);
  }

  /**
   * Get contract events
   * @param eventName Optional event name to filter by
   * @param options Event filter options
   * @returns Filtered events
   */
  async getEvents(eventName?: string, options: { fromBlock?: number, toBlock?: number, limit?: number } = {}): Promise<any[]> {
    const events = await this.client.rpcCall<any[]>(
      'wasm_getContractEvents',
      [
        this.address,
        eventName || null,
        options.fromBlock !== undefined ? this.toBlockTag(options.fromBlock) : 'earliest',
        options.toBlock !== undefined ? this.toBlockTag(options.toBlock) : 'latest',
        options.limit || 1000
      ]
    );

    return events;
  }

  /**
   * Get contract metadata
   * @returns Contract metadata
   */
  async getMetadata(): Promise<ContractMetadata> {
    await this.init();
    return this.metadata!;
  }

  /**
   * Encode function arguments based on the function's input parameters
   * @param func Function metadata
   * @param args Arguments to encode
   * @returns Hex-encoded arguments
   */
  private encodeArgs(func: FunctionMetadata, args: any[]): string {
    if (args.length !== func.inputs.length) {
      throw new Error(`Expected ${func.inputs.length} arguments, got ${args.length}`);
    }

    // For a full implementation, this would use a proper ABI encoder
    // For now, we'll just JSON encode the arguments and convert to hex
    const jsonArgs = JSON.stringify(args);
    return `0x${Buffer.from(jsonArgs).toString('hex')}`;
  }

  /**
   * Decode function result based on the function's output parameters
   * @param resultHex Hex-encoded result
   * @param outputs Output parameter metadata
   * @returns Decoded result
   */
  private decodeResult(resultHex: string, outputs: any[]): any {
    // For a full implementation, this would use a proper ABI decoder
    // For now, we'll just decode the hex and parse as JSON
    if (!resultHex || resultHex === '0x') {
      return null;
    }

    const hexString = resultHex.startsWith('0x') ? resultHex.slice(2) : resultHex;
    const jsonString = Buffer.from(hexString, 'hex').toString('utf8');
    
    try {
      return JSON.parse(jsonString);
    } catch (e) {
      return jsonString;
    }
  }

  /**
   * Build call data for a function call
   * @param functionName Function name
   * @param encodedArgs Encoded arguments
   * @returns Hex-encoded call data
   */
  private buildCallData(functionName: string, encodedArgs: string): string {
    // For a full implementation, this would properly encode the function selector and arguments
    // For now, we'll just concatenate the function name and arguments with a separator
    const funcSelector = `0x${Buffer.from(functionName).toString('hex')}`;
    return `${funcSelector}${encodedArgs.slice(2)}`;
  }

  /**
   * Convert a value to a hex string
   * @param value Value to convert
   * @returns Hex string
   */
  private toBigIntHex(value: number | string | bigint): string {
    return ethers.toBeHex(value);
  }

  /**
   * Convert a block number to a block tag
   * @param blockNumber Block number
   * @returns Block tag
   */
  private toBlockTag(blockNumber: number): string {
    return blockNumber === 0 ? 'earliest' : ethers.toBeHex(blockNumber);
  }
} 