import { ethers } from 'ethers';
import { SignedTransaction } from './types';

/**
 * Wallet for signing transactions
 */
export class Wallet {
  private signer: ethers.Wallet;

  /**
   * Create a wallet from a private key
   * @param privateKey Private key or mnemonic
   */
  constructor(privateKey: string) {
    this.signer = new ethers.Wallet(privateKey);
  }

  /**
   * Get wallet address
   */
  getAddress(): string {
    return this.signer.address;
  }

  /**
   * Sign a transaction
   * @param transaction Transaction object
   * @returns Signed transaction
   */
  async signTransaction(transaction: any): Promise<SignedTransaction> {
    // Convert to proper transaction format
    const tx: ethers.TransactionRequest = {
      to: transaction.to,
      data: transaction.data,
      value: transaction.value,
      gasLimit: transaction.gasLimit,
      gasPrice: transaction.gasPrice,
      nonce: transaction.nonce,
    };

    // Sign transaction
    const signedTx = await this.signer.signTransaction(tx);
    
    return {
      hash: ethers.keccak256(signedTx),
      from: this.signer.address,
      serialized: signedTx,
    };
  }

  /**
   * Sign a message
   * @param message Message to sign
   * @returns Signature
   */
  async signMessage(message: string | Uint8Array): Promise<string> {
    if (typeof message === 'string') {
      return this.signer.signMessage(message);
    } else {
      return this.signer.signMessage(message);
    }
  }

  /**
   * Create a random wallet
   * @returns New wallet instance
   */
  static createRandom(): Wallet {
    const randomWallet = ethers.Wallet.createRandom();
    return new Wallet(randomWallet.privateKey);
  }

  /**
   * Create a wallet from a mnemonic phrase
   * @param mnemonic Mnemonic phrase
   * @returns Wallet instance
   */
  static fromMnemonic(mnemonic: string): Wallet {
    const wallet = ethers.Wallet.fromPhrase(mnemonic);
    return new Wallet(wallet.privateKey);
  }

  /**
   * Create a wallet from a JSON keystore
   * @param json JSON keystore
   * @param password Password to decrypt
   * @returns Promise resolving to wallet
   */
  static async fromJson(json: string, password: string): Promise<Wallet> {
    const wallet = await ethers.Wallet.fromEncryptedJson(json, password);
    return new Wallet(wallet.privateKey);
  }

  /**
   * Export wallet to encrypted JSON
   * @param password Password to encrypt with
   * @returns Promise resolving to JSON string
   */
  async toJson(password: string): Promise<string> {
    return this.signer.encryptSync(password);
  }
} 