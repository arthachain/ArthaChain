/// Constants for the blockchain SDK
class BlockchainConstants {
  /// Default gas limit for transactions
  static const int DEFAULT_GAS_LIMIT = 2000000;

  /// Default gas price in Gwei
  static const int DEFAULT_GAS_PRICE_GWEI = 20;

  /// Default number of confirmations to wait for
  static const int DEFAULT_CONFIRMATIONS = 1;

  /// Maximum number of retry attempts for transactions
  static const int MAX_RETRY_ATTEMPTS = 3;

  /// Delay between retry attempts (milliseconds)
  static const int RETRY_DELAY_MS = 1000;

  /// Ethereum mainnet chain ID
  static const int CHAIN_ID_MAINNET = 1;

  /// Goerli testnet chain ID
  static const int CHAIN_ID_GOERLI = 5;

  /// Sepolia testnet chain ID
  static const int CHAIN_ID_SEPOLIA = 11155111;

  /// Hardhat local chain ID
  static const int CHAIN_ID_HARDHAT = 31337;

  /// Ganache local chain ID
  static const int CHAIN_ID_GANACHE = 1337;

  /// Ethereum mainnet RPC URL
  static const String RPC_URL_MAINNET = 'https://mainnet.infura.io/v3/';

  /// Goerli testnet RPC URL
  static const String RPC_URL_GOERLI = 'https://goerli.infura.io/v3/';

  /// Sepolia testnet RPC URL
  static const String RPC_URL_SEPOLIA = 'https://sepolia.infura.io/v3/';

  /// Default local RPC URL
  static const String RPC_URL_LOCAL = 'http://127.0.0.1:8545';

  /// Default block explorer URL for Ethereum mainnet
  static const String ETHERSCAN_URL = 'https://etherscan.io';

  /// Default block explorer URL for Goerli testnet
  static const String ETHERSCAN_GOERLI_URL = 'https://goerli.etherscan.io';

  /// Default block explorer URL for Sepolia testnet
  static const String ETHERSCAN_SEPOLIA_URL = 'https://sepolia.etherscan.io';

  /// Default network ID (Ethereum mainnet)
  static const int DEFAULT_NETWORK_ID = CHAIN_ID_MAINNET;

  /// Default network name
  static const String DEFAULT_NETWORK_NAME = 'mainnet';

  /// Ethereum minimum transaction value (in wei)
  static const String MIN_TRANSACTION_VALUE = '0';

  /// Wei to Ether conversion factor
  static const int WEI_PER_ETHER = 1000000000000000000;

  /// Gwei to Wei conversion factor
  static const int WEI_PER_GWEI = 1000000000;

  /// Ethereum address length (in bytes)
  static const int ETH_ADDRESS_BYTES_LENGTH = 20;

  /// Ethereum address string length (with '0x' prefix)
  static const int ETH_ADDRESS_STRING_LENGTH = 42;

  /// Ethereum private key length (in bytes)
  static const int PRIVATE_KEY_BYTES_LENGTH = 32;

  /// BIP-39 default word count
  static const int DEFAULT_MNEMONIC_WORD_COUNT = 12;

  /// Ethereum HD wallet path
  static const String HD_PATH = "m/44'/60'/0'/0/0";

  // Prevent instantiation
  BlockchainConstants._();
}
