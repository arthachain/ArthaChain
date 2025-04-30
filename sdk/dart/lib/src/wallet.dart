import 'dart:typed_data';
import 'package:hex/hex.dart';
import 'package:web3dart/web3dart.dart';
import 'package:bip39/bip39.dart' as bip39;
import 'package:http/http.dart';
import 'package:blockchain_sdk/src/constants.dart';
import 'dart:math';

/// A class representing an Ethereum wallet for blockchain interactions
class Wallet {
  /// The Ethereum credentials
  final EthPrivateKey _credentials;

  /// Web3 client for interacting with the blockchain
  final Web3Client _client;

  /// The wallet's Ethereum address
  String get address => _credentials.address.hex;

  /// The private key as a hex string
  String get privateKey => HEX.encode(_credentials.privateKey);

  /// Constructor with private key and RPC URL
  Wallet(String privateKey, String rpcUrl)
      : _credentials = EthPrivateKey.fromHex(privateKey),
        _client = Web3Client(rpcUrl, Client());

  /// Creates a new random wallet with connection to the specified RPC URL
  factory Wallet.createRandom(String rpcUrl) {
    final random = EthPrivateKey.createRandom(Random.secure());
    return Wallet(HEX.encode(random.privateKey), rpcUrl);
  }

  /// Creates a wallet from a mnemonic phrase
  factory Wallet.fromMnemonic(String mnemonic, String rpcUrl) {
    if (!bip39.validateMnemonic(mnemonic)) {
      throw Exception('Invalid mnemonic phrase');
    }

    final seed = bip39.mnemonicToSeed(mnemonic);
    // Simplified for now - in a real implementation, you would use a proper HD wallet derivation
    final privateKey = HEX.encode(seed.sublist(0, 32));
    return Wallet(privateKey, rpcUrl);
  }

  /// Generates a new random mnemonic phrase
  static String generateMnemonic() {
    return bip39.generateMnemonic(strength: 128); // 12 words
  }

  /// Get the ETH balance of the wallet
  Future<EtherAmount> getBalance() async {
    return await _client.getBalance(_credentials.address);
  }

  /// Send ETH to another address
  Future<String> sendTransaction({
    required String to,
    required BigInt amount,
    BigInt? gasPrice,
    BigInt? gasLimit,
    int? nonce,
  }) async {
    final transaction = Transaction(
      to: EthereumAddress.fromHex(to),
      value: EtherAmount.fromBigInt(EtherUnit.wei, amount),
      gasPrice: gasPrice != null
          ? EtherAmount.fromBigInt(EtherUnit.wei, gasPrice)
          : null,
      maxGas: gasLimit?.toInt() ?? BlockchainConstants.DEFAULT_GAS_LIMIT,
      nonce: nonce,
    );

    final txHash = await _client.sendTransaction(
      _credentials,
      transaction,
      chainId: null, // Chain ID should be specified for production
    );

    return txHash;
  }

  /// Sign a message with the private key
  Future<Uint8List> signMessage(Uint8List message) async {
    return _credentials.signPersonalMessageToUint8List(message);
  }

  /// Close the wallet and release resources
  void close() {
    _client.dispose();
  }
}
