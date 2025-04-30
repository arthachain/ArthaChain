import 'package:dio/dio.dart';
import 'package:web3dart/web3dart.dart' hide Wallet;
import 'package:http/http.dart' as http;

import 'contract.dart';
import 'wallet.dart';
import 'models/models.dart';

/// Main client for interacting with the blockchain
class BlockchainClient {
  /// HTTP client for web3
  final Web3Client _web3client;

  /// HTTP client for general API calls
  final Dio _dio;

  /// RPC endpoint URL
  final String endpoint;

  /// Optional wallet for signing transactions
  Wallet? _wallet;

  /// Request ID counter
  int _requestId = 1;

  /// Create a new client with the given endpoint
  BlockchainClient(this.endpoint)
      : _dio = Dio(BaseOptions(
          baseUrl: endpoint,
          headers: {'Content-Type': 'application/json'},
          responseType: ResponseType.json,
        )),
        _web3client = Web3Client(endpoint, http.Client());

  /// Set a wallet for signing transactions
  void setWallet(Wallet wallet) {
    _wallet = wallet;
  }

  /// Get the current wallet
  Wallet? get wallet => _wallet;

  /// Create a contract instance
  Contract contract(String address) {
    return Contract('[]', address, _web3client, wallet: _wallet);
  }

  /// Deploy a new contract
  Future<ContractReceipt> deployContract(
    List<int> bytecode, {
    List<int>? args,
    int gasLimit = 10000000,
  }) async {
    if (_wallet == null) {
      throw Exception('Wallet required for contract deployment');
    }

    // Send transaction - using the wallet's sendTransaction method
    final txHash = await _wallet!.sendTransaction(
      to: '0x0000000000000000000000000000000000000000', // Zero address for contract creation
      amount: BigInt.zero,
      gasLimit: BigInt.from(gasLimit),
    );

    // Get transaction receipt
    final response = await _rpcCall<Map<String, dynamic>>(
      'eth_getTransactionReceipt',
      [txHash],
    );

    return ContractReceipt.fromJson(response);
  }

  /// Send a signed transaction
  Future<ContractReceipt> sendTransaction(SignedTransaction tx) async {
    final response = await _rpcCall<Map<String, dynamic>>(
      'wasm_sendRawTransaction',
      [tx.serialized],
    );

    return ContractReceipt.fromJson(response);
  }

  /// Get transaction by hash
  Future<TransactionInfo> getTransaction(String txHash) async {
    final response = await _rpcCall<Map<String, dynamic>>(
      'eth_getTransactionByHash',
      [txHash],
    );

    return TransactionInfo.fromJson(response);
  }

  /// Get latest block
  Future<BlockInfo> getLatestBlock() async {
    final response = await _rpcCall<Map<String, dynamic>>(
      'eth_getBlockByNumber',
      ['latest', false],
    );

    return BlockInfo.fromJson(response);
  }

  /// Get block by hash
  Future<BlockInfo> getBlockByHash(String blockHash) async {
    final response = await _rpcCall<Map<String, dynamic>>(
      'eth_getBlockByHash',
      [blockHash, false],
    );

    return BlockInfo.fromJson(response);
  }

  /// Get block by number
  Future<BlockInfo> getBlockByNumber(int blockNumber) async {
    final hexBlockNumber = '0x${blockNumber.toRadixString(16)}';

    final response = await _rpcCall<Map<String, dynamic>>(
      'eth_getBlockByNumber',
      [hexBlockNumber, false],
    );

    return BlockInfo.fromJson(response);
  }

  /// Get account balance
  Future<BigInt> getBalance(String address) async {
    final response = await _rpcCall<String>(
      'eth_getBalance',
      [address, 'latest'],
    );

    return BigInt.parse(response.substring(2), radix: 16);
  }

  /// Get account nonce
  Future<int> getNonce(String address) async {
    final response = await _rpcCall<String>(
      'eth_getTransactionCount',
      [address, 'latest'],
    );

    return int.parse(response.substring(2), radix: 16);
  }

  /// Make a raw JSON-RPC call
  Future<T> _rpcCall<T>(String method, List<dynamic> params) async {
    final response = await _dio.post(
      '',
      data: {
        'jsonrpc': '2.0',
        'id': _requestId++,
        'method': method,
        'params': params,
      },
    );

    final data = response.data as Map<String, dynamic>;

    if (data.containsKey('error')) {
      final error = data['error'] as Map<String, dynamic>;
      throw Exception('RPC error (${error['code']}): ${error['message']}');
    }

    return data['result'] as T;
  }

  /// Dispose resources
  void dispose() {
    _web3client.dispose();
  }
}
