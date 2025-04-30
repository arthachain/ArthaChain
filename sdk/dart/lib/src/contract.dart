import 'package:web3dart/web3dart.dart' hide Wallet;
import 'wallet.dart';

/// Represents a smart contract interaction wrapper
class Contract {
  /// The contract address
  final EthereumAddress _address;

  /// The Web3 client
  final Web3Client _client;

  /// The wallet to use for transactions
  final Wallet? _wallet;

  /// Cached contract instance
  late final DeployedContract _contract;

  /// Constructor with ABI, address, and client
  Contract(
    String abiJson,
    String address,
    this._client, {
    Wallet? wallet,
    String? contractName,
  })  : _address = EthereumAddress.fromHex(address),
        _wallet = wallet {
    // Create deployed contract instance
    _contract = DeployedContract(
      ContractAbi.fromJson(abiJson, contractName ?? 'SmartContract'),
      _address,
    );
  }

  /// Call a read-only function (view/pure)
  Future<List<dynamic>> call(
    String functionName,
    List<dynamic> params, {
    BlockNum? atBlock,
  }) async {
    final function = _contract.function(functionName);
    return await _client.call(
      contract: _contract,
      function: function,
      params: params,
      atBlock: atBlock,
    );
  }

  /// Send a transaction to write to the contract
  /// Requires a wallet to sign the transaction
  Future<String> send(
    String functionName,
    List<dynamic> params, {
    BigInt? value,
    BigInt? maxGas,
    EtherAmount? gasPrice,
  }) async {
    if (_wallet == null) {
      throw Exception('Wallet is required for sending transactions');
    }

    // Using the sendTransaction method from our Wallet implementation
    return await _wallet!.sendTransaction(
      to: _address.hex,
      amount: value ?? BigInt.zero,
      gasLimit: BigInt.from(maxGas?.toInt() ?? 0),
    );
  }

  /// Get contract events logs
  Future<List<FilterEvent>> getEvents(
    String eventName, {
    List<List<String>>? topics,
    BlockNum? fromBlock,
    BlockNum? toBlock,
  }) async {
    final filter = FilterOptions(
      address: _address,
      topics: topics,
      fromBlock: fromBlock,
      toBlock: toBlock,
    );

    final events = await _client.getLogs(filter);
    return events;
  }

  /// Listen to contract events in real-time
  Stream<FilterEvent> events(
    String eventName, {
    List<List<String>>? topics,
    BlockNum? fromBlock,
  }) {
    final filter = FilterOptions(
      address: _address,
      topics: topics,
      fromBlock: fromBlock,
    );

    return _client.events(filter);
  }

  /// Get the contract address
  String get address => _address.hex;

  /// Helper to convert hex string to bytes
  static List<int> hexToBytes(String hex) {
    if (hex.startsWith('0x')) {
      hex = hex.substring(2);
    }

    if (hex.length % 2 != 0) {
      hex = '0$hex';
    }

    List<int> result = [];
    for (int i = 0; i < hex.length; i += 2) {
      result.add(int.parse(hex.substring(i, i + 2), radix: 16));
    }
    return result;
  }
}
