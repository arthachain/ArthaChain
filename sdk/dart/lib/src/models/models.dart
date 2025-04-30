/// Transaction receipt after sending a transaction
class ContractReceipt {
  /// Block hash
  final String blockHash;

  /// Block number
  final String blockNumber;

  /// Contract address (for contract creation)
  final String? contractAddress;

  /// Sender address
  final String from;

  /// Recipient address
  final String? to;

  /// Gas used
  final String gasUsed;

  /// Transaction status (1 = success, 0 = failure)
  final String status;

  /// Transaction hash
  final String transactionHash;

  /// Transaction logs
  final List<ContractLog> logs;

  /// Create a new contract receipt
  ContractReceipt({
    required this.blockHash,
    required this.blockNumber,
    this.contractAddress,
    required this.from,
    this.to,
    required this.gasUsed,
    required this.status,
    required this.transactionHash,
    required this.logs,
  });

  /// Create a receipt from JSON
  factory ContractReceipt.fromJson(Map<String, dynamic> json) {
    final logs = (json['logs'] as List<dynamic>?)
            ?.map((log) => ContractLog.fromJson(log as Map<String, dynamic>))
            .toList() ??
        [];

    return ContractReceipt(
      blockHash: json['blockHash'] as String,
      blockNumber: json['blockNumber'] as String,
      contractAddress: json['contractAddress'] as String?,
      from: json['from'] as String,
      to: json['to'] as String?,
      gasUsed: json['gasUsed'] as String,
      status: json['status'] as String,
      transactionHash: json['transactionHash'] as String,
      logs: logs,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() => {
        'blockHash': blockHash,
        'blockNumber': blockNumber,
        'contractAddress': contractAddress,
        'from': from,
        'to': to,
        'gasUsed': gasUsed,
        'status': status,
        'transactionHash': transactionHash,
        'logs': logs.map((log) => log.toJson()).toList(),
      };
}

/// Contract log/event
class ContractLog {
  /// Contract address
  final String address;

  /// Event topics
  final List<String> topics;

  /// Event data
  final String data;

  /// Block number
  final String blockNumber;

  /// Transaction hash
  final String transactionHash;

  /// Log index in the block
  final String logIndex;

  /// Create a new contract log
  ContractLog({
    required this.address,
    required this.topics,
    required this.data,
    required this.blockNumber,
    required this.transactionHash,
    required this.logIndex,
  });

  /// Create a log from JSON
  factory ContractLog.fromJson(Map<String, dynamic> json) {
    return ContractLog(
      address: json['address'] as String,
      topics:
          (json['topics'] as List<dynamic>).map((t) => t as String).toList(),
      data: json['data'] as String,
      blockNumber: json['blockNumber'] as String,
      transactionHash: json['transactionHash'] as String,
      logIndex: json['logIndex'] as String,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() => {
        'address': address,
        'topics': topics,
        'data': data,
        'blockNumber': blockNumber,
        'transactionHash': transactionHash,
        'logIndex': logIndex,
      };
}

/// Block information
class BlockInfo {
  /// Block number
  final String number;

  /// Block hash
  final String hash;

  /// Parent block hash
  final String parentHash;

  /// Block timestamp
  final String timestamp;

  /// Transaction hashes in this block
  final List<String> transactions;

  /// Block size in bytes
  final String size;

  /// Gas used
  final String gasUsed;

  /// Gas limit
  final String gasLimit;

  /// Create a new block info
  BlockInfo({
    required this.number,
    required this.hash,
    required this.parentHash,
    required this.timestamp,
    required this.transactions,
    required this.size,
    required this.gasUsed,
    required this.gasLimit,
  });

  /// Create block info from JSON
  factory BlockInfo.fromJson(Map<String, dynamic> json) {
    return BlockInfo(
      number: json['number'] as String,
      hash: json['hash'] as String,
      parentHash: json['parentHash'] as String,
      timestamp: json['timestamp'] as String,
      transactions: (json['transactions'] as List<dynamic>)
          .map((t) => t as String)
          .toList(),
      size: json['size'] as String,
      gasUsed: json['gasUsed'] as String,
      gasLimit: json['gasLimit'] as String,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() => {
        'number': number,
        'hash': hash,
        'parentHash': parentHash,
        'timestamp': timestamp,
        'transactions': transactions,
        'size': size,
        'gasUsed': gasUsed,
        'gasLimit': gasLimit,
      };
}

/// Transaction information
class TransactionInfo {
  /// Transaction hash
  final String hash;

  /// Block hash
  final String blockHash;

  /// Block number
  final String blockNumber;

  /// Sender address
  final String from;

  /// Recipient address (null for contract creation)
  final String? to;

  /// Transaction value
  final String value;

  /// Gas price
  final String gasPrice;

  /// Gas limit
  final String gas;

  /// Transaction data
  final String input;

  /// Transaction nonce
  final String nonce;

  /// Create a new transaction info
  TransactionInfo({
    required this.hash,
    required this.blockHash,
    required this.blockNumber,
    required this.from,
    this.to,
    required this.value,
    required this.gasPrice,
    required this.gas,
    required this.input,
    required this.nonce,
  });

  /// Create transaction info from JSON
  factory TransactionInfo.fromJson(Map<String, dynamic> json) {
    return TransactionInfo(
      hash: json['hash'] as String,
      blockHash: json['blockHash'] as String,
      blockNumber: json['blockNumber'] as String,
      from: json['from'] as String,
      to: json['to'] as String?,
      value: json['value'] as String,
      gasPrice: json['gasPrice'] as String,
      gas: json['gas'] as String,
      input: json['input'] as String,
      nonce: json['nonce'] as String,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() => {
        'hash': hash,
        'blockHash': blockHash,
        'blockNumber': blockNumber,
        'from': from,
        'to': to,
        'value': value,
        'gasPrice': gasPrice,
        'gas': gas,
        'input': input,
        'nonce': nonce,
      };
}

/// Signed transaction ready to send
class SignedTransaction {
  /// Transaction hash
  final String hash;

  /// Sender address
  final String from;

  /// Serialized transaction data
  final String serialized;

  /// Raw transaction data (optional)
  final String? raw;

  /// Create a new signed transaction
  SignedTransaction({
    required this.hash,
    required this.from,
    required this.serialized,
    this.raw,
  });

  /// Create signed transaction from JSON
  factory SignedTransaction.fromJson(Map<String, dynamic> json) {
    return SignedTransaction(
      hash: json['hash'] as String,
      from: json['from'] as String,
      serialized: json['serialized'] as String,
      raw: json['raw'] as String?,
    );
  }

  /// Convert to JSON
  Map<String, dynamic> toJson() => {
        'hash': hash,
        'from': from,
        'serialized': serialized,
        'raw': raw,
      };
}
