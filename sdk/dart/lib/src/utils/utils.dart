import 'dart:convert';
import 'dart:typed_data';
import 'package:web3dart/web3dart.dart';
import 'package:convert/convert.dart';
import 'package:hex/hex.dart';

/// Validates if the provided string is a valid Ethereum address
bool isValidAddress(String address) {
  if (!address.startsWith('0x')) return false;
  if (address.length != 42) return false;

  try {
    EthereumAddress.fromHex(address);
    return true;
  } catch (_) {
    return false;
  }
}

/// Converts wei to ether
String weiToEther(BigInt wei) {
  return EtherAmount.fromBigInt(EtherUnit.wei, wei)
      .getValueInUnit(EtherUnit.ether)
      .toString();
}

/// Converts ether to wei
BigInt etherToWei(String ether) {
  return EtherAmount.fromBase10String(EtherUnit.ether, ether).getInWei;
}

/// Normalizes a transaction hash by ensuring it has the 0x prefix
String normalizeHash(String hash) {
  return hash.startsWith('0x') ? hash : '0x$hash';
}

/// Shortens an address for display purposes
String shortenAddress(String address, [int chars = 4]) {
  if (address.isEmpty) return '';
  final normalized = normalizeHash(address);
  return '${normalized.substring(0, chars + 2)}...${normalized.substring(42 - chars)}';
}

/// Adds a delay (sleep) using futures
Future<void> delay(int ms) {
  return Future.delayed(Duration(milliseconds: ms));
}

/// Retry a function with exponential backoff
Future<T> retry<T>(
  Future<T> Function() fn, {
  int maxRetries = 5,
  int initialDelay = 500,
}) async {
  late final dynamic lastError;
  int retryCount = 0;
  int delayMs = initialDelay;

  while (retryCount < maxRetries) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      retryCount++;

      if (retryCount >= maxRetries) {
        break;
      }

      await delay(delayMs);
      delayMs = (delayMs * 1.5).toInt(); // Exponential backoff
    }
  }

  throw lastError;
}

/// Converts a hex string to a UTF-8 string
String hexToUtf8(String hex) {
  final bytes = hexToBytes(hex);
  return utf8.decode(bytes);
}

/// Converts a UTF-8 string to a hex string
String utf8ToHex(String str) {
  return '0x${hex.encode(utf8.encode(str))}';
}

/// Converts hex string to bytes
Uint8List hexToBytes(String hex) {
  String cleanHex = hex.startsWith('0x') ? hex.substring(2) : hex;
  // Ensure even length
  if (cleanHex.length % 2 != 0) {
    cleanHex = '0$cleanHex';
  }
  return Uint8List.fromList(HEX.decode(cleanHex));
}

/// Converts bytes to hex string
String bytesToHex(Uint8List bytes, {bool include0x = true}) {
  return include0x ? '0x${HEX.encode(bytes)}' : HEX.encode(bytes);
}
