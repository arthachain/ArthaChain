"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.BlockchainClient = void 0;
var axios_1 = require("axios");
var ethers_1 = require("ethers");
var contract_1 = require("./contract");
var wallet_1 = require("./wallet");
/**
 * Main client for interacting with the blockchain
 */
var BlockchainClient = /** @class */ (function () {
    /**
     * Create a new blockchain client
     * @param endpoint RPC endpoint URL
     */
    function BlockchainClient(endpoint) {
        this.nextId = 1;
        this.endpoint = endpoint;
        this.http = axios_1.default.create({
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
    BlockchainClient.prototype.withWallet = function (wallet) {
        if (typeof wallet === 'string') {
            this.wallet = new wallet_1.Wallet(wallet);
        }
        else {
            this.wallet = wallet;
        }
        return this;
    };
    /**
     * Get the current wallet
     */
    BlockchainClient.prototype.getWallet = function () {
        return this.wallet;
    };
    /**
     * Create a contract instance
     * @param address Contract address
     * @returns Contract instance
     */
    BlockchainClient.prototype.contract = function (address) {
        return new contract_1.Contract(this, address);
    };
    /**
     * Deploy a new contract
     * @param bytecode Contract bytecode
     * @param args Constructor arguments
     * @param gasLimit Gas limit for deployment
     * @returns Contract receipt with address
     */
    BlockchainClient.prototype.deployContract = function (bytecode_1, args_1) {
        return __awaiter(this, arguments, void 0, function (bytecode, args, gasLimit) {
            var bytecodeHex, argsHex, tx, signedTx;
            if (gasLimit === void 0) { gasLimit = 10000000; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!this.wallet) {
                            throw new Error('Wallet required for contract deployment');
                        }
                        bytecodeHex = typeof bytecode === 'string'
                            ? bytecode.startsWith('0x') ? bytecode : "0x".concat(bytecode)
                            : "0x".concat(Buffer.from(bytecode).toString('hex'));
                        argsHex = !args ? '0x' : (typeof args === 'string'
                            ? args.startsWith('0x') ? args : "0x".concat(args)
                            : "0x".concat(Buffer.from(args).toString('hex')));
                        tx = {
                            to: null,
                            data: bytecodeHex,
                            args: argsHex,
                            gasLimit: gasLimit,
                        };
                        return [4 /*yield*/, this.wallet.signTransaction(tx)];
                    case 1:
                        signedTx = _a.sent();
                        // Send transaction
                        return [2 /*return*/, this.sendTransaction(signedTx)];
                }
            });
        });
    };
    /**
     * Send a signed transaction
     * @param transaction Signed transaction
     * @returns Transaction receipt
     */
    BlockchainClient.prototype.sendTransaction = function (transaction) {
        return __awaiter(this, void 0, void 0, function () {
            var response, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.http.post('', {
                            jsonrpc: '2.0',
                            id: this.nextId++,
                            method: 'wasm_sendRawTransaction',
                            params: [transaction.serialized]
                        })];
                    case 1:
                        response = _a.sent();
                        data = response.data;
                        if (data.error) {
                            throw new Error("RPC error (".concat(data.error.code, "): ").concat(data.error.message));
                        }
                        return [2 /*return*/, data.result];
                }
            });
        });
    };
    /**
     * Get transaction by hash
     * @param txHash Transaction hash
     * @returns Transaction info
     */
    BlockchainClient.prototype.getTransaction = function (txHash) {
        return __awaiter(this, void 0, void 0, function () {
            var response, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.http.post('', {
                            jsonrpc: '2.0',
                            id: this.nextId++,
                            method: 'eth_getTransactionByHash',
                            params: [txHash]
                        })];
                    case 1:
                        response = _a.sent();
                        data = response.data;
                        if (data.error) {
                            throw new Error("RPC error (".concat(data.error.code, "): ").concat(data.error.message));
                        }
                        return [2 /*return*/, data.result];
                }
            });
        });
    };
    /**
     * Get latest block
     * @returns Block info
     */
    BlockchainClient.prototype.getLatestBlock = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.http.post('', {
                            jsonrpc: '2.0',
                            id: this.nextId++,
                            method: 'eth_getBlockByNumber',
                            params: ['latest', false]
                        })];
                    case 1:
                        response = _a.sent();
                        data = response.data;
                        if (data.error) {
                            throw new Error("RPC error (".concat(data.error.code, "): ").concat(data.error.message));
                        }
                        return [2 /*return*/, data.result];
                }
            });
        });
    };
    /**
     * Get block by hash
     * @param blockHash Block hash
     * @returns Block info
     */
    BlockchainClient.prototype.getBlockByHash = function (blockHash) {
        return __awaiter(this, void 0, void 0, function () {
            var response, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.http.post('', {
                            jsonrpc: '2.0',
                            id: this.nextId++,
                            method: 'eth_getBlockByHash',
                            params: [blockHash, false]
                        })];
                    case 1:
                        response = _a.sent();
                        data = response.data;
                        if (data.error) {
                            throw new Error("RPC error (".concat(data.error.code, "): ").concat(data.error.message));
                        }
                        return [2 /*return*/, data.result];
                }
            });
        });
    };
    /**
     * Get block by number
     * @param blockNumber Block number
     * @returns Block info
     */
    BlockchainClient.prototype.getBlockByNumber = function (blockNumber) {
        return __awaiter(this, void 0, void 0, function () {
            var hexBlockNumber, response, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        hexBlockNumber = ethers_1.ethers.toBeHex(blockNumber);
                        return [4 /*yield*/, this.http.post('', {
                                jsonrpc: '2.0',
                                id: this.nextId++,
                                method: 'eth_getBlockByNumber',
                                params: [hexBlockNumber, false]
                            })];
                    case 1:
                        response = _a.sent();
                        data = response.data;
                        if (data.error) {
                            throw new Error("RPC error (".concat(data.error.code, "): ").concat(data.error.message));
                        }
                        return [2 /*return*/, data.result];
                }
            });
        });
    };
    /**
     * Get account balance
     * @param address Account address
     * @returns Balance in smallest denomination
     */
    BlockchainClient.prototype.getBalance = function (address) {
        return __awaiter(this, void 0, void 0, function () {
            var response, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.http.post('', {
                            jsonrpc: '2.0',
                            id: this.nextId++,
                            method: 'eth_getBalance',
                            params: [address, 'latest']
                        })];
                    case 1:
                        response = _a.sent();
                        data = response.data;
                        if (data.error) {
                            throw new Error("RPC error (".concat(data.error.code, "): ").concat(data.error.message));
                        }
                        return [2 /*return*/, BigInt(data.result)];
                }
            });
        });
    };
    /**
     * Get account nonce
     * @param address Account address
     * @returns Nonce
     */
    BlockchainClient.prototype.getNonce = function (address) {
        return __awaiter(this, void 0, void 0, function () {
            var response, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.http.post('', {
                            jsonrpc: '2.0',
                            id: this.nextId++,
                            method: 'eth_getTransactionCount',
                            params: [address, 'latest']
                        })];
                    case 1:
                        response = _a.sent();
                        data = response.data;
                        if (data.error) {
                            throw new Error("RPC error (".concat(data.error.code, "): ").concat(data.error.message));
                        }
                        return [2 /*return*/, parseInt(data.result, 16)];
                }
            });
        });
    };
    /**
     * Make a raw JSON-RPC call
     * @param method RPC method name
     * @param params RPC parameters
     * @returns RPC result
     */
    BlockchainClient.prototype.rpcCall = function (method, params) {
        return __awaiter(this, void 0, void 0, function () {
            var response, data;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.http.post('', {
                            jsonrpc: '2.0',
                            id: this.nextId++,
                            method: method,
                            params: params
                        })];
                    case 1:
                        response = _a.sent();
                        data = response.data;
                        if (data.error) {
                            throw new Error("RPC error (".concat(data.error.code, "): ").concat(data.error.message));
                        }
                        return [2 /*return*/, data.result];
                }
            });
        });
    };
    return BlockchainClient;
}());
exports.BlockchainClient = BlockchainClient;
