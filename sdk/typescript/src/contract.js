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
exports.Contract = void 0;
var ethers_1 = require("ethers");
/**
 * WASM Contract interface
 */
var Contract = /** @class */ (function () {
    /**
     * Create a contract instance
     * @param client Blockchain client
     * @param address Contract address
     */
    function Contract(client, address) {
        this.functions = new Map();
        this.client = client;
        this.address = address;
    }
    /**
     * Get contract address
     */
    Contract.prototype.getAddress = function () {
        return this.address;
    };
    /**
     * Initialize contract metadata
     * Fetches contract metadata if not already loaded
     */
    Contract.prototype.init = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!!this.metadata) return [3 /*break*/, 2];
                        return [4 /*yield*/, this.loadMetadata()];
                    case 1:
                        _a.sent();
                        _a.label = 2;
                    case 2: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Load contract metadata
     */
    Contract.prototype.loadMetadata = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _a, _i, _b, func;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        _a = this;
                        return [4 /*yield*/, this.client.rpcCall('wasm_getContractMetadata', [this.address])];
                    case 1:
                        _a.metadata = _c.sent();
                        // Index functions by name for quick lookup
                        for (_i = 0, _b = this.metadata.functions; _i < _b.length; _i++) {
                            func = _b[_i];
                            this.functions.set(func.name, func);
                        }
                        return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Call a view function (read-only, doesn't modify state)
     * @param functionName Function name
     * @param args Function arguments
     * @returns Function result
     */
    Contract.prototype.callView = function (functionName) {
        var args = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            args[_i - 1] = arguments[_i];
        }
        return __awaiter(this, void 0, void 0, function () {
            var func, encodedArgs, resultHex;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.init()];
                    case 1:
                        _a.sent();
                        func = this.functions.get(functionName);
                        if (!func) {
                            throw new Error("Function not found: ".concat(functionName));
                        }
                        if (!func.isView) {
                            throw new Error("Function is not a view function: ".concat(functionName));
                        }
                        encodedArgs = this.encodeArgs(func, args);
                        return [4 /*yield*/, this.client.rpcCall('wasm_callReadOnlyFunction', [this.address, functionName, encodedArgs])];
                    case 2:
                        resultHex = _a.sent();
                        // Decode result
                        return [2 /*return*/, this.decodeResult(resultHex, func.outputs)];
                }
            });
        });
    };
    /**
     * Call a function that modifies state (sends a transaction)
     * @param functionName Function name
     * @param args Function arguments
     * @param options Transaction options
     * @returns Transaction receipt
     */
    Contract.prototype.call = function (functionName_1) {
        return __awaiter(this, arguments, void 0, function (functionName, args, options) {
            var wallet, func, encodedArgs, data, tx, signedTx;
            if (args === void 0) { args = []; }
            if (options === void 0) { options = {}; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.init()];
                    case 1:
                        _a.sent();
                        wallet = this.client.getWallet();
                        if (!wallet) {
                            throw new Error('Wallet required for contract calls');
                        }
                        func = this.functions.get(functionName);
                        if (!func) {
                            throw new Error("Function not found: ".concat(functionName));
                        }
                        // If payable function and no value provided, ensure value is 0
                        if (func.isPayable && options.value === undefined) {
                            options.value = "0";
                        }
                        // If not payable but value provided, throw error
                        if (!func.isPayable && options.value && options.value !== "0") {
                            throw new Error("Cannot send value to non-payable function: ".concat(functionName));
                        }
                        encodedArgs = this.encodeArgs(func, args);
                        data = this.buildCallData(functionName, encodedArgs);
                        tx = {
                            to: this.address,
                            data: data,
                            value: options.value !== undefined ? this.toBigIntHex(options.value) : '0x0',
                            gasLimit: options.gasLimit || 1000000,
                            gasPrice: options.gasPrice !== undefined ? this.toBigIntHex(options.gasPrice) : undefined,
                            nonce: options.nonce,
                        };
                        return [4 /*yield*/, wallet.signTransaction(tx)];
                    case 2:
                        signedTx = _a.sent();
                        return [4 /*yield*/, this.client.sendTransaction(signedTx)];
                    case 3: return [2 /*return*/, _a.sent()];
                }
            });
        });
    };
    /**
     * Estimate gas for a contract call
     * @param functionName Function name
     * @param args Function arguments
     * @param value Amount of tokens to send with call
     * @returns Estimated gas amount
     */
    Contract.prototype.estimateGas = function (functionName_1) {
        return __awaiter(this, arguments, void 0, function (functionName, args, value) {
            var func, encodedArgs, gasHex;
            if (args === void 0) { args = []; }
            if (value === void 0) { value = 0; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.init()];
                    case 1:
                        _a.sent();
                        func = this.functions.get(functionName);
                        if (!func) {
                            throw new Error("Function not found: ".concat(functionName));
                        }
                        encodedArgs = this.encodeArgs(func, args);
                        return [4 /*yield*/, this.client.rpcCall('wasm_estimateGas', [this.address, functionName, encodedArgs, this.toBigIntHex(value)])];
                    case 2:
                        gasHex = _a.sent();
                        return [2 /*return*/, BigInt(gasHex)];
                }
            });
        });
    };
    /**
     * Get contract events
     * @param eventName Optional event name to filter by
     * @param options Event filter options
     * @returns Filtered events
     */
    Contract.prototype.getEvents = function (eventName_1) {
        return __awaiter(this, arguments, void 0, function (eventName, options) {
            var events;
            if (options === void 0) { options = {}; }
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.client.rpcCall('wasm_getContractEvents', [
                            this.address,
                            eventName || null,
                            options.fromBlock !== undefined ? this.toBlockTag(options.fromBlock) : 'earliest',
                            options.toBlock !== undefined ? this.toBlockTag(options.toBlock) : 'latest',
                            options.limit || 1000
                        ])];
                    case 1:
                        events = _a.sent();
                        return [2 /*return*/, events];
                }
            });
        });
    };
    /**
     * Get contract metadata
     * @returns Contract metadata
     */
    Contract.prototype.getMetadata = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.init()];
                    case 1:
                        _a.sent();
                        return [2 /*return*/, this.metadata];
                }
            });
        });
    };
    /**
     * Encode function arguments based on the function's input parameters
     * @param func Function metadata
     * @param args Arguments to encode
     * @returns Hex-encoded arguments
     */
    Contract.prototype.encodeArgs = function (func, args) {
        if (args.length !== func.inputs.length) {
            throw new Error("Expected ".concat(func.inputs.length, " arguments, got ").concat(args.length));
        }
        // For a full implementation, this would use a proper ABI encoder
        // For now, we'll just JSON encode the arguments and convert to hex
        var jsonArgs = JSON.stringify(args);
        return "0x".concat(Buffer.from(jsonArgs).toString('hex'));
    };
    /**
     * Decode function result based on the function's output parameters
     * @param resultHex Hex-encoded result
     * @param outputs Output parameter metadata
     * @returns Decoded result
     */
    Contract.prototype.decodeResult = function (resultHex, outputs) {
        // For a full implementation, this would use a proper ABI decoder
        // For now, we'll just decode the hex and parse as JSON
        if (!resultHex || resultHex === '0x') {
            return null;
        }
        var hexString = resultHex.startsWith('0x') ? resultHex.slice(2) : resultHex;
        var jsonString = Buffer.from(hexString, 'hex').toString('utf8');
        try {
            return JSON.parse(jsonString);
        }
        catch (e) {
            return jsonString;
        }
    };
    /**
     * Build call data for a function call
     * @param functionName Function name
     * @param encodedArgs Encoded arguments
     * @returns Hex-encoded call data
     */
    Contract.prototype.buildCallData = function (functionName, encodedArgs) {
        // For a full implementation, this would properly encode the function selector and arguments
        // For now, we'll just concatenate the function name and arguments with a separator
        var funcSelector = "0x".concat(Buffer.from(functionName).toString('hex'));
        return "".concat(funcSelector).concat(encodedArgs.slice(2));
    };
    /**
     * Convert a value to a hex string
     * @param value Value to convert
     * @returns Hex string
     */
    Contract.prototype.toBigIntHex = function (value) {
        return ethers_1.ethers.toBeHex(value);
    };
    /**
     * Convert a block number to a block tag
     * @param blockNumber Block number
     * @returns Block tag
     */
    Contract.prototype.toBlockTag = function (blockNumber) {
        return blockNumber === 0 ? 'earliest' : ethers_1.ethers.toBeHex(blockNumber);
    };
    return Contract;
}());
exports.Contract = Contract;
