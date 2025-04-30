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
exports.Wallet = void 0;
var ethers_1 = require("ethers");
/**
 * Wallet for signing transactions
 */
var Wallet = /** @class */ (function () {
    /**
     * Create a wallet from a private key
     * @param privateKey Private key or mnemonic
     */
    function Wallet(privateKey) {
        this.signer = new ethers_1.ethers.Wallet(privateKey);
    }
    /**
     * Get wallet address
     */
    Wallet.prototype.getAddress = function () {
        return this.signer.address;
    };
    /**
     * Sign a transaction
     * @param transaction Transaction object
     * @returns Signed transaction
     */
    Wallet.prototype.signTransaction = function (transaction) {
        return __awaiter(this, void 0, void 0, function () {
            var tx, signedTx;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        tx = {
                            to: transaction.to,
                            data: transaction.data,
                            value: transaction.value,
                            gasLimit: transaction.gasLimit,
                            gasPrice: transaction.gasPrice,
                            nonce: transaction.nonce,
                        };
                        return [4 /*yield*/, this.signer.signTransaction(tx)];
                    case 1:
                        signedTx = _a.sent();
                        return [2 /*return*/, {
                                hash: ethers_1.ethers.keccak256(signedTx),
                                from: this.signer.address,
                                serialized: signedTx,
                            }];
                }
            });
        });
    };
    /**
     * Sign a message
     * @param message Message to sign
     * @returns Signature
     */
    Wallet.prototype.signMessage = function (message) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                if (typeof message === 'string') {
                    return [2 /*return*/, this.signer.signMessage(message)];
                }
                else {
                    return [2 /*return*/, this.signer.signMessage(message)];
                }
                return [2 /*return*/];
            });
        });
    };
    /**
     * Create a random wallet
     * @returns New wallet instance
     */
    Wallet.createRandom = function () {
        var randomWallet = ethers_1.ethers.Wallet.createRandom();
        return new Wallet(randomWallet.privateKey);
    };
    /**
     * Create a wallet from a mnemonic phrase
     * @param mnemonic Mnemonic phrase
     * @returns Wallet instance
     */
    Wallet.fromMnemonic = function (mnemonic) {
        var wallet = ethers_1.ethers.Wallet.fromPhrase(mnemonic);
        return new Wallet(wallet.privateKey);
    };
    /**
     * Create a wallet from a JSON keystore
     * @param json JSON keystore
     * @param password Password to decrypt
     * @returns Promise resolving to wallet
     */
    Wallet.fromJson = function (json, password) {
        return __awaiter(this, void 0, void 0, function () {
            var wallet;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, ethers_1.ethers.Wallet.fromEncryptedJson(json, password)];
                    case 1:
                        wallet = _a.sent();
                        return [2 /*return*/, new Wallet(wallet.privateKey)];
                }
            });
        });
    };
    /**
     * Export wallet to encrypted JSON
     * @param password Password to encrypt with
     * @returns Promise resolving to JSON string
     */
    Wallet.prototype.toJson = function (password) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                return [2 /*return*/, this.signer.encryptSync(password)];
            });
        });
    };
    return Wallet;
}());
exports.Wallet = Wallet;
