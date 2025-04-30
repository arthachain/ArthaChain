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
exports.isValidAddress = isValidAddress;
exports.weiToEther = weiToEther;
exports.etherToWei = etherToWei;
exports.normalizeHash = normalizeHash;
exports.shortenAddress = shortenAddress;
exports.delay = delay;
exports.retry = retry;
exports.hexToUtf8 = hexToUtf8;
exports.utf8ToHex = utf8ToHex;
var ethers_1 = require("ethers");
/**
 * Validates if the provided string is a valid Ethereum address
 * @param address Address to validate
 * @returns True if valid address
 */
function isValidAddress(address) {
    return ethers_1.ethers.isAddress(address);
}
/**
 * Converts wei to ether
 * @param wei Amount in wei
 * @returns Amount in ether
 */
function weiToEther(wei) {
    return ethers_1.ethers.formatEther(wei);
}
/**
 * Converts ether to wei
 * @param ether Amount in ether
 * @returns Amount in wei
 */
function etherToWei(ether) {
    return ethers_1.ethers.parseEther(ether);
}
/**
 * Normalizes a transaction hash by ensuring it has the 0x prefix
 * @param hash Transaction hash
 * @returns Normalized hash
 */
function normalizeHash(hash) {
    return hash.startsWith('0x') ? hash : "0x".concat(hash);
}
/**
 * Shortens an address for display purposes
 * @param address Full address
 * @param chars Number of characters to keep at start and end
 * @returns Shortened address
 */
function shortenAddress(address, chars) {
    if (chars === void 0) { chars = 4; }
    if (!address)
        return '';
    var normalized = normalizeHash(address);
    return "".concat(normalized.substring(0, chars + 2), "...").concat(normalized.substring(42 - chars));
}
/**
 * Adds a delay (sleep) using promises
 * @param ms Milliseconds to wait
 * @returns Promise that resolves after the delay
 */
function delay(ms) {
    return new Promise(function (resolve) { return setTimeout(resolve, ms); });
}
/**
 * Retry a function with exponential backoff
 * @param fn Function to retry
 * @param maxRetries Maximum number of retries
 * @param initialDelay Initial delay in milliseconds
 * @returns Result of the function
 */
function retry(fn_1) {
    return __awaiter(this, arguments, void 0, function (fn, maxRetries, initialDelay) {
        var lastError, retryCount, delayMs, error_1;
        if (maxRetries === void 0) { maxRetries = 5; }
        if (initialDelay === void 0) { initialDelay = 500; }
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    retryCount = 0;
                    delayMs = initialDelay;
                    _a.label = 1;
                case 1:
                    if (!(retryCount < maxRetries)) return [3 /*break*/, 7];
                    _a.label = 2;
                case 2:
                    _a.trys.push([2, 4, , 6]);
                    return [4 /*yield*/, fn()];
                case 3: return [2 /*return*/, _a.sent()];
                case 4:
                    error_1 = _a.sent();
                    lastError = error_1;
                    retryCount++;
                    if (retryCount >= maxRetries) {
                        return [3 /*break*/, 7];
                    }
                    return [4 /*yield*/, delay(delayMs)];
                case 5:
                    _a.sent();
                    delayMs *= 1.5; // Exponential backoff
                    return [3 /*break*/, 6];
                case 6: return [3 /*break*/, 1];
                case 7: throw lastError;
            }
        });
    });
}
/**
 * Converts a hex string to a UTF-8 string
 * @param hex Hex string
 * @returns UTF-8 string
 */
function hexToUtf8(hex) {
    return Buffer.from(hex.startsWith('0x') ? hex.slice(2) : hex, 'hex').toString('utf8');
}
/**
 * Converts a UTF-8 string to a hex string
 * @param str UTF-8 string
 * @returns Hex string
 */
function utf8ToHex(str) {
    return '0x' + Buffer.from(str, 'utf8').toString('hex');
}
