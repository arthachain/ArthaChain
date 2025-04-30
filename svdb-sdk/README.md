# SVDB SDK

A JavaScript SDK for interacting with the State Variable Database (SVDB).

## Installation

```bash
npm install svdb-sdk
```

## Usage

### Basic Setup

```javascript
const SvdbClient = require('svdb-sdk');

const client = new SvdbClient({
    baseUrl: 'http://your-svdb-server:8080',
    timeout: 30000, // optional, defaults to 30000ms
    encryptionKey: 'your-32-byte-encryption-key' // optional, for encrypted storage
});
```

### Basic Operations

#### Store Data
```javascript
// Store plain data
await client.store('myKey', Buffer.from('Hello, World!'));

// Store encrypted data
await client.store('myKey', Buffer.from('Sensitive Data'), { encrypted: true });
```

#### Retrieve Data
```javascript
const data = await client.retrieve('myKey');
console.log(data.toString()); // 'Hello, World!'
```

#### Delete Data
```javascript
await client.delete('myKey');
```

#### Check Existence
```javascript
const exists = await client.exists('myKey');
console.log(exists); // true or false
```

#### Verify Data
```javascript
const hash = crypto.createHash('sha256').update(data).digest('hex');
const isValid = await client.verify('myKey', hash);
console.log(isValid); // true or false
```

### Batch Operations

#### Batch Store
```javascript
await client.batchStore([
    { key: 'key1', data: Buffer.from('data1'), encrypted: true },
    { key: 'key2', data: Buffer.from('data2'), encrypted: false }
]);
```

#### Batch Retrieve
```javascript
const results = await client.batchRetrieve(['key1', 'key2']);
results.forEach(result => {
    console.log(`Key: ${result.key}, Data: ${result.data.toString()}`);
});
```

#### Batch Delete
```javascript
await client.batchDelete(['key1', 'key2']);
```

### Encryption

The SDK supports AES-256-GCM encryption for secure data storage. To use encryption:

1. Provide a 32-byte encryption key when initializing the client
2. Set the `encrypted` option to `true` when storing data
3. The SDK will automatically handle encryption/decryption

Example:
```javascript
const client = new SvdbClient({
    baseUrl: 'http://your-svdb-server:8080',
    encryptionKey: crypto.randomBytes(32) // Generate a secure key
});

// Store encrypted data
await client.store('sensitiveKey', Buffer.from('Secret Data'), { encrypted: true });

// Retrieve and automatically decrypt
const decryptedData = await client.retrieve('sensitiveKey');
```

## Error Handling

The SDK provides detailed error messages for various failure scenarios:

- Network errors
- Server errors
- Invalid data format
- Encryption/decryption errors
- Key not found errors

Example:
```javascript
try {
    await client.store('key', data);
} catch (error) {
    console.error('Error:', error.message);
}
```

## Best Practices

1. Always use encryption for sensitive data
2. Implement proper error handling
3. Use batch operations for better performance when dealing with multiple keys
4. Keep your encryption key secure and never share it
5. Use appropriate timeouts for your use case

## API Reference

### Constructor
- `new SvdbClient(config)`
  - `config.baseUrl`: Server URL (required)
  - `config.timeout`: Request timeout in ms (optional)
  - `config.encryptionKey`: 32-byte encryption key (optional)

### Methods
- `store(key, data, options)`: Store data
- `retrieve(key)`: Retrieve data
- `delete(key)`: Delete data
- `exists(key)`: Check if key exists
- `verify(key, hash)`: Verify data integrity
- `batchStore(operations)`: Store multiple items
- `batchRetrieve(keys)`: Retrieve multiple items
- `batchDelete(keys)`: Delete multiple items

## License

MIT 