# Faucet API

The testnet provides a faucet API that allows developers to request test tokens for their accounts. This is useful for testing applications without needing real tokens.

## Request Tokens

**Endpoint:** `POST /api/faucet/request`

**Request:**
```json
{
  "recipient": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7"
}
```

**Response:**
```json
{
  "status": "success",
  "transaction_hash": "0x3bcd7f317c74c536bdebffc44a93d33eadbc81a59a4d93e7e0c7f0f9e5b5a14a",
  "amount": 1000,
  "recipient": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7"
}
```

## Rate Limits

The faucet has the following rate limits to prevent abuse:

- Maximum 5 requests per IP address per day
- Maximum 3 requests per account address per day
- Cooldown period of 1 hour between requests

If you exceed these limits, you'll receive an error response:

```json
{
  "status": "error",
  "message": "Rate limit exceeded. Please try again in 3540 seconds.",
  "cooldown_remaining": 3540
}
```

## Error Codes

| HTTP Status | Error | Description |
|-------------|-------|-------------|
| 400 | INVALID_ADDRESS | The recipient address is invalid |
| 429 | RATE_LIMITED | Rate limit exceeded for IP or address |
| 503 | FAUCET_EMPTY | The faucet has run out of tokens |
| 500 | INTERNAL_ERROR | An internal error occurred while processing the request |

## Example Usage

### JavaScript (Fetch API)

```javascript
async function requestTokens(address) {
  try {
    const response = await fetch('http://localhost:3000/api/faucet/request', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        recipient: address
      })
    });
    
    const data = await response.json();
    
    if (response.ok) {
      console.log(`Successfully requested tokens. Transaction hash: ${data.transaction_hash}`);
      return data;
    } else {
      console.error(`Error requesting tokens: ${data.message}`);
      throw new Error(data.message);
    }
  } catch (error) {
    console.error('Failed to request tokens:', error);
    throw error;
  }
}
```

### Python (Requests)

```python
import requests

def request_tokens(address):
    url = 'http://localhost:3000/api/faucet/request'
    payload = {
        'recipient': address
    }
    
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        
        if response.status_code == 200:
            print(f"Successfully requested tokens. Transaction hash: {data['transaction_hash']}")
            return data
        else:
            print(f"Error requesting tokens: {data['message']}")
            raise Exception(data['message'])
    except Exception as e:
        print(f"Failed to request tokens: {str(e)}")
        raise
```

### cURL

```bash
curl -X POST \
  http://localhost:3000/api/faucet/request \
  -H 'Content-Type: application/json' \
  -d '{
    "recipient": "0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7"
}'
```

## Checking Balance

After requesting tokens, you can check your balance using the account endpoint:

```
GET /api/accounts/0xf317c74c536bdebffc44a93d33eadbc81a59a4d93e7
```

The balance field in the response will show your current token balance.

## Faucet Configuration

The testnet faucet is configured with the following parameters:

- Default amount per request: 1000 tokens
- Faucet address: Located in the genesis configuration
- Automatic refill: The faucet is automatically refilled when its balance falls below a threshold

## Best Practices

1. Only request tokens when needed for testing
2. If you need large amounts of tokens for specific testing scenarios, contact the testnet administrators
3. Release unused tokens back to the faucet when possible by sending them to the faucet address 