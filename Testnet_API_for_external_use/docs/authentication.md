# Authentication

This document outlines the authentication mechanisms for the Artha Chain testnet API.

## API Keys

The Artha Chain testnet API uses API keys for authentication. API keys are used to track and control how the API is being used, to prevent abuse, and to manage rate limits.

### Obtaining an API Key

1. Register on the [Artha Chain Developer Portal](https://developers.arthachain.com)
2. Navigate to the "API Keys" section
3. Click "Create New API Key"
4. Provide a name for your API key and select the appropriate access level
5. Your API key will be displayed once - make sure to save it in a secure location

### Using Your API Key

Include your API key in all requests to the API using the `X-API-Key` header:

```
X-API-Key: your_api_key_here
```

Example using cURL:

```bash
curl -X GET "https://testnet-api.arthachain.com/api/v1/blocks/latest" \
  -H "X-API-Key: your_api_key_here"
```

Example using JavaScript (fetch):

```javascript
fetch('https://testnet-api.arthachain.com/api/v1/blocks/latest', {
  headers: {
    'X-API-Key': 'your_api_key_here'
  }
})
.then(response => response.json())
.then(data => console.log(data));
```

### API Key Tiers

| Tier | Rate Limit | Features |
|------|------------|----------|
| Basic | 50 requests/minute | Read-only access to blockchain data |
| Developer | 200 requests/minute | Basic + transaction submission |
| Enterprise | 1000 requests/minute | Developer + websocket subscriptions + prioritized support |

## JWT Authentication (for Advanced Services)

Some advanced services, like admin functions or developer tooling, use JWT (JSON Web Token) authentication in addition to API keys.

### Obtaining a JWT Token

1. Use the login endpoint to obtain a JWT token:

```bash
curl -X POST "https://testnet-api.arthachain.com/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "your_email@example.com", "password": "your_password"}'
```

2. The response will contain a JWT token:

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_at": 1652345678
}
```

### Using JWT Authentication

Include the JWT token in the `Authorization` header with the Bearer scheme:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Example using cURL:

```bash
curl -X GET "https://testnet-api.arthachain.com/api/v1/admin/metrics" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "X-API-Key: your_api_key_here"
```

### Token Refresh

JWT tokens expire after a period of time. To refresh your token:

```bash
curl -X POST "https://testnet-api.arthachain.com/api/v1/auth/refresh" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "X-API-Key: your_api_key_here"
```

## Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 401 | UNAUTHORIZED | Missing or invalid API key |
| 401 | INVALID_TOKEN | Invalid or expired JWT token |
| 403 | FORBIDDEN | API key does not have access to the requested resource |
| 429 | RATE_LIMIT_EXCEEDED | Rate limit exceeded for the API key |

## Security Best Practices

1. **Never share your API key or credentials**: Keep your API key and JWT tokens secure and do not expose them in client-side code.

2. **Use environment variables**: Store your API key in environment variables rather than hardcoding them.

3. **Set up IP restrictions**: In the developer portal, you can restrict API key usage to specific IP addresses.

4. **Rotate API keys periodically**: Generate new API keys regularly and deprecate old ones.

5. **Use HTTPS**: Always use HTTPS for making API requests to ensure data is encrypted in transit.

6. **Implement proper error handling**: Handle authentication errors gracefully in your application.

7. **Monitor API usage**: Regularly review your API usage in the developer portal to detect any unusual activity.

## Sample Code for Authentication

### JavaScript (Node.js)

```javascript
const axios = require('axios');

const API_KEY = process.env.ARTHA_API_KEY;
const API_BASE_URL = 'https://testnet-api.arthachain.com/api/v1';

// Setup axios instance with default headers
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'X-API-Key': API_KEY
  }
});

// Example GET request
async function getLatestBlock() {
  try {
    const response = await api.get('/blocks/latest');
    return response.data;
  } catch (error) {
    console.error('Error fetching latest block:', error.response?.data || error.message);
    throw error;
  }
}

// Example with JWT authentication
async function login(email, password) {
  try {
    const response = await api.post('/auth/login', {
      email: email,
      password: password
    });
    
    // Save token for future requests
    const token = response.data.token;
    
    // Update headers for future requests
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    
    return token;
  } catch (error) {
    console.error('Login failed:', error.response?.data || error.message);
    throw error;
  }
}
```

### Python

```python
import os
import requests

API_KEY = os.environ.get('ARTHA_API_KEY')
API_BASE_URL = 'https://testnet-api.arthachain.com/api/v1'

headers = {
    'X-API-Key': API_KEY
}

# Example GET request
def get_latest_block():
    try:
        response = requests.get(f'{API_BASE_URL}/blocks/latest', headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching latest block: {e}")
        raise

# Example with JWT authentication
def login(email, password):
    try:
        login_data = {
            'email': email,
            'password': password
        }
        response = requests.post(f'{API_BASE_URL}/auth/login', json=login_data, headers=headers)
        response.raise_for_status()
        
        # Save token for future requests
        token = response.json()['token']
        
        # Update headers for future requests
        headers['Authorization'] = f'Bearer {token}'
        
        return token
    except requests.exceptions.RequestException as e:
        print(f"Login failed: {e}")
        raise
``` 