# Error Handling

This document describes the error handling approach used by the Artha Chain testnet API.

## Error Response Format

All API errors follow a consistent JSON format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      // Optional: Additional context-specific information
    },
    "request_id": "unique-request-identifier"
  }
}
```

- **code**: A machine-readable error code string
- **message**: A human-readable description of the error
- **details**: Optional object containing additional error context
- **request_id**: A unique identifier for the request, useful for troubleshooting

## HTTP Status Codes

The API uses standard HTTP status codes to indicate the success or failure of requests:

| Status Code | Description |
|-------------|-------------|
| 200 OK | The request was successful |
| 201 Created | The resource was successfully created |
| 400 Bad Request | The request was invalid or cannot be served |
| 401 Unauthorized | Authentication is required or has failed |
| 403 Forbidden | The authenticated user lacks permission for the requested resource |
| 404 Not Found | The requested resource does not exist |
| 409 Conflict | The request conflicts with the current state of the resource |
| 422 Unprocessable Entity | The request was well-formed but contains semantic errors |
| 429 Too Many Requests | Rate limit exceeded (see rate-limiting.md) |
| 500 Internal Server Error | An unexpected error occurred on the server |
| 503 Service Unavailable | The service is temporarily unavailable |

## Common Error Codes

Below are common error codes you may encounter when using the API:

### Authentication Errors

| Error Code | Description |
|------------|-------------|
| `INVALID_API_KEY` | The provided API key is invalid |
| `EXPIRED_API_KEY` | The provided API key has expired |
| `MISSING_API_KEY` | No API key was provided in the request |
| `INVALID_JWT` | The provided JWT token is invalid or malformed |
| `EXPIRED_JWT` | The provided JWT token has expired |

### Request Errors

| Error Code | Description |
|------------|-------------|
| `INVALID_REQUEST` | The request is malformed or missing required parameters |
| `INVALID_PARAMETER` | One or more parameters have invalid values |
| `MISSING_PARAMETER` | A required parameter is missing |
| `UNSUPPORTED_MEDIA_TYPE` | The request content type is not supported |
| `RATE_LIMIT_EXCEEDED` | You have exceeded your rate limit |

### Resource Errors

| Error Code | Description |
|------------|-------------|
| `RESOURCE_NOT_FOUND` | The requested resource does not exist |
| `RESOURCE_ALREADY_EXISTS` | Attempting to create a resource that already exists |
| `RESOURCE_CONFLICT` | The request conflicts with the current state of the resource |

### Transaction Errors

| Error Code | Description |
|------------|-------------|
| `TX_VALIDATION_FAILED` | Transaction validation failed |
| `TX_INSUFFICIENT_FUNDS` | Insufficient funds for the transaction |
| `TX_NONCE_ERROR` | Invalid transaction nonce |
| `TX_GAS_TOO_LOW` | Gas price or limit is too low |
| `TX_REJECTED` | Transaction was rejected by the network |

### Node Errors

| Error Code | Description |
|------------|-------------|
| `NODE_SYNC_ERROR` | The node is still syncing with the network |
| `NODE_UNAVAILABLE` | The requested node is temporarily unavailable |
| `BLOCKCHAIN_ERROR` | An error occurred in the blockchain |

### Contract Errors

| Error Code | Description |
|------------|-------------|
| `CONTRACT_EXECUTION_FAILED` | Smart contract execution failed |
| `CONTRACT_DEPLOYMENT_FAILED` | Smart contract deployment failed |
| `CONTRACT_NOT_FOUND` | The requested contract does not exist |
| `CONTRACT_ABI_ERROR` | Error in contract ABI |

## Error Details

The `details` field provides additional information specific to the error:

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Invalid parameter value",
    "details": {
      "parameter": "address",
      "value": "0xinvalid",
      "constraint": "Must be a valid 42-character hexadecimal address"
    },
    "request_id": "req-123456"
  }
}
```

## Handling Errors in Client Applications

### Best Practices

1. **Always check the HTTP status code** first to determine the general category of the error.
2. **Parse the error code** to handle specific error conditions programmatically.
3. **Log the request_id** for troubleshooting purposes.
4. **Implement appropriate retry logic** for transient errors (e.g., 429, 503).
5. **Display user-friendly error messages** based on the provided error message.

### Example Error Handling (JavaScript)

```javascript
async function makeApiRequest(endpoint, options) {
  try {
    const response = await fetch(`https://api.testnet.arthachain.com/v1${endpoint}`, options);
    
    if (!response.ok) {
      const errorData = await response.json();
      
      // Log for debugging
      console.error(`API Error (${errorData.error.request_id}):`, errorData);
      
      // Handle specific error cases
      switch (errorData.error.code) {
        case 'RATE_LIMIT_EXCEEDED':
          // Implement retry with backoff
          const retryAfter = parseInt(response.headers.get('Retry-After') || '30');
          return new Promise(resolve => setTimeout(() => 
            resolve(makeApiRequest(endpoint, options)), retryAfter * 1000));
          
        case 'INVALID_API_KEY':
        case 'EXPIRED_API_KEY':
          // Trigger authentication flow
          await refreshAuthentication();
          return makeApiRequest(endpoint, options);
          
        case 'TX_INSUFFICIENT_FUNDS':
          // Show user-friendly message
          throw new UserFacingError('Insufficient funds for this transaction.');
          
        default:
          // Generic error handling
          throw new Error(errorData.error.message || 'An unexpected error occurred');
      }
    }
    
    return await response.json();
  } catch (error) {
    // Handle network errors
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      throw new Error('Network error. Please check your connection.');
    }
    throw error;
  }
}
```

### Example Error Handling (Python)

```python
import requests
import time
import logging

logger = logging.getLogger(__name__)

def make_api_request(endpoint, **kwargs):
    url = f"https://api.testnet.arthachain.com/v1{endpoint}"
    
    try:
        response = requests.request(method=kwargs.get('method', 'GET'), url=url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        error_data = e.response.json().get('error', {})
        request_id = error_data.get('request_id', 'unknown')
        
        # Log the error with request ID
        logger.error(f"API Error ({request_id}): {error_data}")
        
        # Handle specific error types
        error_code = error_data.get('code')
        
        if error_code == 'RATE_LIMIT_EXCEEDED':
            retry_after = int(e.response.headers.get('Retry-After', 30))
            logger.info(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            return make_api_request(endpoint, **kwargs)
            
        elif error_code in ('INVALID_JWT', 'EXPIRED_JWT'):
            # Refresh authentication
            refresh_auth_token()
            if 'headers' in kwargs:
                kwargs['headers']['Authorization'] = f"Bearer {get_current_token()}"
            return make_api_request(endpoint, **kwargs)
            
        # Re-raise with more context
        raise Exception(f"{error_data.get('message', 'Unknown API error')}")
        
    except requests.exceptions.ConnectionError:
        raise Exception("Network error. Please check your connection.")
```

## Validation Errors

For validation errors, the `details` field provides information about each validation failure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "fields": [
        {
          "field": "amount",
          "message": "Must be a positive number",
          "value": -10
        },
        {
          "field": "recipient",
          "message": "Must be a valid address",
          "value": "0xinvalid"
        }
      ]
    },
    "request_id": "req-123456"
  }
}
```

## Batch Requests and Partial Errors

For batch requests, the API may return partial success with errors for specific items:

```json
{
  "results": [
    { "id": "tx1", "status": "success", "data": { /* result */ } },
    { 
      "id": "tx2", 
      "status": "error", 
      "error": {
        "code": "TX_VALIDATION_FAILED",
        "message": "Transaction validation failed"
      }
    },
    { "id": "tx3", "status": "success", "data": { /* result */ } }
  ],
  "summary": {
    "total": 3,
    "succeeded": 2,
    "failed": 1
  }
}
```

## Getting Help

If you encounter persistent errors or need assistance troubleshooting API issues:

1. Include the `request_id` in all support communications
2. Contact the API support team at api-support@arthachain.com
3. Check the developer forums at https://developers.arthachain.com/forum 