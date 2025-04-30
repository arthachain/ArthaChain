# Rate Limiting

This document explains the rate limiting policies for the Artha Chain testnet API to ensure equitable access and service stability.

## Overview

Rate limiting is applied to prevent API abuse and ensure fair usage across all clients. Different rate limits apply based on your API key tier.

## Rate Limit Tiers

| Tier | Rate Limit | Burst Allowance | Recovery Time |
|------|------------|-----------------|---------------|
| Basic | 50 requests/minute | 60 requests | 60 seconds |
| Developer | 200 requests/minute | 220 requests | 60 seconds |
| Enterprise | 1000 requests/minute | 1100 requests | 60 seconds |

## Rate Limit Headers

Each API response includes headers that provide information about your current rate limit status:

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | The maximum number of requests you're permitted to make per minute |
| `X-RateLimit-Remaining` | The number of requests remaining in the current rate limit window |
| `X-RateLimit-Reset` | The time at which the current rate limit window resets in UTC epoch seconds |

Example response headers:

```
X-RateLimit-Limit: 200
X-RateLimit-Remaining: 195
X-RateLimit-Reset: 1619439600
```

## Rate Limit Exceeded Response

When you exceed your rate limit, the API will return a `429 Too Many Requests` HTTP status code with the following response body:

```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "You have exceeded your rate limit. Please try again after X seconds.",
  "retry_after": 30
}
```

The `retry_after` field indicates the number of seconds to wait before making another request.

## Best Practices for Rate Limit Management

### Implement Backoff Logic

Implement an exponential backoff strategy in your client applications. When you receive a 429 status code, wait for the duration specified in the `retry_after` field before retrying, and increase the wait time for consecutive failures.

```javascript
async function makeRequestWithBackoff(url, options, maxRetries = 5) {
  let retries = 0;
  
  while (retries < maxRetries) {
    try {
      const response = await fetch(url, options);
      
      if (response.status !== 429) {
        return response;
      }
      
      const data = await response.json();
      const retryAfter = data.retry_after || Math.pow(2, retries);
      console.log(`Rate limited. Retrying after ${retryAfter} seconds...`);
      
      await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
      retries++;
      
    } catch (error) {
      console.error('Request failed:', error);
      throw error;
    }
  }
  
  throw new Error('Maximum retries reached');
}
```

### Cache Responses

Cache responses when appropriate to reduce the number of API calls:

```javascript
const cache = new Map();

async function fetchWithCache(url, options, cacheDurationMs = 60000) {
  if (cache.has(url) && cache.get(url).expiry > Date.now()) {
    return cache.get(url).data;
  }
  
  const response = await fetch(url, options);
  const data = await response.json();
  
  cache.set(url, {
    data,
    expiry: Date.now() + cacheDurationMs
  });
  
  return data;
}
```

### Batch Requests

Where possible, use batch operations to reduce the number of API calls:

```
GET /api/v1/transactions/batch?ids=tx1,tx2,tx3,tx4,tx5
```

instead of:

```
GET /api/v1/transactions/tx1
GET /api/v1/transactions/tx2
GET /api/v1/transactions/tx3
GET /api/v1/transactions/tx4
GET /api/v1/transactions/tx5
```

### Monitor Your Usage

Monitor your rate limit usage through the provided headers and adjust your request patterns accordingly. Consider implementing alerts when approaching your limits.

### Request Rate Limit Increase

If you consistently reach your rate limits, consider:

1. Upgrading to a higher API key tier
2. Optimizing your code to reduce unnecessary API calls
3. Contacting support for a custom rate limit solution for special use cases

## Webhooks and WebSockets

Webhooks and WebSocket connections have separate rate limits:

### Webhooks
- **Event delivery**: 10 events/second per endpoint
- **Failed delivery retries**: 5 attempts with exponential backoff

### WebSockets
- **Connection limit**: 10 concurrent connections per API key
- **Message rate**: 100 messages/minute per connection

## Endpoint-Specific Rate Limits

Some endpoints may have custom rate limits that differ from the general limits:

| Endpoint | Rate Limit | Explanation |
|----------|------------|-------------|
| `/api/v1/transactions/submit` | 10 per minute (Basic)<br>50 per minute (Developer)<br>200 per minute (Enterprise) | Higher resource cost for transaction submission |
| `/api/v1/blocks/height/:height` | 100 per minute (all tiers) | Higher cache hit ratio allows higher limits |
| `/api/v1/search` | 20 per minute (Basic)<br>50 per minute (Developer)<br>100 per minute (Enterprise) | Resource-intensive search operations |

## Quota System for High-Value Endpoints

Some high-value or resource-intensive endpoints use a quota system instead of a simple rate limit:

| Endpoint | Daily Quota | Refill Rate |
|----------|-------------|-------------|
| `/api/v1/analytics/address-activity` | 500 calls (Basic)<br>2000 calls (Developer)<br>10000 calls (Enterprise) | Resets at 00:00 UTC |
| `/api/v1/contracts/analyze` | 100 calls (Basic)<br>500 calls (Developer)<br>2000 calls (Enterprise) | Resets at 00:00 UTC |

## Contact Information

For any questions regarding rate limits or to request a custom rate limit solution, please contact:

- Email: api-support@arthachain.com
- Support portal: https://developers.arthachain.com/support 